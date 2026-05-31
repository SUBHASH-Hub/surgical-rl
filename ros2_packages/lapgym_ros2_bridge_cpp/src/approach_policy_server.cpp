/**
 * Phase 4F: ApproachPolicyServer (C++)
 *
 * ROS 2 action server in C++ that navigates the surgical instrument
 * to the grasping zone using a proportional controller.
 *
 * Calls the Python SofaStepService via /sofa_step ROS 2 service
 * for each physics step — keeping SOFA Python bindings in Python
 * while all ROS 2 action/stop logic runs in C++.
 *
 * Demonstrates:
 *   - rclcpp action server with goal/cancel/execute callbacks
 *   - std::atomic<bool> for thread-safe surgeon stop flag
 *   - std::thread for background stop listener (same pattern as Python
 *     separate rclpy.Context, but in C++ with a dedicated executor)
 *   - ROS 2 service client for SOFA physics bridge
 *   - MultiThreadedExecutor for concurrent callback groups
 *   - C++17 features: std::array, structured bindings
 *
 * Author: Subhash Arockiadoss
 */

#include <chrono>
#include <memory>
#include <thread>
#include <atomic>
#include <array>
#include <cmath>
#include <string>
#include <functional>
#include <future>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/bool.hpp"

#include "lapgym_interfaces/action/retract.hpp"
#include "lapgym_interfaces/srv/sofa_step.hpp"

// Fixed grasping target in world metres (from Phase 4B analysis)
static constexpr std::array<float, 3> GRASPING_TARGET = {
    -0.0485583f, 0.0085f, 0.0356076f
};
static constexpr float APPROACH_THRESHOLD = 0.025f;  // 25mm
static constexpr float APPROACH_GAIN      = 2.0f;
static constexpr int   DEFAULT_MAX_STEPS  = 400;

using Retract           = lapgym_interfaces::action::Retract;
using SofaStep          = lapgym_interfaces::srv::SofaStep;
using GoalHandleRetract = rclcpp_action::ServerGoalHandle<Retract>;


class ApproachPolicyServerCpp : public rclcpp::Node
{
public:
    ApproachPolicyServerCpp()
    : Node("approach_policy_server_cpp"),
      surgeon_stopped_(false),
      emergency_(false)
    {
        // -- Action server -------------------------------------------------
        action_server_ = rclcpp_action::create_server<Retract>(
            this,
            "approach_policy_cpp",
            std::bind(&ApproachPolicyServerCpp::handleGoal,    this,
                      std::placeholders::_1, std::placeholders::_2),
            std::bind(&ApproachPolicyServerCpp::handleCancel,  this,
                      std::placeholders::_1),
            std::bind(&ApproachPolicyServerCpp::handleAccepted, this,
                      std::placeholders::_1)
        );

        // -- SOFA step service client ---------------------------------------
        // Dedicated callback group so service responses can be processed
        // while the action execute() loop is running on another thread.
        // Without this separate group, the executor cannot process the
        // service response while blocked inside execute() — deadlock.
        client_callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        sofa_client_ = create_client<SofaStep>(
            "/sofa_step",
            rmw_qos_profile_services_default,
            client_callback_group_);

        RCLCPP_INFO(get_logger(), "Waiting for /sofa_step service...");
        while (!sofa_client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(get_logger(), "Interrupted while waiting");
                return;
            }
            RCLCPP_INFO(get_logger(), "Still waiting for /sofa_step...");
        }
        RCLCPP_INFO(get_logger(), "/sofa_step service ready");

        // -- Surgeon stop subscriber on dedicated background thread --------
        // Same pattern as Python separate rclpy.Context:
        // The execute() loop blocks on sofaStep() service calls ~65ms.
        // We need stop callbacks to fire DURING that blocking period.
        // Solution: separate callback group + dedicated SingleThreadedExecutor
        // spinning on its own std::thread — completely independent of main executor.
        stop_callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions stop_opts;
        stop_opts.callback_group = stop_callback_group_;

        surgeon_stop_sub_ = create_subscription<std_msgs::msg::Bool>(
            "/surgeon_stop", 10,
            std::bind(&ApproachPolicyServerCpp::surgeonStopCb, this,
                      std::placeholders::_1),
            stop_opts);

        emergency_sub_ = create_subscription<std_msgs::msg::Bool>(
            "/emergency_stop", 10,
            std::bind(&ApproachPolicyServerCpp::emergencyCb, this,
                      std::placeholders::_1),
            stop_opts);

        // Spin stop subscriptions on dedicated background thread
        stop_executor_ =
            std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        stop_executor_->add_callback_group(
            stop_callback_group_, get_node_base_interface());
        stop_thread_ = std::thread([this]() {
            stop_executor_->spin();
        });

        RCLCPP_INFO(get_logger(),
            "ApproachPolicyServerCpp ready on /approach_policy_cpp");
    }

    ~ApproachPolicyServerCpp()
    {
        if (stop_executor_) {
            stop_executor_->cancel();
        }
        if (stop_thread_.joinable()) {
            stop_thread_.join();
        }
    }

private:
    // -----------------------------------------------------------------------
    // Action server callbacks
    // -----------------------------------------------------------------------

    rclcpp_action::GoalResponse handleGoal(
        const rclcpp_action::GoalUUID & /*uuid*/,
        std::shared_ptr<const Retract::Goal> /*goal*/)
    {
        RCLCPP_INFO(get_logger(), "Goal received — accepting");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handleCancel(
        const std::shared_ptr<GoalHandleRetract> /*goal_handle*/)
    {
        RCLCPP_INFO(get_logger(), "Cancel request — accepting");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handleAccepted(const std::shared_ptr<GoalHandleRetract> goal_handle)
    {
        // Detach execution onto its own thread.
        // This is required with MultiThreadedExecutor so the execute loop
        // does not block the executor's callback processing threads.
        std::thread([this, goal_handle]() {
            execute(goal_handle);
        }).detach();
    }

    // -----------------------------------------------------------------------
    // Main execution loop
    // -----------------------------------------------------------------------

    void execute(const std::shared_ptr<GoalHandleRetract> goal_handle)
    {
        RCLCPP_INFO(get_logger(), "Executing approach policy (C++)");

        auto goal     = goal_handle->get_goal();
        int max_steps = (goal->max_steps > 0)
                        ? static_cast<int>(goal->max_steps)
                        : DEFAULT_MAX_STEPS;

        // Reset SOFA environment at start of episode
        sofaReset();

        auto feedback        = std::make_shared<Retract::Feedback>();
        int  step            = 0;
        float final_distance = 0.0f;
        std::string termination = "timeout";

        while (step < max_steps) {

            // -- Preemption check FIRST every step -------------------------
            if (goal_handle->is_canceling()) {
                auto result        = std::make_shared<Retract::Result>();
                result->success        = false;
                result->steps_taken    = step;
                result->final_distance = final_distance;
                result->termination    = "preempted";
                goal_handle->canceled(result);
                RCLCPP_INFO(get_logger(), "Goal preempted at step %d", step);
                return;
            }

            // -- Emergency stop check --------------------------------------
            if (emergency_.load()) {
                auto result            = std::make_shared<Retract::Result>();
                result->success        = false;
                result->steps_taken    = step;
                result->final_distance = final_distance;
                result->termination    = "emergency_stop";
                try { goal_handle->abort(result); } catch (...) {}
                RCLCPP_ERROR(get_logger(),
                    "EMERGENCY STOP at step %d", step);
                return;
            }

            // -- Surgeon stop freeze loop ----------------------------------
            // C++ equivalent of Python's stop_event.wait(timeout=0.05)
            // stop_callback_group_ processes /surgeon_stop independently
            // so surgeon_stopped_ updates even while we sleep here
            while (surgeon_stopped_.load() && !emergency_.load()) {
                if (goal_handle->is_canceling()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            // -- Get tool world position — zero-action probe step ----------
            auto state   = sofaStep({0.0f, 0.0f, 0.0f});
            float tool_x = state->tool_world_x;
            float tool_y = state->tool_world_y;
            float tool_z = state->tool_world_z;

            // -- Compute distance to grasping target -----------------------
            float ex   = GRASPING_TARGET[0] - tool_x;
            float ey   = GRASPING_TARGET[1] - tool_y;
            float ez   = GRASPING_TARGET[2] - tool_z;
            float dist = std::sqrt(ex*ex + ey*ey + ez*ez);
            final_distance = dist;

            // -- Check if close enough -------------------------------------
            if (dist < APPROACH_THRESHOLD) {
                termination = "goal_reached";
                RCLCPP_INFO(get_logger(),
                    "Approach complete at step %d dist=%.1fmm",
                    step, dist * 1000.0f);
                break;
            }

            // -- Proportional controller -----------------------------------
            float norm = dist + 1e-8f;
            float ax = std::max(-3.0f, std::min(3.0f,
                        (ex / norm) * APPROACH_GAIN));
            float ay = std::max(-3.0f, std::min(3.0f,
                        (ey / norm) * APPROACH_GAIN));
            float az = std::max(-3.0f, std::min(3.0f,
                        (ez / norm) * APPROACH_GAIN));

            // -- Step SOFA physics with computed action --------------------
            auto result_step = sofaStep({ax, ay, az});
            step++;

            // -- Post-step surgeon stop freeze loop ------------------------
            // Catches stop pressed DURING the sofaStep() service call
            while (surgeon_stopped_.load() && !emergency_.load()) {
                if (goal_handle->is_canceling()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            // -- Publish feedback every step -------------------------------
            feedback->distance_to_goal = dist;
            feedback->distance_mm      = dist * 1000.0f;
            feedback->step             = step;
            feedback->in_collision     = result_step->in_collision;
            feedback->collision_cost   = result_step->collision_cost;
            goal_handle->publish_feedback(feedback);

            if (step % 5 == 0) {
                RCLCPP_INFO(get_logger(),
                    "Approach (C++) step %3d | Dist: %.1fmm",
                    step, dist * 1000.0f);
            }

            if (result_step->terminated) {
                termination = "collision";
                break;
            }
        }

        // -- Build and return result ---------------------------------------
        bool success     = (termination == "goal_reached");
        auto result      = std::make_shared<Retract::Result>();
        result->success        = success;
        result->steps_taken    = step;
        result->final_distance = final_distance;
        result->termination    = termination;
        goal_handle->succeed(result);

        RCLCPP_INFO(get_logger(),
            "Approach complete: %s steps=%d dist=%.1fmm",
            termination.c_str(), step, final_distance * 1000.0f);
    }

    // -----------------------------------------------------------------------
    // SOFA service helpers
    // -----------------------------------------------------------------------

    void sofaReset()
    {
        auto request    = std::make_shared<SofaStep::Request>();
        request->action = {0.0f, 0.0f, 0.0f};
        request->reset  = true;

        auto future = sofa_client_->async_send_request(request);

        // future.wait_for() blocks this thread only — does NOT touch the
        // executor. The MultiThreadedExecutor processes the service response
        // on the client_callback_group_ thread while we wait here.
        auto status = future.wait_for(std::chrono::seconds(10));
        if (status != std::future_status::ready) {
            RCLCPP_ERROR(get_logger(), "SOFA reset timed out");
            return;
        }
        RCLCPP_INFO(get_logger(), "SOFA environment reset via service");
    }

    std::shared_ptr<SofaStep::Response> sofaStep(
        std::array<float, 3> action)
    {
        auto request    = std::make_shared<SofaStep::Request>();
        request->action = action;
        request->reset  = false;

        auto future = sofa_client_->async_send_request(request);

        // Same pattern as sofaReset — future.wait_for() does not block
        // the executor. Response arrives on client_callback_group_ thread.
        auto status = future.wait_for(std::chrono::seconds(5));
        if (status != std::future_status::ready) {
            RCLCPP_ERROR(get_logger(), "SOFA step timed out");
            return std::make_shared<SofaStep::Response>();
        }
        return future.get();
    }

    // -----------------------------------------------------------------------
    // Stop / emergency callbacks — run on stop_callback_group_ background thread
    // -----------------------------------------------------------------------

    void surgeonStopCb(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data) {
            surgeon_stopped_.store(true);
            RCLCPP_WARN(get_logger(),
                "Approach (C++): SURGEON STOP received");
        } else {
            surgeon_stopped_.store(false);
            RCLCPP_INFO(get_logger(),
                "Approach (C++): SURGEON RESUME received");
        }
    }

    void emergencyCb(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data && !emergency_.load()) {
            emergency_.store(true);
            RCLCPP_ERROR(get_logger(),
                "Approach (C++): EMERGENCY STOP received");
        }
    }

    // -----------------------------------------------------------------------
    // Member variables
    // -----------------------------------------------------------------------

    // Action server
    rclcpp_action::Server<Retract>::SharedPtr action_server_;

    // SOFA service client + its dedicated callback group
    rclcpp::Client<SofaStep>::SharedPtr        sofa_client_;
    rclcpp::CallbackGroup::SharedPtr           client_callback_group_;

    // Thread-safe stop flags — atomic so execute() thread and
    // stop_callback_group_ thread can read/write without a mutex
    std::atomic<bool> surgeon_stopped_;
    std::atomic<bool> emergency_;

    // Stop listener — dedicated callback group + background executor + thread
    // Mirrors the Python separate rclpy.Context pattern
    rclcpp::CallbackGroup::SharedPtr                             stop_callback_group_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr         surgeon_stop_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr         emergency_sub_;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor>   stop_executor_;
    std::thread                                                  stop_thread_;
};


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ApproachPolicyServerCpp>();

    // MultiThreadedExecutor is required here for three reasons:
    // 1. Action execute() runs in its own detached thread — needs concurrent
    //    callback processing so feedback publish() works during execution
    // 2. client_callback_group_ must be able to process /sofa_step responses
    //    while execute() is blocked waiting on future.wait_for()
    // 3. stop_callback_group_ spins on its own thread — MultiThreadedExecutor
    //    allows this without conflicting with the main executor
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}