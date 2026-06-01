/**
 * Phase 4G: HoldPolicyServer (C++)
 *
 * Holds instrument at current position by sending zero-delta actions.
 * Simplest C++ action server — no ML, no PPO, pure control logic.
 *
 * Demonstrates same safety patterns as approach_policy_server.cpp:
 *   - std::atomic<bool> for surgeon stop and emergency flags
 *   - Separate callback group for stop listener
 *   - MultiThreadedExecutor for concurrent execution
 *   - Dual freeze loop (before + after sofa step)
 *
 * Author: Subhash Arockiadoss
 */

#include <chrono>
#include <memory>
#include <thread>
#include <atomic>
#include <array>
#include <string>
#include <functional>
#include <future>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/bool.hpp"

#include "lapgym_interfaces/action/retract.hpp"
#include "lapgym_interfaces/srv/sofa_step.hpp"

static constexpr int DEFAULT_MAX_STEPS = 500;

using Retract           = lapgym_interfaces::action::Retract;
using SofaStep          = lapgym_interfaces::srv::SofaStep;
using GoalHandleRetract = rclcpp_action::ServerGoalHandle<Retract>;


class HoldPolicyServerCpp : public rclcpp::Node
{
public:
    HoldPolicyServerCpp()
    : Node("hold_policy_server_cpp"),
      surgeon_stopped_(false),
      emergency_(false)
    {
        // -- Action server -------------------------------------------------
        action_server_ = rclcpp_action::create_server<Retract>(
            this,
            "hold_policy_cpp",
            std::bind(&HoldPolicyServerCpp::handleGoal,     this,
                      std::placeholders::_1, std::placeholders::_2),
            std::bind(&HoldPolicyServerCpp::handleCancel,   this,
                      std::placeholders::_1),
            std::bind(&HoldPolicyServerCpp::handleAccepted, this,
                      std::placeholders::_1)
        );

        // -- SOFA step service client ---------------------------------------
        client_callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        sofa_client_ = create_client<SofaStep>(
            "/sofa_step",
            rmw_qos_profile_services_default,
            client_callback_group_);

        RCLCPP_INFO(get_logger(), "Waiting for /sofa_step service...");
        while (!sofa_client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) return;
        }
        RCLCPP_INFO(get_logger(), "/sofa_step service ready");

        // -- Stop listener on background thread ----------------------------
        stop_callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions stop_opts;
        stop_opts.callback_group = stop_callback_group_;

        surgeon_stop_sub_ = create_subscription<std_msgs::msg::Bool>(
            "/surgeon_stop", 10,
            std::bind(&HoldPolicyServerCpp::surgeonStopCb, this,
                      std::placeholders::_1),
            stop_opts);

        emergency_sub_ = create_subscription<std_msgs::msg::Bool>(
            "/emergency_stop", 10,
            std::bind(&HoldPolicyServerCpp::emergencyCb, this,
                      std::placeholders::_1),
            stop_opts);

        stop_executor_ =
            std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        stop_executor_->add_callback_group(
            stop_callback_group_, get_node_base_interface());
        stop_thread_ = std::thread([this]() {
            stop_executor_->spin();
        });

        RCLCPP_INFO(get_logger(),
            "HoldPolicyServerCpp ready on /hold_policy_cpp");
    }

    ~HoldPolicyServerCpp()
    {
        if (stop_executor_) stop_executor_->cancel();
        if (stop_thread_.joinable()) stop_thread_.join();
    }

private:
    // -----------------------------------------------------------------------
    // Action callbacks
    // -----------------------------------------------------------------------

    rclcpp_action::GoalResponse handleGoal(
        const rclcpp_action::GoalUUID & /*uuid*/,
        std::shared_ptr<const Retract::Goal> /*goal*/)
    {
        RCLCPP_INFO(get_logger(), "Hold goal received — accepting");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handleCancel(
        const std::shared_ptr<GoalHandleRetract> /*goal_handle*/)
    {
        RCLCPP_INFO(get_logger(), "Hold cancel — accepting");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handleAccepted(const std::shared_ptr<GoalHandleRetract> goal_handle)
    {
        std::thread([this, goal_handle]() {
            execute(goal_handle);
        }).detach();
    }

    // -----------------------------------------------------------------------
    // Main execution loop — zero actions, hold position
    // -----------------------------------------------------------------------

    void execute(const std::shared_ptr<GoalHandleRetract> goal_handle)
    {
        RCLCPP_INFO(get_logger(), "Hold policy active — holding position");

        auto goal     = goal_handle->get_goal();
        int max_steps = (goal->max_steps > 0)
                        ? static_cast<int>(goal->max_steps)
                        : DEFAULT_MAX_STEPS;

        auto feedback = std::make_shared<Retract::Feedback>();
        int  step     = 0;

        while (step < max_steps) {

            // -- Preemption check ------------------------------------------
            if (goal_handle->is_canceling()) {
                auto result            = std::make_shared<Retract::Result>();
                result->success        = true;   // hold always succeeds
                result->steps_taken    = step;
                result->final_distance = 0.0f;
                result->termination    = "preempted";
                goal_handle->canceled(result);
                RCLCPP_INFO(get_logger(), "Hold preempted at step %d", step);
                return;
            }

            // -- Emergency stop --------------------------------------------
            if (emergency_.load()) {
                auto result            = std::make_shared<Retract::Result>();
                result->success        = false;
                result->steps_taken    = step;
                result->final_distance = 0.0f;
                result->termination    = "emergency_stop";
                try { goal_handle->abort(result); } catch (...) {}
                RCLCPP_ERROR(get_logger(),
                    "Hold EMERGENCY STOP at step %d", step);
                return;
            }

            // -- Surgeon stop freeze loop ----------------------------------
            while (surgeon_stopped_.load() && !emergency_.load()) {
                if (goal_handle->is_canceling()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            // -- Send zero action — hold position --------------------------
            // Zero delta means instrument does not move
            // SOFA physics still advances but position unchanged
            sofaStep({0.0f, 0.0f, 0.0f});
            step++;

            // -- Post-step freeze loop -------------------------------------
            while (surgeon_stopped_.load() && !emergency_.load()) {
                if (goal_handle->is_canceling()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            // -- Feedback every step ---------------------------------------
            feedback->distance_to_goal = 0.0f;
            feedback->distance_mm      = 0.0f;
            feedback->step             = step;
            feedback->in_collision     = false;
            feedback->collision_cost   = 0.0f;
            goal_handle->publish_feedback(feedback);

            if (step % 5 == 0) {
                RCLCPP_INFO(get_logger(),
                    "Holding (C++) step %3d", step);
            }

            // Hold at 10Hz — no need for full speed
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // -- Timeout — normal completion -----------------------------------
        auto result            = std::make_shared<Retract::Result>();
        result->success        = true;
        result->steps_taken    = step;
        result->final_distance = 0.0f;
        result->termination    = "timeout";
        goal_handle->succeed(result);
        RCLCPP_INFO(get_logger(), "Hold timeout reached at step %d", step);
    }

    // -----------------------------------------------------------------------
    // SOFA service helper
    // -----------------------------------------------------------------------

    void sofaStep(std::array<float, 3> action)
    {
        auto request    = std::make_shared<SofaStep::Request>();
        request->action = action;
        request->reset  = false;

        auto future = sofa_client_->async_send_request(request);
        future.wait_for(std::chrono::seconds(5));
        // result not needed for hold — we only care that step happened
    }

    // -----------------------------------------------------------------------
    // Stop callbacks
    // -----------------------------------------------------------------------

    void surgeonStopCb(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data) {
            surgeon_stopped_.store(true);
            RCLCPP_WARN(get_logger(), "Hold (C++): SURGEON STOP received");
        } else {
            surgeon_stopped_.store(false);
            RCLCPP_INFO(get_logger(), "Hold (C++): SURGEON RESUME received");
        }
    }

    void emergencyCb(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data && !emergency_.load()) {
            emergency_.store(true);
            RCLCPP_ERROR(get_logger(),
                "Hold (C++): EMERGENCY STOP received");
        }
    }

    // -----------------------------------------------------------------------
    // Member variables
    // -----------------------------------------------------------------------

    rclcpp_action::Server<Retract>::SharedPtr                    action_server_;
    rclcpp::Client<SofaStep>::SharedPtr                          sofa_client_;
    rclcpp::CallbackGroup::SharedPtr                             client_callback_group_;
    std::atomic<bool>                                            surgeon_stopped_;
    std::atomic<bool>                                            emergency_;
    rclcpp::CallbackGroup::SharedPtr                             stop_callback_group_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr         surgeon_stop_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr         emergency_sub_;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor>   stop_executor_;
    std::thread                                                  stop_thread_;
};


int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HoldPolicyServerCpp>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}