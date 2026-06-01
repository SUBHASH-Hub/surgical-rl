/**
 * Phase 4G: RetractPolicyServer (C++)
 *
 * ROS 2 action server in C++ that retracts tissue using the Phase 2D
 * PPO policy. Demonstrates the complete hybrid C++/Python pattern:
 *
 *   C++ handles:  action server, threading, stop flags, control loop
 *   Python handles: PPO inference (/ppo_predict), SOFA physics (/sofa_step)
 *
 * Data flow per step:
 *   obs[7] → /ppo_predict → action[3] → /sofa_step → new_obs, dist
 *   repeat until dist < threshold or terminated
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
#include "lapgym_interfaces/srv/ppo_predict.hpp"

static constexpr float GOAL_THRESHOLD  = 0.03f;   // 30mm
static constexpr int   DEFAULT_MAX_STEPS = 300;

using Retract           = lapgym_interfaces::action::Retract;
using SofaStep          = lapgym_interfaces::srv::SofaStep;
using PPOPredict        = lapgym_interfaces::srv::PPOPredict;
using GoalHandleRetract = rclcpp_action::ServerGoalHandle<Retract>;


class RetractPolicyServerCpp : public rclcpp::Node
{
public:
    RetractPolicyServerCpp()
    : Node("retract_policy_server_cpp"),
      surgeon_stopped_(false),
      emergency_(false)
    {
        // -- Action server -------------------------------------------------
        action_server_ = rclcpp_action::create_server<Retract>(
            this,
            "retract_policy_cpp",
            std::bind(&RetractPolicyServerCpp::handleGoal,     this,
                      std::placeholders::_1, std::placeholders::_2),
            std::bind(&RetractPolicyServerCpp::handleCancel,   this,
                      std::placeholders::_1),
            std::bind(&RetractPolicyServerCpp::handleAccepted, this,
                      std::placeholders::_1)
        );

        // -- Service clients — each on dedicated callback group ------------
        // Two services: /sofa_step (physics) and /ppo_predict (ML)
        // Each needs its own callback group so responses can be processed
        // concurrently while execute() loop is running
        sofa_callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        sofa_client_ = create_client<SofaStep>(
            "/sofa_step",
            rmw_qos_profile_services_default,
            sofa_callback_group_);

        ppo_callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        ppo_client_ = create_client<PPOPredict>(
            "/ppo_predict",
            rmw_qos_profile_services_default,
            ppo_callback_group_);

        // Wait for both services
        RCLCPP_INFO(get_logger(), "Waiting for /sofa_step...");
        while (!sofa_client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) return;
        }
        RCLCPP_INFO(get_logger(), "/sofa_step ready");

        RCLCPP_INFO(get_logger(), "Waiting for /ppo_predict...");
        while (!ppo_client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) return;
        }
        RCLCPP_INFO(get_logger(), "/ppo_predict ready");

        // -- Stop listener on dedicated background thread -----------------
        stop_callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions stop_opts;
        stop_opts.callback_group = stop_callback_group_;

        surgeon_stop_sub_ = create_subscription<std_msgs::msg::Bool>(
            "/surgeon_stop", 10,
            std::bind(&RetractPolicyServerCpp::surgeonStopCb, this,
                      std::placeholders::_1),
            stop_opts);

        emergency_sub_ = create_subscription<std_msgs::msg::Bool>(
            "/emergency_stop", 10,
            std::bind(&RetractPolicyServerCpp::emergencyCb, this,
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
            "RetractPolicyServerCpp ready on /retract_policy_cpp");
    }

    ~RetractPolicyServerCpp()
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
        RCLCPP_INFO(get_logger(), "Retract goal received — accepting");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handleCancel(
        const std::shared_ptr<GoalHandleRetract> /*goal_handle*/)
    {
        RCLCPP_INFO(get_logger(), "Retract cancel — accepting");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handleAccepted(const std::shared_ptr<GoalHandleRetract> goal_handle)
    {
        std::thread([this, goal_handle]() {
            execute(goal_handle);
        }).detach();
    }

    // -----------------------------------------------------------------------
    // Main execution loop
    // -----------------------------------------------------------------------

    void execute(const std::shared_ptr<GoalHandleRetract> goal_handle)
    {
        RCLCPP_INFO(get_logger(), "Executing retract policy (C++)");

        auto goal     = goal_handle->get_goal();
        int max_steps = (goal->max_steps > 0)
                        ? static_cast<int>(goal->max_steps)
                        : DEFAULT_MAX_STEPS;

        // Reset environment and get initial observation
        auto init = sofaReset();
        // obs is the 7D observation: tool_xyz + goal_xyz + phase
        std::array<float, 7> obs;
        for (int i = 0; i < 7; i++) obs[i] = init->observation[i];

        auto feedback        = std::make_shared<Retract::Feedback>();
        int  step            = 0;
        float final_distance = 0.0f;
        std::string termination = "timeout";

        while (step < max_steps) {

            // -- Preemption check FIRST ------------------------------------
            if (goal_handle->is_canceling()) {
                auto result        = std::make_shared<Retract::Result>();
                result->success        = false;
                result->steps_taken    = step;
                result->final_distance = final_distance;
                result->termination    = "preempted";
                goal_handle->canceled(result);
                return;
            }

            // -- Emergency stop -------------------------------------------
            if (emergency_.load()) {
                auto result            = std::make_shared<Retract::Result>();
                result->success        = false;
                result->steps_taken    = step;
                result->final_distance = final_distance;
                result->termination    = "emergency_stop";
                try { goal_handle->abort(result); } catch (...) {}
                RCLCPP_ERROR(get_logger(), "EMERGENCY STOP at step %d", step);
                return;
            }

            // -- Surgeon stop freeze loop ----------------------------------
            while (surgeon_stopped_.load() && !emergency_.load()) {
                if (goal_handle->is_canceling()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            // -- Step 1: Send obs to PPO, get action ----------------------
            // Python PPOPredictService calls policy.predict(obs)
            // PyTorch releases GIL → GPU computes → returns action[3]
            auto ppo_response = ppoPredict(obs);
            if (!ppo_response->success) {
                RCLCPP_ERROR(get_logger(), "PPO predict failed at step %d", step);
                break;
            }
            std::array<float, 3> action = {
                ppo_response->action[0],
                ppo_response->action[1],
                ppo_response->action[2]
            };

            // -- Step 2: Send action to SOFA, get new state ---------------
            auto sofa_response = sofaStep(action);
            step++;

            // Extract new observation for next PPO call
            for (int i = 0; i < 7; i++) obs[i] = sofa_response->observation[i];

            // Use distance from sofa response
            float dist = sofa_response->distance_to_end_position;
            final_distance = dist;

            // -- Post-step surgeon stop freeze loop -----------------------
            while (surgeon_stopped_.load() && !emergency_.load()) {
                if (goal_handle->is_canceling()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            // -- Check goal reached ---------------------------------------
            if (dist < GOAL_THRESHOLD && dist > 0.0f) {
                termination = "goal_reached";
                RCLCPP_INFO(get_logger(),
                    "Goal reached at step %d dist=%.1fmm",
                    step, dist * 1000.0f);
                break;
            }

            // -- Check terminated (collision) -----------------------------
            if (sofa_response->terminated) {
                termination = "collision";
                break;
            }

            // -- Publish feedback -----------------------------------------
            feedback->distance_to_goal = dist;
            feedback->distance_mm      = dist * 1000.0f;
            feedback->step             = step;
            feedback->in_collision     = sofa_response->in_collision;
            feedback->collision_cost   = sofa_response->collision_cost;
            goal_handle->publish_feedback(feedback);

            if (step % 5 == 0) {
                RCLCPP_INFO(get_logger(),
                    "Retract (C++) step %3d | Dist: %.1fmm",
                    step, dist * 1000.0f);
            }
        }

        // -- Return result ------------------------------------------------
        bool success = (termination == "goal_reached");
        auto result  = std::make_shared<Retract::Result>();
        result->success        = success;
        result->steps_taken    = step;
        result->final_distance = final_distance;
        result->termination    = termination;
        goal_handle->succeed(result);

        RCLCPP_INFO(get_logger(),
            "Retract complete: %s steps=%d dist=%.1fmm",
            termination.c_str(), step, final_distance * 1000.0f);
    }

    // -----------------------------------------------------------------------
    // Service helpers
    // -----------------------------------------------------------------------

    std::shared_ptr<SofaStep::Response> sofaReset()
    {
        auto request    = std::make_shared<SofaStep::Request>();
        request->action = {0.0f, 0.0f, 0.0f};
        request->reset  = true;
        auto future = sofa_client_->async_send_request(request);
        auto status = future.wait_for(std::chrono::seconds(10));
        if (status != std::future_status::ready) {
            RCLCPP_ERROR(get_logger(), "SOFA reset timed out");
            return std::make_shared<SofaStep::Response>();
        }
        RCLCPP_INFO(get_logger(), "SOFA reset complete");
        return future.get();
    }

    std::shared_ptr<SofaStep::Response> sofaStep(
        std::array<float, 3> action)
    {
        auto request    = std::make_shared<SofaStep::Request>();
        request->action = action;
        request->reset  = false;
        auto future = sofa_client_->async_send_request(request);
        auto status = future.wait_for(std::chrono::seconds(5));
        if (status != std::future_status::ready) {
            RCLCPP_ERROR(get_logger(), "SOFA step timed out");
            return std::make_shared<SofaStep::Response>();
        }
        return future.get();
    }

    std::shared_ptr<PPOPredict::Response> ppoPredict(
        std::array<float, 7> obs)
    {
        auto request = std::make_shared<PPOPredict::Request>();
        for (int i = 0; i < 7; i++) request->observation[i] = obs[i];
        auto future = ppo_client_->async_send_request(request);
        auto status = future.wait_for(std::chrono::seconds(5));
        if (status != std::future_status::ready) {
            RCLCPP_ERROR(get_logger(), "PPO predict timed out");
            return std::make_shared<PPOPredict::Response>();
        }
        return future.get();
    }

    // -----------------------------------------------------------------------
    // Stop callbacks
    // -----------------------------------------------------------------------

    void surgeonStopCb(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data) {
            surgeon_stopped_.store(true);
            RCLCPP_WARN(get_logger(), "Retract (C++): SURGEON STOP");
        } else {
            surgeon_stopped_.store(false);
            RCLCPP_INFO(get_logger(), "Retract (C++): SURGEON RESUME");
        }
    }

    void emergencyCb(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data && !emergency_.load()) {
            emergency_.store(true);
            RCLCPP_ERROR(get_logger(), "Retract (C++): EMERGENCY STOP");
        }
    }

    // -----------------------------------------------------------------------
    // Member variables
    // -----------------------------------------------------------------------

    rclcpp_action::Server<Retract>::SharedPtr                    action_server_;

    // Two service clients — each with dedicated callback group
    rclcpp::Client<SofaStep>::SharedPtr                          sofa_client_;
    rclcpp::CallbackGroup::SharedPtr                             sofa_callback_group_;
    rclcpp::Client<PPOPredict>::SharedPtr                        ppo_client_;
    rclcpp::CallbackGroup::SharedPtr                             ppo_callback_group_;

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
    auto node = std::make_shared<RetractPolicyServerCpp>();

    // Four callback groups need concurrent execution:
    // 1. Action server (default)
    // 2. sofa_callback_group_ — /sofa_step responses
    // 3. ppo_callback_group_  — /ppo_predict responses
    // 4. stop_callback_group_ — surgeon stop (on own thread)
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}