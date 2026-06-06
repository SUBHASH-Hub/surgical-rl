/**
 * Phase 4H: Google Test unit tests for ProportionalController
 *
 * Tests the pure C++ logic of ProportionalController independently
 * of ROS2 — no nodes, no services, no SOFA required.
 *
 * Why unit tests matter for medical devices:
 *   IEC 62304 Class C requires software unit verification.
 *   These tests verify the controller clips correctly, computes
 *   the correct direction, and handles edge cases — independently
 *   of the full system integration test.
 *
 * Run with: colcon test --packages-select lapgym_ros2_bridge_cpp
 *           colcon test-result --verbose
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>

// -- Copy ProportionalController here for standalone testing ----------------
// In production this would be a shared header. For now we duplicate it
// so tests have no ROS2 dependency — pure C++ logic only.
class ProportionalController
{
public:
    ProportionalController(float gain, float max_action)
    : gain_(gain), max_action_(max_action)
    {}

    Eigen::Vector3f compute(const Eigen::Vector3f& error)
    {
        return (error.normalized() * gain_)
                .cwiseMax(-max_action_)
                .cwiseMin( max_action_);
    }

    void reset() {}

private:
    float gain_;
    float max_action_;
};

// ---------------------------------------------------------------------------
// Test 1 — Action points in correct direction
// If target is directly to the right (+X), action should be +X
// ---------------------------------------------------------------------------
TEST(ProportionalController, DirectionIsCorrect)
{
    ProportionalController ctrl(2.0f, 3.0f);
    Eigen::Vector3f error(1.0f, 0.0f, 0.0f);  // error pointing in +X
    Eigen::Vector3f action = ctrl.compute(error);

    EXPECT_GT(action.x(), 0.0f);   // action should be positive X
    EXPECT_NEAR(action.y(), 0.0f, 1e-5f);
    EXPECT_NEAR(action.z(), 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Test 2 — Action is clipped to max bound
// Large error should produce action clipped to max_action
// ---------------------------------------------------------------------------
TEST(ProportionalController, ClipsToMaxBound)
{
    ProportionalController ctrl(5.0f, 3.0f); // gain=5 ensures clipping fires
    Eigen::Vector3f big_error(100.0f, 0.0f, 0.0f);
    Eigen::Vector3f action = ctrl.compute(big_error);

    EXPECT_NEAR(action.x(), 3.0f, 1e-4f);   // clipped to max_action=3.0
    EXPECT_LE(action.x(), 3.0f);             // never exceeds bound
}

// ---------------------------------------------------------------------------
// Test 3 — Action is clipped to negative bound
// Error pointing in -X should produce action clipped to -max_action
// ---------------------------------------------------------------------------
TEST(ProportionalController, ClipsToNegativeBound)
{
    ProportionalController ctrl(5.0f, 3.0f);  // gain=5 ensures clipping fires
    Eigen::Vector3f neg_error(-100.0f, 0.0f, 0.0f);
    Eigen::Vector3f action = ctrl.compute(neg_error);

    EXPECT_NEAR(action.x(), -3.0f, 1e-4f);  // clipped to -max_action
    EXPECT_GE(action.x(), -3.0f);           // never below negative bound
}

// ---------------------------------------------------------------------------
// Test 4 — Unit error gives action equal to gain
// error.norm()=1 → normalised error = error → action = error * gain
// ---------------------------------------------------------------------------
TEST(ProportionalController, UnitErrorGivesGainMagnitude)
{
    ProportionalController ctrl(2.0f, 3.0f);
    Eigen::Vector3f unit_error(1.0f, 0.0f, 0.0f);  // already unit length
    Eigen::Vector3f action = ctrl.compute(unit_error);

    // gain=2.0, max_action=3.0 → action.x should be 2.0 (not clipped)
    EXPECT_NEAR(action.x(), 2.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// Test 5 — 3D diagonal error produces correct direction
// Error at 45 degrees in XY plane — action should point same direction
// ---------------------------------------------------------------------------
TEST(ProportionalController, DiagonalErrorCorrectDirection)
{
    ProportionalController ctrl(2.0f, 3.0f);
    Eigen::Vector3f diagonal(1.0f, 1.0f, 0.0f);
    Eigen::Vector3f action = ctrl.compute(diagonal);

    // Both X and Y should be equal and positive
    EXPECT_GT(action.x(), 0.0f);
    EXPECT_GT(action.y(), 0.0f);
    EXPECT_NEAR(action.x(), action.y(), 1e-4f);  // equal magnitude
    EXPECT_NEAR(action.z(), 0.0f, 1e-5f);
}

int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
