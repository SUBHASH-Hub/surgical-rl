cmake_minimum_required(VERSION 3.8)
project(lapgym_ros2_bridge_cpp)

# C++17 standard — required for std::array initializer lists
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Enable warnings — good engineering practice
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -- Find packages ----------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(lapgym_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# -- Executable -------------------------------------------------------------
# -- Approach server -------------------------------------------------------
add_executable(approach_policy_server_cpp
  src/approach_policy_server.cpp
)

# -- Link dependencies ------------------------------------------------------
ament_target_dependencies(approach_policy_server_cpp
  rclcpp
  rclcpp_action
  lapgym_interfaces
  std_msgs
)
#  For each executable to use Eigen
target_link_libraries(approach_policy_server_cpp
  Eigen3::Eigen)

# -- Install ----------------------------------------------------------------
# Copies executable to install/lapgym_ros2_bridge_cpp/lib/
# This is where ros2 run looks for executables
install(TARGETS approach_policy_server_cpp
  DESTINATION lib/${PROJECT_NAME}
)


# -- Hold server -----------------------------------------------------------
add_executable(hold_policy_server_cpp
  src/hold_policy_server.cpp
)
ament_target_dependencies(hold_policy_server_cpp
  rclcpp rclcpp_action lapgym_interfaces std_msgs)
install(TARGETS hold_policy_server_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# -- Retract server --------------------------------------------------------
add_executable(retract_policy_server_cpp
  src/retract_policy_server.cpp
)
ament_target_dependencies(retract_policy_server_cpp
  rclcpp rclcpp_action lapgym_interfaces std_msgs)
install(TARGETS retract_policy_server_cpp
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
# -- Google Tests -----------------------------------------------------------
# Unit tests for ProportionalController — no ROS2 dependency required.
# IEC 62304 Class C: unit verification of safety-relevant control logic.
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_proportional_controller
    test/test_proportional_controller.cpp
  )
  target_link_libraries(test_proportional_controller
    Eigen3::Eigen
  )
endif()
