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

# -- Executable -------------------------------------------------------------
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

# -- Install ----------------------------------------------------------------
# Copies executable to install/lapgym_ros2_bridge_cpp/lib/
# This is where ros2 run looks for executables
install(TARGETS approach_policy_server_cpp
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()