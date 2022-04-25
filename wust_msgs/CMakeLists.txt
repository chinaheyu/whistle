cmake_minimum_required(VERSION 3.0.2)
project(wust_msgs)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  geometry_msgs
  message_generation
  nav_msgs
  std_msgs
)

add_message_files(
  FILES
  EnemyInfo.msg
  WhistleResult.msg
  CarPosition.msg
  RemainBullet.msg
)

add_service_files(
  FILES
  SetTopAlgorithm.srv
)

add_action_files(
  FILES
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(CATKIN_DEPENDS std_msgs geometry_msgs nav_msgs message_runtime)

