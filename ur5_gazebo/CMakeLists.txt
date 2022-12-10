cmake_minimum_required(VERSION 3.0.2)
project(ur5_gazebo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_plugins
  ur5_msgs
)
find_package(gazebo REQUIRED)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

add_library(ur5_model_plugin SHARED src/ur5_model_plugin.cpp)

add_dependencies(ur5_model_plugin ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(ur5_model_plugin
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)