cmake_minimum_required(VERSION 3.02)
project(ur5_mujoco)

find_package(catkin REQUIRED COMPONENTS
  roslib
)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Loading MuJoCo library
set(MUJOCO_DIR ${CMAKE_CURRENT_LIST_DIR}/mujoco)
set(MUJOCO_BUILD_TESTS OFF CACHE INTERNAL "Build tests for MuJoCo" FORCE)
add_subdirectory(${MUJOCO_DIR})

add_executable(${PROJECT_NAME}_node src/ur5_mujoco_node.cpp src/glfw_func.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  mujoco
  glfw
)
target_include_directories(${PROJECT_NAME}_node PRIVATE 
  ${MUJOCO_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/include/ur5_mujoco
)