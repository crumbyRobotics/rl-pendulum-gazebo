cmake_minimum_required(VERSION 3.16)

project(test_mujoco CXX)

set(MUJOCO_DIR ${CMAKE_CURRENT_LIST_DIR}/mujoco)
add_subdirectory(${MUJOCO_DIR})

add_executable(test_mujoco test_mujoco.cpp)
target_link_libraries(test_mujoco 
    mujoco
    glfw
)
target_include_directories(test_mujoco PRIVATE ${MUJOCO_DIR})