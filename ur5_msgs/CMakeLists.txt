cmake_minimum_required(VERSION 3.0.2)
project(ur5_msgs)

find_package(catkin REQUIRED COMPONENTS
    message_generation
)

add_message_files(
    FILES
    State.msg
    Action.msg
)
generate_messages()

catkin_package()