#pragma once

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

#include <ros/ros.h>
#include <std_msgs/Float32.h>

struct UR5ModelPlugin : public gazebo::ModelPlugin {
    UR5ModelPlugin()
        : gazebo::ModelPlugin::ModelPlugin(),
          nh()
    {
    }

    void Load(gazebo::physics::ModelPtr _model, sdf::ElementPtr sdf) override
    {
        if (sdf->HasElement("topic_name")) {
            topic_name = sdf->GetElement("topic_name")->Get<std::string>();
        } else {
            topic_name = "cmd_action";
        }

        model = _model;

        hinge1 = model->GetJoint("hinge1");

        update_conn = gazebo::event::Events::ConnectWorldUpdateBegin(
            std::bind(&UR5ModelPlugin::onUpdate, this));

        state_pub = nh.advertise<std_msgs::Float32>("pendulum_state", 1000);
        action_sub = nh.subscribe(topic_name, 10, &UR5ModelPlugin::actionCallback, this);
    }

    void onUpdate()
    {
        state.data = hinge1->Position(0);

        hinge1->SetForce(0, action.data);

        state_pub.publish(state);
        ros::spinOnce();
    }

    void actionCallback(const std_msgs::Float32& _action)
    {
        action = _action;
    }


    std_msgs::Float32 state;
    std_msgs::Float32 action;

private:
    gazebo::physics::ModelPtr model;

    gazebo::physics::JointPtr hinge1;

    gazebo::event::ConnectionPtr update_conn;

    ros::NodeHandle nh;
    ros::Publisher state_pub;
    ros::Subscriber action_sub;

    std::string topic_name;
};
