#pragma once

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

#include <ros/ros.h>
#include <ur5_msgs/Action.h>
#include <ur5_msgs/State.h>

struct UR5ModelPlugin : public gazebo::ModelPlugin {
    UR5ModelPlugin()
        : gazebo::ModelPlugin::ModelPlugin()
    {
    }

    void Load(gazebo::physics::ModelPtr _model, sdf::ElementPtr sdf) override
    {
        if (sdf->HasElement("topic_name")) {
            topic_name = sdf->GetElement("topic_name")->Get<std::string>();
        } else {
            topic_name = "ur5_action";
        }

        std::cout << "Loading Model..." << std::endl;
        std::cout << "Model Name: " << _model->GetName() << std::endl;

        model = _model;

        for (const auto& j : model->GetJoints()) {
            std::cout << "Joint Name: " << j->GetName() << std::endl;
        }
        hinge1 = model->GetJoint("hinge1");

        update_conn = gazebo::event::Events::ConnectWorldUpdateBegin(
            std::bind(&UR5ModelPlugin::onUpdate, this));

        state_pub = nh.advertise<ur5_msgs::State>("ur5_state", 1000);
        action_sub = nh.subscribe(topic_name, 10, &UR5ModelPlugin::actionCallback, this);
    }

    void onUpdate()
    {
        state.angle1 = hinge1->Position(0);
        state.angularvel1 = hinge1->GetVelocity(0);

        hinge1->SetForce(0, action.torque1);

        state_pub.publish(state);

        ros::spinOnce();
    }

    void actionCallback(const ur5_msgs::Action& _action)
    {
        action = _action;
    }


    ur5_msgs::State state;
    ur5_msgs::Action action;

private:
    gazebo::physics::ModelPtr model;

    gazebo::physics::JointPtr hinge1;

    gazebo::event::ConnectionPtr update_conn;

    ros::NodeHandle nh;
    ros::Publisher state_pub;
    ros::Subscriber action_sub;

    std::string topic_name;
};
