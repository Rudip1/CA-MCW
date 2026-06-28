#include "vf_robot_gazebo/ultrasound.h"
#include <vf_robot_messages/msg/ultra_sound.hpp>  // ROS2 style include
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/range.hpp>

Ultrasound::Ultrasound()
: Node("ultrasound")  // ROS2 Node
{
    RCLCPP_INFO(this->get_logger(), "Ultrasound: initializing...");

    // Publisher
    ultrasound_publisher = this->create_publisher<vf_robot_messages::msg::UltraSound>("/esp/range", 1);

    // Subscribers
    _ultrasound_subscriber_0 = this->create_subscription<sensor_msgs::msg::Range>(
        "/ultrasound/front_left", 1,
        std::bind(&Ultrasound::ultrasoundCallback, this, std::placeholders::_1)
    );
    _ultrasound_subscriber_1 = this->create_subscription<sensor_msgs::msg::Range>(
        "/ultrasound/front_right", 1,
        std::bind(&Ultrasound::ultrasoundCallback, this, std::placeholders::_1)
    );
    _ultrasound_subscriber_2 = this->create_subscription<sensor_msgs::msg::Range>(
        "/ultrasound/right", 1,
        std::bind(&Ultrasound::ultrasoundCallback, this, std::placeholders::_1)
    );
    _ultrasound_subscriber_3 = this->create_subscription<sensor_msgs::msg::Range>(
        "/ultrasound/rear", 1,
        std::bind(&Ultrasound::ultrasoundCallback, this, std::placeholders::_1)
    );
    _ultrasound_subscriber_4 = this->create_subscription<sensor_msgs::msg::Range>(
        "/ultrasound/left", 1,
        std::bind(&Ultrasound::ultrasoundCallback, this, std::placeholders::_1)
    );
}

void Ultrasound::ultrasoundCallback(const sensor_msgs::msg::Range::SharedPtr msg) {
    vf_robot_messages::msg::UltraSound vfm;
    vfm.code = sensor_ids[msg->header.frame_id];
    vfm.range = msg->range;

    ultrasound_publisher->publish(vfm);
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Ultrasound>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
