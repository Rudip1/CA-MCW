#ifndef ULTRASOUND_H_
#define ULTRASOUND_H_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <vf_robot_messages/msg/ultra_sound.hpp>
#include <map>
#include <string>
#include <memory>

class Ultrasound : public rclcpp::Node {
public:
    Ultrasound();  // No NodeHandle in ROS2

private:
    void ultrasoundCallback(const sensor_msgs::msg::Range::SharedPtr msg);
    int getSensorId(const std::string &input);

    // Publisher
    rclcpp::Publisher<vf_robot_messages::msg::UltraSound>::SharedPtr ultrasound_publisher;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr _ultrasound_subscriber_0;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr _ultrasound_subscriber_1;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr _ultrasound_subscriber_2;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr _ultrasound_subscriber_3;
    rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr _ultrasound_subscriber_4;

    // Sensor IDs
    std::map<std::string, int> sensor_ids{
        {"front_left", 0},
        {"front_right", 1},
        {"right", 2},
        {"rear", 3},
        {"left", 4}
    };
};

#endif // ULTRASOUND_H_
