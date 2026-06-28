// nodes/data_collector_node.cpp — HDF5 logger for COLLECT mode
// Phase 7 implementation. Phase 0 stub: spins, does nothing.

#include <rclcpp/rclcpp.hpp>

class DataCollectorNode : public rclcpp::Node {
public:
  DataCollectorNode() : Node("data_collector_node") {
    RCLCPP_INFO(get_logger(), "%s started (Phase 0 stub)", get_name());
    // TODO Phase 7: implementation
  }
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DataCollectorNode>());
  rclcpp::shutdown();
  return 0;
}
