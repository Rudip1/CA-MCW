// nodes/map_backend_node.cpp — Phase 6.
//
// Production path: feature_extractor_node holds the IMapBackend in-process
// (no service round trip on the 20 Hz loop). This node exists as a
// diagnostic / introspection surface — it instantiates the same backend
// based on `backend_selection.yaml` and publishes a small status string
// + capability flags on `/vf/map_backend_status` so an operator can see
// at a glance which backend is live.
//
// Re-instantiating the backend here is cheap (RTAB sqlite open is ~ms);
// it does NOT share state with feature_extractor_node's instance.

#include <chrono>
#include <memory>
#include <sstream>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "vf_robot_controller/perception/map_backend/i_map_backend.hpp"
#include "vf_robot_controller/perception/map_backend/cuvslam_backend.hpp"
#include "vf_robot_controller/perception/map_backend/rtabmap_backend.hpp"
#include "vf_robot_controller/perception/map_backend/static_map_backend.hpp"

namespace vfp = vf_robot_controller::perception;

class MapBackendNode : public rclcpp::Node {
public:
  MapBackendNode()
  : Node("map_backend_node")
  {
    declare_parameter<std::string>("backend", "none");
    declare_parameter<std::string>("rtabmap_db_path", "");
    declare_parameter<std::string>("static_map_yaml", "");
    declare_parameter<double>("status_rate_hz", 1.0);

    backend_ = makeBackend();

    pub_ = create_publisher<std_msgs::msg::String>(
      "/vf/map_backend_status", rclcpp::QoS(1).transient_local());

    const double rate = std::max(0.1, get_parameter("status_rate_hz").as_double());
    const auto period = std::chrono::duration<double>(1.0 / rate);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { publishStatus(); });

    RCLCPP_INFO(get_logger(), "map_backend_node ready (backend='%s', available=%s)",
                get_parameter("backend").as_string().c_str(),
                (backend_ && backend_->isAvailable()) ? "true" : "false");
  }

private:
  std::shared_ptr<vfp::IMapBackend> makeBackend()
  {
    const std::string sel = get_parameter("backend").as_string();
    if (sel == "rtabmap") {
      const auto p = get_parameter("rtabmap_db_path").as_string();
      auto b = std::make_shared<vfp::RtabmapBackend>(p);
      if (!b->isAvailable()) {
        RCLCPP_WARN(get_logger(),
          "RtabmapBackend unavailable (db='%s'); trying static fallback", p.c_str());
        const auto sp = get_parameter("static_map_yaml").as_string();
        if (!sp.empty()) {
          auto s = std::make_shared<vfp::StaticMapBackend>(sp);
          if (s->isAvailable()) return s;
        }
        return b;  // not available, but kept for diagnostics
      }
      return b;
    }
    if (sel == "static") {
      const auto sp = get_parameter("static_map_yaml").as_string();
      return std::make_shared<vfp::StaticMapBackend>(sp);
    }
    if (sel == "cuvslam") {
      return std::make_shared<vfp::CuvslamBackend>();
    }
    return nullptr;
  }

  void publishStatus()
  {
    std_msgs::msg::String msg;
    std::ostringstream os;
    os << "backend=" << get_parameter("backend").as_string()
       << " available=" << ((backend_ && backend_->isAvailable()) ? "true" : "false");
    if (backend_) {
      const auto c = backend_->capabilities();
      os << " persistent_2d=" << (c.persistent_2d ? "1" : "0")
         << " topology=" << (c.topology ? "1" : "0")
         << " structure_3d=" << (c.structure_3d ? "1" : "0");
    }
    msg.data = os.str();
    pub_->publish(msg);
  }

  std::shared_ptr<vfp::IMapBackend> backend_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MapBackendNode>());
  rclcpp::shutdown();
  return 0;
}
