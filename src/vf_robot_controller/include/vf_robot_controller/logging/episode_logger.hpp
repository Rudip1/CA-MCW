// EpisodeLogger — writes one HDF5 file per navigation episode.
// Schema documented in docs/data_format.md.
// Phase 7 implementation.

#ifndef VF_ROBOT_CONTROLLER__LOGGING__EPISODE_LOGGER_HPP_
#define VF_ROBOT_CONTROLLER__LOGGING__EPISODE_LOGGER_HPP_

#include <string>

namespace vf_robot_controller::logging {

class EpisodeLogger {
public:
  EpisodeLogger() = default;
  // TODO Phase 7: open(filename), append(...), close()
};

}  // namespace vf_robot_controller::logging

#endif
