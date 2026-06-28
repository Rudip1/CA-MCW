// StaticMapBackend — Phase 6.
//
// Reads map_server-style yaml + pgm, builds an in-memory occupancy grid,
// and answers `queryPersistentObstacles` by sampling the grid along
// (angle, radius) directions from the robot pose.
//
// Capabilities: { persistent_2d: true, topology: false, structure_3d: false }
//
// We hand-parse the yaml (only six keys, fixed schema produced by
// map_server / SLAM toolbox / RTAB-Map) and the PGM (P5 binary) to keep
// the dependency surface minimal — same approach Phase 4 took for the
// voxel filter rather than pulling PCL.

#include "vf_robot_controller/perception/map_backend/static_map_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace vf_robot_controller::perception {

namespace {

struct MapYaml {
  std::string image;
  double resolution{0.05};
  double origin_x{0.0};
  double origin_y{0.0};
  int    negate{0};
  double occupied_thresh{0.65};
  double free_thresh{0.25};
  bool   ok{false};
};

// Strip leading / trailing whitespace and one optional trailing comment.
std::string trim(const std::string & in)
{
  std::string s = in;
  const auto h = s.find('#');
  if (h != std::string::npos) s.erase(h);
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// Hand-rolled flat-yaml reader. Handles `key: value` and
// `origin: [x, y, z]`. Anything else is ignored — sufficient for the
// fixed schema used by ROS map_server.
MapYaml parseMapYaml(const std::string & path)
{
  MapYaml out;
  std::ifstream f(path);
  if (!f.is_open()) return out;
  std::string line;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty()) continue;
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = trim(line.substr(0, colon));
    std::string val = trim(line.substr(colon + 1));
    if (val.empty()) continue;
    if (key == "image") {
      out.image = val;
    } else if (key == "resolution") {
      out.resolution = std::stod(val);
    } else if (key == "origin") {
      // expected [x, y, z]
      auto lb = val.find('['); auto rb = val.find(']');
      if (lb != std::string::npos && rb != std::string::npos) {
        std::string body = val.substr(lb + 1, rb - lb - 1);
        std::replace(body.begin(), body.end(), ',', ' ');
        std::istringstream is(body);
        double x = 0, y = 0, z = 0; is >> x >> y >> z;
        out.origin_x = x; out.origin_y = y;
      }
    } else if (key == "negate") {
      out.negate = std::stoi(val);
    } else if (key == "occupied_thresh") {
      out.occupied_thresh = std::stod(val);
    } else if (key == "free_thresh") {
      out.free_thresh = std::stod(val);
    }
  }
  out.ok = !out.image.empty() && out.resolution > 0.0;
  return out;
}

// Minimal P5 (binary 8-bit) PGM reader. Returns row-major bytes top-to-bottom
// (PGM convention: first byte is upper-left). Width / height filled.
bool readPgmP5(const std::string & path,
               int & width, int & height, std::vector<uint8_t> & data)
{
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) return false;
  auto read_token = [&](std::string & tok) -> bool {
    tok.clear();
    char c;
    while (f.get(c)) {
      if (c == '#') {
        // comment to EOL
        while (f.get(c) && c != '\n') {}
        continue;
      }
      if (std::isspace(static_cast<unsigned char>(c))) {
        if (!tok.empty()) return true;
        continue;
      }
      tok.push_back(c);
    }
    return !tok.empty();
  };
  std::string magic;
  if (!read_token(magic) || magic != "P5") return false;
  std::string w, h, m;
  if (!read_token(w) || !read_token(h) || !read_token(m)) return false;
  width = std::stoi(w); height = std::stoi(h);
  const int maxval = std::stoi(m);
  if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) return false;
  // read_token already consumed the single trailing whitespace separator
  // after `maxval` (the first whitespace it sees after a non-empty token
  // makes it return), so the file cursor is now on the first pixel byte.
  data.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  f.read(reinterpret_cast<char *>(data.data()),
         static_cast<std::streamsize>(data.size()));
  return f.good() || (f.eof() && f.gcount() == static_cast<std::streamsize>(data.size()));
}

}  // namespace

// Backend impl. We keep the small grid + index data in private members
// declared inline here via an opaque pImpl-ish struct stashed on the heap.
struct StaticMapBackend::Impl {
  bool   available{false};
  int    width{0};
  int    height{0};
  double resolution{0.05};
  double origin_x{0.0};
  double origin_y{0.0};
  // True when cell is "occupied" by map_server's threshold rules. Indexed
  // [j * width + i] with j growing in +y world direction (we flip PGM rows
  // so origin is bottom-left, matching nav2 / map_server convention).
  std::vector<uint8_t> occupied;
};

StaticMapBackend::StaticMapBackend(const std::string & yaml_path)
: impl_(new Impl())
{
  const MapYaml y = parseMapYaml(yaml_path);
  if (!y.ok) return;

  // image path may be relative to the yaml directory.
  std::filesystem::path image_path = y.image;
  if (image_path.is_relative()) {
    std::filesystem::path base(yaml_path);
    image_path = base.parent_path() / image_path;
  }

  int w = 0, h = 0;
  std::vector<uint8_t> raw;
  if (!readPgmP5(image_path.string(), w, h, raw)) return;

  impl_->width = w;
  impl_->height = h;
  impl_->resolution = y.resolution;
  impl_->origin_x = y.origin_x;
  impl_->origin_y = y.origin_y;
  impl_->occupied.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0);

  // PGM convention: white = free (255), black = occupied (0). With
  // `negate: 0` (the typical map_server output) we map p = (255 - raw)/255
  // and compare to occupied_thresh / free_thresh. Flip the row order so
  // (0,0) of the index is the bottom-left world-aligned cell.
  for (int j = 0; j < h; ++j) {
    const int src_row = h - 1 - j;
    for (int i = 0; i < w; ++i) {
      const uint8_t pix = raw[static_cast<size_t>(src_row) * w + i];
      double p = (y.negate != 0) ? (pix / 255.0) : ((255 - pix) / 255.0);
      uint8_t occ = 0;
      if (p >= y.occupied_thresh) occ = 1;
      // free_thresh / unknown otherwise: we keep 0 (treat as free).
      impl_->occupied[static_cast<size_t>(j) * w + i] = occ;
    }
  }

  available_ = true;
  impl_->available = true;
}

StaticMapBackend::~StaticMapBackend() = default;

bool StaticMapBackend::isAvailable() const { return available_; }

BackendCapabilities StaticMapBackend::capabilities() const
{
  return {true, false, false};
}

std::vector<float> StaticMapBackend::queryPersistentObstacles(
  const Pose2D & robot_pose,
  const std::vector<float> & angles,
  const std::vector<float> & radii) const
{
  std::vector<float> out(angles.size() * radii.size(), 0.0f);
  if (!available_ || !impl_ || impl_->resolution <= 0.0) return out;

  const auto & I = *impl_;

  auto worldToCell = [&](double wx, double wy, int & ix, int & iy) {
    ix = static_cast<int>(std::floor((wx - I.origin_x) / I.resolution));
    iy = static_cast<int>(std::floor((wy - I.origin_y) / I.resolution));
    return ix >= 0 && iy >= 0 && ix < I.width && iy < I.height;
  };
  auto cellOccupied = [&](int ix, int iy) {
    if (ix < 0 || iy < 0 || ix >= I.width || iy >= I.height) return uint8_t{1};
    return I.occupied[static_cast<size_t>(iy) * I.width + ix];
  };

  // Step along ray at half a cell to keep DDA cheap and accurate.
  const double step = I.resolution * 0.5;

  for (size_t a = 0; a < angles.size(); ++a) {
    const double ca = std::cos(static_cast<double>(angles[a]));
    const double sa = std::sin(static_cast<double>(angles[a]));
    for (size_t r = 0; r < radii.size(); ++r) {
      const double rad = static_cast<double>(radii[r]);
      // March the ray; accumulate the *closest* hit fraction. If a hit is
      // found within the radius, output is 1 - (hit_dist / rad) so the
      // value is in [0,1] and grows when obstacles are nearer.
      bool hit = false;
      double hit_dist = rad;
      const int n = std::max(2, static_cast<int>(std::ceil(rad / step)));
      for (int k = 1; k <= n; ++k) {
        const double d = (rad * k) / n;
        const double wx = robot_pose.x + d * ca;
        const double wy = robot_pose.y + d * sa;
        int ix = 0, iy = 0;
        if (!worldToCell(wx, wy, ix, iy)) {
          // Off-map = treat as walled (matches "we don't know, assume blocked")
          hit = true; hit_dist = d; break;
        }
        if (cellOccupied(ix, iy)) {
          hit = true; hit_dist = d; break;
        }
      }
      if (hit) {
        const float frac = static_cast<float>(1.0 - hit_dist / rad);
        out[a * radii.size() + r] = std::clamp(frac, 0.0f, 1.0f);
      }
    }
  }
  return out;
}

std::optional<TopologicalFeatures>
StaticMapBackend::queryTopology(const Pose2D &) const
{
  return std::nullopt;
}

std::optional<StructuralFeatures3D>
StaticMapBackend::query3DStructure(const Pose2D &) const
{
  return std::nullopt;
}

}  // namespace vf_robot_controller::perception
