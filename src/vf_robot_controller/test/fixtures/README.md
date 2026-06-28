# Test fixtures

Recorded artifacts used by unit and integration tests. Committed to the repo
so tests are reproducible without a live robot.

Files needed (fill in as phases progress):

- `sample_costmap.bin` — serialized nav2_costmap_2d for GCF tests (Phase 4)
- `sample_pointcloud.pcd` — small RealSense sample for voxel filter / GCF 3D (Phase 4)
- `sample_rtabmap.db` — short recorded SLAM session for RtabmapBackend tests (Phase 6)
- `sample_meta_critic.onnx` — tiny exported model for OnnxWeightProvider tests (Phase 8)
- `tiny_episode.h5` — 100-sample HDF5 fixture for dataset/training tests (Phase 8)
- `sample_static_map/map.pgm`, `sample_static_map/map.yaml` — pgm fixture (Phase 6)

Record fixtures once. Commit. Use forever.
