# Hospital Tommaso World — Attribution

This Gazebo world is adapted from **Tommaso Vandermeer**'s
[Hospitalbot-Path-Planning](https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning)
project, which provides a ROS 2 + Gazebo + OpenAI Gym + Stable-Baselines3
infrastructure for training reinforcement-learning agents on a motion-planning
problem inside an indoor hospital scene built around the AWS RoboMaker
hospital model catalogue and a Pioneer 3AT differential-drive robot with a
180° LIDAR.

> Repository: <https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning>
> Author: Tommaso Vandermeer
> Used here as: the Gazebo simulation scene only (we replace the Pioneer 3AT
> with the UVC1 ViroFighter robot defined in `vf_robot_description`).

All credit for the world layout, the AWS RoboMaker hospital asset bundle, and
the underlying training infrastructure design goes to the upstream project.

## Files in this folder

| File | Source |
|---|---|
| `hospital_Tommaso_hospital.world`   | upstream world, physics block locally tuned for contact stability |
| `hospital_Tommaso_training.world`   | upstream training world |
| `hospital_Tommaso_test.world`       | upstream test world |
| `hospital_Tommaso_discretized.world`| upstream discretized variant |

Local modifications are limited to physics-solver settings (no asset
modification). See top-of-file comments inside each `.world`.

## References

- Hospital world models: <https://github.com/aws-robotics/aws-robomaker-hospital-world>
- Pioneer 3AT model: <https://github.com/MOGI-ROS/Week-9-10-Gazebo-sensors>
- Master's thesis (Tommaso Vandermeer):
  <https://www.tesi.unipd.it/handle/20.500.12608/49697>
