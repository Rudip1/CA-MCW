# Hospital Tommaso Models — Attribution

These Gazebo models are bundled together by **Tommaso Vandermeer** in his
[Hospitalbot-Path-Planning](https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning)
project, which uses ROS 2 + Gazebo + OpenAI Gym + Stable-Baselines3 to train
reinforcement-learning motion-planning agents in an indoor hospital
environment. The catalogue itself is a re-distribution of the open-source
**AWS RoboMaker hospital world** asset pack, plus a Pioneer 3AT robot model.

> Upstream project: <https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning>
> Author: Tommaso Vandermeer
> Original AWS asset pack: <https://github.com/aws-robotics/aws-robomaker-hospital-world>
> Pioneer 3AT model: <https://github.com/MOGI-ROS/Week-9-10-Gazebo-sensors>

All credit for the model assets goes to AWS RoboMaker and the upstream
contributors. This folder is included only so that
`worlds/hospital_Tommaso_world/*.world` resolves its `model://…` URIs locally.

## Local modifications

Only one model has been touched, for stability reasons:

- `aws_robomaker_hospital_floor_01_floor/model.sdf` — the upstream SDF set
  `<mu>100</mu> <mu2>50</mu2>` which combined with our wheel collisions
  produced stick-slip jitter at rest. Normalised to `mu=mu2=1.0` (Gazebo
  default). See the inline comment in the file for context.

No mesh/visual asset has been modified.
