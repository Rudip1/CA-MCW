[![Build Status](https://drone.euroknows.com/api/badges/euroknows/UVCRobotMessages/status.svg)](https://drone.euroknows.com/euroknows/UVCRobotMessages)

# ROS messages and services package

## Add as a git submodule into the src folder

git submodule add ssh://git@git.euroknows.com:2222/euroknows/UVCRobotMessages.git src/vfmessages

## Build messages and services in Dockerfile

RUN . /opt/ros/melodic/setup.sh \
 && catkin build vfmessages \
 && catkin build packagename


## Check client manager project for example