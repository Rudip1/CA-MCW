# Robot model

This package contains all the stuff that is needed for gazebo simulation.
This includes the model of the UVC robot, and world files that describe simulation environments
where the robot can move around.

### How is the robot model built up
The model consists of links, joints. Links are parts of the robot that are assumed to be rigid bodies, while joints are to connect the links
by a constrained motion.
All links have two parts: collisions and visuals. Collisions define the physical borders of the link, while visuals are just for visualization.
The idea behind this is that in case of complicated geometry it is computationally cheaper to visualize the geometry in a simplified way.
Both collisions and visuals are built up from basic geometric shapes: spheres, cubes and cylinders.
In case of UVC_robot model the collision and visual parts are the same.

Additionally there are plugins attached to the model. Currently the UVC_robot model has two plugins, one for the differential drive 
and one for the depth camera. These plugins are ROS nodes that simulate the behaviour of components.



### How to edit the robot model
The robot model is defined in the model.sdf file. This is practically an XML. One way to modify the model is editing this file, 
however it is not a user friendly way.
Instead, open up gazebo (from the applications menu not from terminal. 
Don't ask why, but when the model is loaded it causes a ROS error message and gazebo crashes if you open gazebo from terminal.)
Select the insert tab on the top left corner and click 'Add Path'. Select this folder: `catkin_ws/src/robot_model/models` and then click Open.
Now the UVC_robot model must appear in the list. Select it, and place an instance to the world.

Right click on the model and select 'Edit model'. This opens up the model editor view. Now you can select any part of the robot model
separately and modify their parameters. E.g. you can select the wheels and with right click you can open the link inspector.
Joints can be modified similarly through the joint inspector.