# JointPnP
Code for Joint pick and place planning paper https://doi.org/10.1109/LRA.2024.3360892

## Dependencies:
- ROS Noetic
- [ll4ma-opt-sandbox](https://bitbucket.org/robot-learning/ll4ma-opt-sandbox/src/main/) - For optimization solver
- [point_cloud_segmentation](https://bitbucket.org/robot-learning/point_cloud_segmentation/src/main/) - For segmentation from simulation or real robot

## Installation:
1. Clone this repo to a catkin workspace
2. Build using catkin tools.

## Sub-Modules:
1. ll4ma_pick_n_place - Defines the pick and place optimization problems
2. prob_grasp_planner - Learned grasp success classifier and MDN prior based on https://doi.org/10.1109/MRA.2020.2976322, reimplemented for reflex hand.
3. gpu_sd_map - Optional module to generated SDF from partial view pointclouds (could be replaced with other sdf libraries)
