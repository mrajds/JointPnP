# JointPnP
Code for ["Pick and Place Planning is Better Than Pick Planning Then Place Planning"](https://doi.org/10.1109/LRA.2024.3360892), IEEE RA-L 2024

## Dependencies:
- ROS Noetic
- [ll4ma-opt-sandbox](https://bitbucket.org/robot-learning/ll4ma-opt-sandbox/src/main/) - For optimization solver
- [point_cloud_segmentation](https://bitbucket.org/robot-learning/point_cloud_segmentation/src/main/) - For segmentation from simulation or real robot

## Installation:
1. Clone this repo to a catkin workspace
2. Build using catkin tools.
3. Copy reflex grasp models from [Hugging Face](https://huggingface.co/ll4ma-lab/ReflexGraspModels) into prob_grasp_planner/model.
4. Update model paths in prob_grasp_planner/src/prob_grasp_planner/grasp_voxel_planner/

## Sub-Modules:
1. ll4ma_pick_n_place - Defines the pick and place optimization problems
2. prob_grasp_planner - Learned grasp success classifier and MDN prior based on https://doi.org/10.1109/MRA.2020.2976322, reimplemented for reflex hand.
3. gpu_sd_map - Optional module to generated SDF from partial view pointclouds (could be replaced with other sdf libraries)
