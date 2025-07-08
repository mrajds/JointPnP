#!/usr/bin/env python3

import rospy
import trimesh
import numpy as np
from ll4ma_pick_n_place.segmentation_client import SegmentationClient, get_objgrid
from ll4ma_pick_n_place.opt_problem import PnP_Problem_CS
from gpu_sd_map.transforms_lib import transform_points, vector2mat
from gpu_sd_map.gripper_augment import GripperVox, ReflexAug

if __name__ == '__main__':
    rospy.init_node('mesh_aug_viz_node')
    obj_grid = get_objgrid(True)
    pblm = PnP_Problem_CS(obj_grid, {})
    obj_grid.gen_sd_map()
    obj_grid.get_mesh()
    
    Pkg_Path = "/home/mohanraj/robot_WS/src/gpu_sd_map"
    Base_BV_Path = Pkg_Path + "/gripper_augmentation/base_hd.binvox" 
    FingerSW_BV_Path = Pkg_Path + "/gripper_augmentation/finger_sweep_hd.binvox"
    Finger_BV_Path = Pkg_Path + "/gripper_augmentation/finger_hd.binvox"
    Base = GripperVox(Base_BV_Path)
    FingerSW = GripperVox(FingerSW_BV_Path)
    Finger = GripperVox(Finger_BV_Path)
    
    gripper = ReflexAug(Base, Finger)

    reflex_mesh = trimesh.voxel.ops.points_to_marching_cubes(gripper.Set_Pose_Preshape(), pitch=0.001)
    reflex_mesh.export('reflex.stl')
    
    gripper_aug = ReflexAug(Base, FingerSW)

    mean = pblm.GraspClient.get_mean(1)

    gripper_pts = gripper_aug.Set_Pose_Preshape(preshape=mean[6])
    oTg = vector2mat(mean[:3], mean[3:6])

    gripper_pts = transform_points(gripper_pts, oTg)

    obj_pts = obj_grid.get_level_set(-0.001, 0)

    mesh = trimesh.voxel.ops.points_to_marching_cubes(obj_pts, pitch=0.001)

    mesh.export('augobj.stl')


    mesh = trimesh.voxel.ops.points_to_marching_cubes(gripper_pts, pitch=0.001)

    mesh.export('auggripper.stl')
