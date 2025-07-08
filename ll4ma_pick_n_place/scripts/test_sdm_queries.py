#!/usr/bin/env python3

import rospy
#import kaolin
import torch
import os
import sys
import pickle
from datetime import datetime
import trimesh
import numpy as np
import tf
import copy
import tf2_ros
import time
import getch
import tkinter

from ll4ma_pick_n_place.opt_problem import PnP_Problem, PnP_Problem_CS, publish_env_cloud
from ll4ma_pick_n_place.gazebo_utils import *
from ll4ma_pick_n_place.planning_client import moveit_client, reflex_client, \
    get_display_state, display_robot_state, get_display_trajectory, display_robot_trajectory
from ll4ma_pick_n_place.segmentation_client import SegmentationClient
from ll4ma_pick_n_place.grasp_utils_client import GraspUtils
from ll4ma_pick_n_place.data_utils import save_problem, restore_problem
from ll4ma_pick_n_place.visualizations import map_rgb, numpy_to_pointcloud2, publish_pointcloud2, \
    publish_2point_vectors

from ll4ma_opt.solvers.line_search import GradientDescent, GaussNewtonMethod, NewtonMethodWolfeLine
from ll4ma_opt.solvers.quasi_newton import BFGSMethod
from ll4ma_opt.solvers.penalty_method import PenaltyMethod, AugmentedLagranMethod
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from geometry_msgs.msg import Pose, WrenchStamped, Quaternion, PoseStamped, Point
from sensor_msgs.msg import JointState
import std_msgs.msg

from grasp_pipeline.srv import *
from prob_grasp_planner.srv import *

from pointsdf_reconstruction.srv import Reconstruct, ReconstructRequest

from gpu_sd_map.srv import CreateEnv, CreateEnvRequest, AddObj2Env, AddObj2EnvRequest
from gpu_sd_map.environment_manager import EnvVox, ObjGrid
from gpu_sd_map.GenSDM import write_shaded_image, min_from_sdm3d, gen_dsdf_y, gen_dsdf_x, flatten_vox
from gpu_sd_map.ros_transforms_lib import convert_array_to_pose, pose_to_array, TransformPose, \
    pose_to_6dof, publish_as_point_cloud

from gpu_sd_map.transforms_lib import transform_points, invert_trans, vector2mat

from prob_grasp_planner.grasp_common_library.data_proc_lib import DataProcLib
from matplotlib import pyplot, cm

OBJ_DATASET_DIR_ = "/home/mohanraj/ycb" #Set this to the local location of YCB dataeset

#OBJ_NAME_ = ["025_mug", "024_bowl", "054_softball"]
OBJ_NAMES_ = ["003_cracker_box", "003_cracker_box", "025_mug", "006_mustard_bottle", "054_softball"]

SESSIONS_DIR = "Sessions/"

def obj_env_collision_test(Obj_Grid, Egrid):
    tTo = Obj_Grid.Trans
    obj_pts = Obj_Grid.obj_pts
    query_pts = transform_points(obj_pts, tTo) + [0.0, 0.0, 0.0]
    print(query_pts.max(0))
    img_grid = np.zeros((query_pts.max(0)*1000 + 10).astype(np.int)) + 255
    sdm, dsd = Egrid.query_points(query_pts)
    #print(sdm.min())
    idx = (query_pts*1000).astype(np.int)
    for i in range(idx.shape[0]):
        #if sdm[i] - 5 <= 0:
        print(obj_pts[i], sdm[i])
        if np.all(idx[i] > 0):
        #    print(idx[i] , sdm[i])
            img_grid[idx[i][0], idx[i][1], idx[i][2]] = sdm[i] - 5
    return img_grid

def kaolin_test(Obj_Grid):
    obj_pts = Obj_Grid.obj_pts
    kaog = kaolin.ops.conversions.pointclouds_to_voxelgrids(torch.tensor([obj_pts]), 1000)
    
    print(torch.max(kaog))

def gen_query_grid_pts(l=0.5, w=0.5, res=1000):
    x = np.arange(.0, l, 1/res)
    y = np.arange(.0, w, 1/res)
    G = np.array(np.meshgrid(x,y)).T.reshape(-1,2)
    G = np.hstack([G, np.zeros((G.shape[0],1))])
    return G

DX_ = np.array([1e-5, 0.0, 0.0])
DY_ = np.array([0.0, 1e-5, 0.0])
DZ_ = np.array([0.0, 0.0, 1e-5])

def query_points_at_position(grid_pts, position):
    QTrans = vector2mat(position)
    query_pts = transform_points(grid_pts, QTrans)
    min_sd = None
    for key in Env_List.keys():
        #keys = [*Env_List]
        sd, dsd = Env_List[key].query_points(query_pts)
        if min_sd is None:
            min_sd = sd
        else:
            min_sd = np.minimum(sd, min_sd)
    colors = map_rgb(min_sd)
    return numpy_to_pointcloud2(query_pts, colors, 'place_table_corner')

def query_grads_at_position(full_pts, position, delt=1e-9):
    QTrans = vector2mat(position)
    grid_pts = np.unique(np.around(full_pts, decimals=2), axis=0)
    query_pts = transform_points(grid_pts, QTrans)
    min_sd = None
    min_dsd = None
    DX_[0] = delt
    DY_[1] = delt
    DZ_[2] = delt
    for key in Env_List.keys():
        sd = np.array(Env_List[key].query_points(query_pts)[0])
        dsdx = Env_List[key].query_points(query_pts + DX_)[0] \
               - Env_List[key].query_points(query_pts - DX_)[0]
        dsdx /= (2*delt) * 1000
        dsdx = dsdx.reshape(-1,1)
        dsdy = Env_List[key].query_points(query_pts + DY_)[0] \
               - Env_List[key].query_points(query_pts - DY_)[0]
        dsdy /= (2*delt) * 1000
        dsdy = dsdy.reshape(-1,1)
        dsdz = Env_List[key].query_points(query_pts + DZ_)[0] \
               - Env_List[key].query_points(query_pts - DZ_)[0]
        dsdz /= (2*delt) * 1000
        dsdz = dsdz.reshape(-1,1)
        dsd = np.hstack([dsdx, dsdy, dsdz])
        if min_sd is None:
            min_sd = sd
            min_dsd = dsd
        else:
            min_sd = np.minimum(sd, min_sd)
            min_dsd[(min_sd == sd)] = dsd[(sd == min_sd)]
    mag_dsd = np.linalg.norm(min_dsd, axis=1)
    colors = map_rgb(mag_dsd)
    colors[:] = [0.0, 1.0, 0.0, 0.0] 
    publish_2point_vectors(query_pts, query_pts + min_dsd, colors, 'place_table_corner')

def gui(grid_pts):
    root = tkinter.Tk()
    position = [0.0, 0.0, 0.0]
    
    def gui_update(i,val):
        position[i] = float(val)
        pc2 = query_points_at_position(grid_pts, position)
        publish_pointcloud2(pc2, 'sdm_test_grid')

    def publish_grads():
        print("Publishing grads")
        query_grads_at_position(grid_pts, position)
        

    def plot_profile():
        print(f"Enter Direction Parameters from {position}\n")
        a = float(input("x:"))
        b = float(input("y:"))
        c = float(input("z:"))
        pos = np.array([position])
        pos2 = np.array([a,b,c])
        num = 1e5
        t = np.arange(-num, num+1)
        pts = pos + np.einsum('i,j->ij', t, pos2)
        for key in Env_List.keys():
            sd = np.array(Env_List[key].query_points(pts)[0])
            pyplot.plot(t, sd)
        pyplot.show()
        

    sx = tkinter.Scale(root, from_=1., to=-1., resolution=1e-3, command=lambda v: gui_update(0, v))
    sx.pack()
    sy = tkinter.Scale(root, from_=1., to=-1., resolution=1e-3, command=lambda v: gui_update(1, v))
    sy.pack()
    sz = tkinter.Scale(root, from_=1., to=-1., resolution=1e-3, command=lambda v: gui_update(2, v))
    sz.pack()
    kdel = 1e-2
    root.bind("<Up>", lambda e: sy.set(sy.get()+kdel))
    root.bind("<Down>", lambda e: sy.set(sy.get()-kdel))
    root.bind("<Left>", lambda e: sx.set(sx.get()-kdel))
    root.bind("<Right>", lambda e: sx.set(sx.get()+kdel))
    root.bind("<Key-KP_Add>", lambda e: sz.set(sz.get()+kdel))
    root.bind("<Key-KP_Subtract>", lambda e: sz.set(sz.get()-kdel))
    root.bind("<Return>", lambda e: publish_grads())
    root.bind("p", lambda e: plot_profile())
    root.mainloop()
    
if __name__ == '__main__':
    rospy.init_node('pnp_scene_test_node')
    old_Env_List = None
    if len(sys.argv) < 2:
        #pkl_f = SESSIONS_DIR + datetime.now().strftime("%h%d%y_%H%M.pkl")
        rospy.logerror("Provide solved problem session for SDM")
    else:
        pkl_f = sys.argv[1]
        rospy.loginfo(f"Loading session :{sys.argv[1]}")
        Env_List, Obj_Grid, _, _, _, _ = restore_problem(pkl_f)
    #pblm = PnP_Problem_CS(Obj_Grid)
    #pblm.Env_List = Env_List
    publish_env_cloud(Env_List)
    grid_pts = gen_query_grid_pts()
    Zr = np.arange(-0.2, 0.2, 1e-2)
    z_max = np,max(Zr)
    position = [0.0, 0.0, 0.0]
    delt = 1e-2
    gui(grid_pts)
    rospy.signal_shutdown("User aborted execution")
    while not rospy.is_shutdown():
        pc2 = query_points_at_position(grid_pts, position)
        publish_pointcloud2(pc2, 'sdm_test_grid')
        k=ord(getch.getch())
        if k==65:
            position[1] += delt
        elif k==66:
            position[1] -= delt
        elif k==68:
            position[0] -= delt
        elif k==67:
            position[0] += delt
        elif k==43:
            position[2] += delt
        elif k==45:
            position[2] -= delt
        elif k==113:
            rospy.signal_shutdown("User aborted execution")
    exit(0)
