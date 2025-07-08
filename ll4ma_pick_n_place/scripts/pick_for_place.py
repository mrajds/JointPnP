#!/usr/bin/env python3

import rospy
import os
import sys
import gc
import torch
import numpy as np

from ll4ma_opt.solvers.penalty_method import AugmentedLagranMethod
from ll4ma_opt.solvers.optimization_solver import SolverReturn
import ll4ma_pick_n_place.placement_heuristics as heurs

from ll4ma_pick_n_place.task_place_problem import Task_Place_Problem
from ll4ma_pick_n_place.pick_for_problem import Pick_For_Problem
from ll4ma_pick_n_place.segmentation_client import SegmentationClient, get_objgrid
from ll4ma_pick_n_place.planning_client import get_display_state, display_robot_state, get_display_trajectory, display_robot_trajectory, moveit_client, reflex_client

from ll4ma_pick_n_place.data_utils import load_env_yaml
from ll4ma_pick_n_place.grasp_utils_client import GraspUtils

from gpu_sd_map.GenSDM import ConvNet

from sklearn.cluster import KMeans

PLACE_COST = 'corner'

def get_quad_convs(im, kern, contained):
    fig_loc = '/home/mohanraj/temp/'
    cs = []
    i = im
    k = kern.copy()
    for _ in range(4):
        conv = ConvNet([1,1]+list(k.shape))
        k = np.expand_dims(k, axis=0)
        k = np.expand_dims(k, axis=0)
        c = conv(i, k, contained=contained)
        c[c==0] = 0
        c[c>0] = 1
        cs.append(c)
        #pyplot.imshow(k[0][0],cmap='gray')
        #pyplot.savefig(fig_loc+f'q{_}.png')
        k = np.rot90(k[0][0]).copy()
    return cs

def define_place_heuristic():
    if PLACE_COST == 'corner':
        return heurs.corner_heuristic()
    return None

def place_init(pblm):
    rospy.loginfo('Generating coarse scene grid')
    SDG = pblm.get_3d_sd_grid()
    rospy.loginfo('Scene generated')
    OI = pblm.get_2d_obj_grid()
    rospy.loginfo('Object generated')
    SI = pblm.flatten_3d_grid(SDG, bound=False)#pblm.get_2d_sd_grid()
    COI = get_quad_convs(SI, OI, False)
    #RI, min_z, max_z = pblm.get_2d_gripper_kernel()
    #convr = ConvNet([1,1]+list(RI.shape))
    #SIr = pblm.flatten_3d_grid(SDG, min_z, max_z)
    #CRI = get_quad_convs(SIr, RI, True)
    CI = np.array(COI)# + np.array(CRI)

    free_pts = np.argwhere(CI==0)
    full_free_pts = np.zeros((free_pts.shape[0], 6))
    full_free_pts[:,:2] = free_pts[:,1:3] * 1e-2
    full_free_pts[:,5] = free_pts[:,0] * (np.pi/2)
    n_c = min(100, full_free_pts.shape[0])
    if n_c == 0:
        rospy.loginfo('No placement locations found')
        return None
    free_pts = KMeans(n_clusters=n_c, n_init="auto").fit(full_free_pts).cluster_centers_
    free_costs = np.zeros(free_pts.shape[0])
    for i in range(free_pts.shape[0]):
        free_costs[i] = place_heuristic.cost(free_pts[i])
    ranked_pts = free_pts[np.argsort(free_costs)]
    pblm.x0 = ranked_pts[0,[0,1,2,5]]
    return pblm.x0

if __name__ == '__main__':
    rospy.init_node('p4p_robot_exec_node')
    Env_List = {}
    ENV_YAML = rospy.get_param('env_yaml', None)

    if ENV_YAML is not None:
        Env_List = load_env_yaml(ENV_YAML)

    obj_grid = get_objgrid(True)
    graspclient = GraspUtils()
    graspclient.init_grasp_nets(obj_grid.sparse_grid, \
                                obj_grid.grid_size, \
                                obj_grid.true_size)
    
    place_heuristic = define_place_heuristic()
    pblm = Task_Place_Problem(obj_grid, Env_List, \
                              place_heuristic = define_place_heuristic())

    obj_grid.gen_sd_map()

    place_init(pblm)

    #pblm.publish_env_cloud()
    pblm.add_collision_constraints()

    solver = AugmentedLagranMethod(pblm, "BFGS", FP_PRECISION=1e-3)

    solret = solver.optimize(pblm.x0, max_iterations=40)

    place_pose_arr = pblm.full_pose(solret.solution)
    #obj_grid.set_pose(pblm.full_pose(solret.solution))

    Env_List['just_placed'] = obj_grid
    pblm.Env_List = Env_List
    pblm.publish_env_cloud()

    torch.cuda.empty_cache()
    
    pblm = Pick_For_Problem(obj_grid, Env_List, place_pose_arr, graspclient)

    pblm.add_collision_constraints()
    solver = AugmentedLagranMethod(pblm, "BFGS", FP_PRECISION=1e-3)

    solret = solver.optimize(pblm.x0, max_iterations=40)

    drs = get_display_state(solret.solution[pblm.G_IDX], pblm.arm_palm_KDL)

    dtraj = get_display_trajectory(solret.iterates[:, pblm.G_IDX, 0], pblm.arm_palm_KDL)

    for _ in range(100):
        display_robot_state(drs)
        #display_robot_state(drs_c, "grasp_corrected_state")
        #display_robot_state(drs_place, "place_robot_state")
        display_robot_trajectory(dtraj, "grasp_joint_iterates")
        #display_robot_trajectory(dtraj_place, "place_joint_iterates")
        rospy.sleep(0.01)
