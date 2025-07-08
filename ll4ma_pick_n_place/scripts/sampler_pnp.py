#!/usr/bin/env python3

import rospy
import os
import sys
import argparse
import pickle
from datetime import datetime
import numpy as np
import tf
import copy
import tf2_ros
import time
from matplotlib import pyplot, cm
import matplotlib
import multiprocessing as mp
import torch
import gc
import GPUtil


from sklearn.cluster import KMeans

from ll4ma_pick_n_place.segmentation_client import SegmentationClient, get_objgrid
from ll4ma_pick_n_place.opt_problem import PnP_Problem_CS, Grasp_Collision
import ll4ma_pick_n_place.placement_heuristics as heurs

from ll4ma_opt.solvers.penalty_method import AugmentedLagranMethod
from ll4ma_opt.solvers.optimization_solver import SolverReturn

from ll4ma_pick_n_place.planning_client import get_display_state, display_robot_state, get_display_trajectory, display_robot_trajectory, moveit_client, reflex_client

from gpu_sd_map.ros_transforms_lib import convert_array_to_pose, mat_to_vec
from gpu_sd_map.transforms_lib import vector2mat, invert_trans
from gpu_sd_map.GenSDM import ConvNet

from geometry_msgs.msg import Pose

from ll4ma_pick_n_place.data_utils import load_env_yaml, read_exp, write_exp

from rospkg import RosPack
rp = RosPack()
pkg_path = rp.get_path('ll4ma_pick_n_place')

ENV_YAML = None #'corner5_a.yaml' #'multi3_a.yaml'

ENVS_DIR = os.path.join(pkg_path, 'envs')

SOLUTION = None

#SOLUTION = np.array([1.849329157674892, 1.0653126387619622, 0.24689684796440842, -0.19932116532744099, -0.22147961225879811, 1.8849554494180354, 1.8304784322849441, 1.2636381883806913, -1.0149858558300786, 1.0934623236381966, 1.1314123549905988, -1.0839136574199097, -1.3579362717359729, -2.74889350231697, 0.7246652618679388])

if ENV_YAML is not None:
    ENV_PATH = os.path.join(ENVS_DIR, ENV_YAML)

GRASP_OBJ_FN = 'grasp_obj'

GRASP_MODE = None

#METHOD = 'then'
METHOD = 'joint'

OPTIMIZE = not True

PLACE_COST = 'corner'

MAX_ITERS = 40

def compute_grasp_offset(oTg, obj_dims, safe=0.05):
    '''
    This function checks if the grasp pose is in the bounding box of object, and moves
    the grasp to safe distance away if so.
    '''
    max_cos_dist = -np.inf
    max_axis = -1
    for i in range(3):
        vec = [0.0] * 3
        vec[i] = 1.0
        cos_dist = abs(np.dot(np.array(vec), oTg[:3,0]))
        if cos_dist > max_cos_dist:
            max_axis = i
            max_cos_dist = cos_dist
    print(max_axis, oTg[max_axis, 3], obj_dims[max_axis])
    offset = max(obj_dims[max_axis]/2 + safe - abs(oTg[max_axis, 3]), 0.0)
    return offset * max_cos_dist

def add_object_meshes(env_dict, planner):
    for key in env_dict.keys():
        VoxGrid = env_dict[key]
        VoxGrid.get_mesh(key+'.stl')
        rospy.loginfo(f'adding mesh {key}.stl....')
        pose = VoxGrid.get_pose()
        planner.add_object_mesh(key, pose.header.frame_id, pose.pose)
        rospy.loginfo(f'added')

def remove_object_meshes(env_dict, planner):
    for key in env_dict.keys():
        planner.remove_object(key)
        os.remove(key+'.stl')

def get_random_mid_pose(orientation):
    pose = Pose()
    pose.orientation = orientation
    pose.position.x = np.random.rand() - 0.5
    pose.position.y = np.random.rand() * 0.4 - 0.2
    pose.position.z = np.random.rand() * 1.5 - 0.3
    return pose

def invoke_solver(ps_pair):
    init, solver = ps_pair
    solret = solver.optimize(init, max_iterations=MAX_ITERS)

def save_init(x0):
    save_data = {}
    save_data['x0'] = x0.tolist()
    save_data['ENV_YAML'] = ENV_YAML
    save_data['PLACE_COST'] = PLACE_COST
    write_exp(save_data)
    
def set_inits(data):
    global ENV_YAML
    ENV_YAML = data['ENV_YAML']
    x0 = data['x0']
    global PLACE_COST
    PLACE_COST = data['PLACE_COST']
    return np.array(x0)
    
def parse_args():
    x0 = None
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_file', type=str, default=None, nargs='?')
    args = parser.parse_args()
    if args.exp_file is not None:
        data = read_exp(args.exp_file)
        x0 = set_inits(data)
    return x0

def define_place_heuristic():
    if PLACE_COST == 'corner':
        return heurs.corner_heuristic()
    return None

def get_wts():
    if METHOD == 'joint':
        return 100.0, 1.0
    elif METHOD == 'then':
        return 100.0, 0.0

def get_post_wts():
    if METHOD == 'joint':
        return 100.0, 1.0
    elif METHOD == 'then':
        return 0.0, 1.0

def get_quad_convs(im, kern, contained):
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
        k = np.rot90(k[0][0]).copy()
        #pyplot.imshow(k,cmap='gray')
        #pyplot.savefig(f'q{_}.png')
    return cs

def informed_init(pblm, num_inits=5):
    if METHOD == 'joint':
        if OPTIMIZE:
            return joint_init(pblm, num_inits)
        else:
            return joint_init(pblm, 50, full_cost=True)
    elif METHOD == 'then':
        return grasp_init(pblm, num_inits)
        
def grasp_init(pblm, num_inits=5):
    inits = []
    costs = []
    for ini in range(num_inits):
        #gidx = None
        gidx = GRASP_MODE
        if gidx is None:
            gidx = ini % 2
        print('Grasp Mode: ', gidx)
        pblm.initialize_grasp(gidx)
        if np.all(pblm.x0 == 0):
            rospy.loginfo('No feasilbe grasps found skipping')
            continue
        x0 =copy.deepcopy(pblm._process_q(pblm.x0))
        inits.append(x0)
        costs.append(pblm.cost(x0, split=True)[0])
    print(costs)
    return np.array(inits)[np.argsort(costs)]

def place_init(pblm):
    rospy.loginfo('Generating coarse scene grid')
    SDG = pblm.get_3d_sd_grid()
    rospy.loginfo('Scene generated')
    OI = pblm.get_2d_obj_grid()
    rospy.loginfo('Object generated')
    SI = pblm.flatten_3d_grid(SDG, bound=False)#pblm.get_2d_sd_grid()
    COI = get_quad_convs(SI, OI, False)
    RI, min_z, max_z = pblm.get_2d_gripper_kernel()
    convr = ConvNet([1,1]+list(RI.shape))
    SIr = pblm.flatten_3d_grid(SDG, min_z, max_z)
    CRI = get_quad_convs(SIr, RI, True)
    CI = np.array(COI) + np.array(CRI)

    free_pts = np.argwhere(CI==0)
    full_free_pts = np.zeros((free_pts.shape[0], 6))
    full_free_pts[:,:2] = free_pts[:,1:3] * 1e-2
    full_free_pts[:,5] = free_pts[:,0] * (np.pi/2)
    n_c = min(100, full_free_pts.shape[0])
    if n_c == 0:
        rospy.loginfo('No placement locations found')
        return None
    free_pts = KMeans(n_clusters=n_c, n_init="auto").fit(full_free_pts).cluster_centers_
    #free_pts = free_pts * 1e-2
    free_costs = np.zeros(free_pts.shape[0])
    for i in range(free_pts.shape[0]):
        free_costs[i] = place_heuristic.cost(free_pts[i])
    ranked_pts = free_pts[np.argsort(free_costs)]
    #ranked_pts = ranked_pts * 1e-3
    pblm.initialize_place(ranked_pts.shape[0], ranked_pts)
    return pblm.x0
        

def joint_init(pblm, num_inits=5, full_cost=False, debug=False):
    inits = []
    costs = []
    rospy.loginfo('Generating coarse scene grid')
    SDG = pblm.get_3d_sd_grid()
    rospy.loginfo('Scene generated')
    OI = pblm.get_2d_obj_grid()
    rospy.loginfo('Object generated')
    SI = pblm.flatten_3d_grid(SDG, bound=False)#pblm.get_2d_sd_grid()
    COI = get_quad_convs(SI, OI, False)

    if debug:
        #pyplot.imshow(RI[0][0])
        #pyplot.show()
        pyplot.imshow(OI, cmap='gray')
        pyplot.savefig(f'obj.png')
        pyplot.imshow(SI, cmap='gray')
        pyplot.savefig(f'env.png')
        for m in range(4):
            pyplot.imshow(COI[m], cmap='gray')
            pyplot.savefig(f'o_{m}.png')
    

    for ini in range(num_inits):
        #gidx = None
        gidx = GRASP_MODE
        if gidx is None:
            gidx = ini % 2
        print('Grasp Mode: ', gidx)
        pblm.initialize_grasp(gidx)
        if np.all(pblm.x0 == 0):
            rospy.loginfo('No feasilbe grasps found skipping')
            continue
        RI, min_z, max_z = pblm.get_2d_gripper_kernel()
        convr = ConvNet([1,1]+list(RI.shape))
        #print(min_z, max_z)
        SIr = pblm.flatten_3d_grid(SDG, min_z, max_z)
        CRI = get_quad_convs(SIr, RI, True)
        
        CI = np.array(COI) + np.array(CRI)

        #for _ in range(4):
        #    pyplot.imshow(CI[0])
        #    pyplot.show()

        free_pts = np.argwhere(CI==0)
        full_free_pts = np.zeros((free_pts.shape[0], 6))
        full_free_pts[:,:2] = free_pts[:,1:3] * 1e-2
        full_free_pts[:,5] = free_pts[:,0] * (np.pi/2)
        #print(full_free_pts)
        n_c = min(100, full_free_pts.shape[0])
        if n_c == 0:
            rospy.loginfo('No placement locations found')
            continue
        free_pts = KMeans(n_clusters=n_c, n_init="auto").fit(full_free_pts).cluster_centers_
        #free_pts = free_pts * 1e-2
        free_costs = np.zeros(free_pts.shape[0])
        for i in range(free_pts.shape[0]):
            free_costs[i] = place_heuristic.cost(free_pts[i])
        ranked_pts = free_pts[np.argsort(free_costs)]
        #ranked_pts = ranked_pts * 1e-3
        if not pblm.initialize_place(ranked_pts.shape[0], ranked_pts):
            rospy.loginfo('No placement configurations found')
            continue
        x0 =copy.deepcopy(pblm._process_q(pblm.x0))
        drs = get_display_state(pblm.x0[pblm.G_IDX], pblm.arm_palm_KDL)
        drs_place = get_display_state(pblm.x0[pblm.P_IDX], pblm.arm_palm_KDL)
        display_robot_state(drs)
        display_robot_state(drs_place, "place_robot_state")
        if debug:
            #pyplot.imshow(RI[0][0])
            #pyplot.show()
            pyplot.imshow(RI, cmap='gray')
            pyplot.savefig(f'{ini}_r.png')
            for m in range(4):
                pyplot.imshow(CRI[m], cmap='gray')
                pyplot.savefig(f'r{ini}_{m}.png')
                pyplot.imshow(CI[m], cmap='gray')
                pyplot.savefig(f'{ini}_{m}.png')
            #pyplot.show()
        inits.append(x0)
        if full_cost:
            costs.append(pblm.cost(x0))
        else:
            costs.append(pblm.cost(x0, split=True)[1])
    #print(inits)
    print(costs)
    return np.array(inits)[np.argsort(costs)]

if __name__ == '__main__':
    rospy.init_node('pnp_robot_exec_node')
    x0 = parse_args() 
    hand = reflex_client()
    planner = moveit_client(ns="left")

    Env_List = {}
    ENV_YAML = rospy.get_param('env_yaml', None)

    if ENV_YAML is not None:
        Env_List = load_env_yaml(ENV_YAML)

    pool = mp.Pool(processes=2)
    
    def shutdown():
        #delete_model_client(obj_n)
        planner.detach_object()
        hand.release()
        hand.set_preshape(0.0)
        home_traj = planner.go_home()
        if input('execute?').lower() == 'y':
            planner.execute_plan(home_traj)
        planner.remove_object(GRASP_OBJ_FN)
        os.remove(GRASP_OBJ_FN+'.stl')
        #remove_object_meshes(Env_List, planner)

    rospy.on_shutdown(shutdown)

    while not rospy.is_shutdown():
        hand.release()
        hand.set_preshape(0.0)
        home_traj = planner.go_home()

        if input('execute?').lower() == 'y':
            planner.execute_plan(home_traj)
        #obj_n = input('Input object identifier: ')
        obj_grid = get_objgrid(True)
        #obj_grid.gen_sd_map()

        g_wt, p_wt = get_wts()
        place_heuristic = define_place_heuristic()
        pblm = PnP_Problem_CS(obj_grid, Env_List, G_Wt=g_wt, P_Wt=p_wt,\
                              place_heuristic = define_place_heuristic(), \
                              grasp_mode=GRASP_MODE)

        start = time.time()
        rospy.loginfo("Generating sdf...")
        obj_grid.gen_sd_map()
        rospy.loginfo("sdf generated, adding mesh to moveit")
        obj_grid.get_mesh()
        planner.add_object_mesh(GRASP_OBJ_FN)
        rospy.loginfo("mesh added")
        rospy.loginfo('initializing...')
        if SOLUTION is None:
            inits = informed_init(pblm, 10)
            #exit(0)
        else:
            inits = [SOLUTION]
        init_time = time.time() - start
        #print(inits)

        #pblm.Env_List = Env_List
        pblm.publish_env_cloud()
        rospy.loginfo("Problem initialized")
        #add_object_meshes(Env_List, planner)
        #planner.add_object_bb(obj_grid.true_size[[1,0,2]])
        #input('check memory')
        pblm.add_collision_constraints()
        solver = AugmentedLagranMethod(pblm, "BFGS", FP_PRECISION=1e-3)

        lowest_fitness = 1000
        lowest_violator = None

        start = time.time()
        
        for ninit,init in enumerate(inits[:]):
            pblm.x0 = pblm._process_q(init)
        
            if x0 is not None:
                print('\n\n\n\n\n',pblm.x0)
                pblm.x0 = pblm._process_q(inits[0])
            else:
                save_init(pblm.x0)

            drs = get_display_state(pblm.x0[pblm.G_IDX], pblm.arm_palm_KDL)
            drs_place = get_display_state(pblm.x0[pblm.P_IDX], pblm.arm_palm_KDL)
            display_robot_state(drs)
            display_robot_state(drs_place, "place_robot_state")
        
            #pblm2 = PnP_Problem_CS(obj_grid, G_Wt=100.0, P_Wt=1.0, grasp_mode=GRASP_MODE)
            #pblm2.Env_List = Env_List
            #pblm2.add_collision_constraints()
        
            #pblm2 = copy.deepcopy(pblm)
            #solver2 = AugmentedLagranMethod(pblm, "BFGS")

            #multi_solve_args = [(pblm.x0, solver), \
            #                    (pblm2.x0, solver2)]
            
            #pool.map(invoke_solver, multi_solve_args)

        

            if (SOLUTION is None) and OPTIMIZE:
                solret = solver.optimize(pblm.x0, max_iterations=MAX_ITERS)
            else:
                solret = SolverReturn
                if SOLUTION is not None:
                    solret.solution = SOLUTION
                    solret.iterates = SOLUTION.reshape(1,-1,1)
                else:
                    fitness = pblm.evaluate_constraints(pblm.x0)
                    fitness = max(list(fitness.values())[30:])
                    if fitness < lowest_fitness:
                        lowest_fitness = fitness
                        lowest_violator = copy.deepcopy(pblm.x0)
                    
                    if fitness > 1e-2:
                        rospy.loginfo(f"High fitness {fitness}, skipping for now...")
                        if pblm.cost(pblm.x0, split=True)[0] > 1 or ninit>10:
                            rospy.loginfo(f"Good grasps exhausted, proceeding with the least violator")
                            pblm.x0 = lowest_violator
                        else:
                            continue
                    st = time.time()
                    solution = pblm.sub_sample()
                    solret.time = time.time() - start
                    solret.solution = solution
                    solret.iterates = solution.reshape(1,-1,1)
            print('Solution:', solret.solution.tolist(),'\nCost:', solret.cost)
            print('Number of steps:', len(solret.iterates)-1)
            print('Solve Time:', solret.time)
            grasp_cost, place_cost = pblm.cost(solret.solution, split=True)
            print('Grasp cost:', grasp_cost, '\tPlace cost:', place_cost)
            grasp_probs = pblm.grasp_probabilities(solret.solution)
            print('Grasp lkh:', grasp_probs[1], '\tGrasp Prior:', grasp_probs[2])
            print('Excel String:\n', \
                  'Grasp Cost \t', \
                  'Place Cost \t', \
                  'Grasp Lkh \t', \
                  'Grasp Prior \t', \
                  'Init time \t', \
                  'Solve time \t', \
                  'Total time \t')
            print(grasp_cost, '\t', \
                  place_cost, '\t', \
                  grasp_probs[1], '\t', \
                  grasp_probs[2], '\t', \
                  init_time, '\t',\
                  solret.time, '\t',\
                  init_time + solret.time)

            print('Performance Stats: (num_calls, total_time)')
            perf_stats = pblm.performance_stats()
            print('Problem Cost   :', perf_stats[0])
            print('Problem Grads  :', perf_stats[1])
            print('Constraints    :')
            print('===========')
            print('Grasp Cost     :', perf_stats[2])
            print('Grasp Grads    :', perf_stats[3])
            print('Placement Cost :', perf_stats[4])
            print('Placement Grads:', perf_stats[5])
            print('Collision Cost :', perf_stats[6])
            print('Collision Grads:', perf_stats[7])
            print('Gripper Cost   :', perf_stats[8])
            print('Gripper Grads  :', perf_stats[9])

            fitness = pblm.evaluate_constraints(solret.solution)
            for const in fitness:
                if fitness[const] > 1e-3:
                    rospy.loginfo(f"{const} violates with fitness: {fitness[const]}")

            if METHOD != 'joint':
                # GPUtil.showUtilization()
                # pblm.delete()
                # del pblm.GraspClient
                # del pblm
                # print(pblm.GraspClient.get_torch_memory())
                # gc.collect()
                # torch.cuda.empty_cache()
                # #print(Grasp_Collision.pblm)
                # GPUtil.showUtilization()
                # print(torch.cuda.memory_summary())
                g_wt, p_wt = get_post_wts()
                pblm.cull()
                pblm.Wt_G = g_wt
                pblm.Wt_P = p_wt
                #pblm = PnP_Problem_CS(obj_grid, G_Wt=g_wt, P_Wt=p_wt,\
                #                  place_heuristic = define_place_heuristic(), \
                #                  grasp_mode=GRASP_MODE)
                pblm.add_constraints()
                pblm.x0 = solret.solution
                pblm.add_collision_constraints()
                start = time.time()
                pblm.x0 = place_init(pblm)
                if pblm.x0 is None:
                    rospy.logwarn("Failed to find any feasible place init for grasp")
                    break
                init_time2 = time.time() - start
                
                solret2 = solver.optimize(pblm.x0, max_iterations=MAX_ITERS)
                fitness = pblm.evaluate_constraints(solret2.solution)

                print('Solution:', solret2.solution.tolist(),'\nCost:', solret2.cost)
                print('Number of steps:', len(solret2.iterates)-1)
                print('Solve Time:', solret2.time)
                grasp_cost, place_cost = pblm.cost(solret2.solution, split=True)
                print('Grasp cost:', grasp_cost, '\tPlace cost:', place_cost)
                grasp_probs = pblm.grasp_probabilities(solret2.solution)
                print('Grasp lkh:', grasp_probs[1], '\tGrasp Prior:', grasp_probs[2])
                print('Excel String:\n', \
                      'Grasp Cost \t', \
                      'Place Cost \t', \
                      'Grasp Lkh \t', \
                      'Grasp Prior \t', \
                      'Init time \t', \
                      'Solve time \t', \
                      'Total time \t')
                print(grasp_cost, '\t', \
                      place_cost, '\t', \
                      grasp_probs[1], '\t', \
                      grasp_probs[2], '\t', \
                      init_time+init_time2, '\t',\
                      solret.time+solret2.time, '\t',\
                      init_time + solret.time + init_time2 + solret2.time)
            
                for const in fitness:
                    if fitness[const] > 1e-3:
                        rospy.loginfo(f"{const} violates with fitness: {fitness[const]}")
                solret2.iterates = np.vstack([solret.iterates, solret2.iterates])
                solret = solret2

            oTg = pblm.joint_to_grasp(solret.solution, asmat=True)
            grasp_correction = compute_grasp_offset(oTg, obj_grid.true_size)
            #rospy.loginfo(f"Grasp Correction: {grasp_correction}")
            orient_tols, pnp_omegas = pblm.get_pnp_diff(solret.solution)
            #rospy.loginfo(f'Desired Task Omegas: {pnp_omegas}')
            drs = get_display_state(solret.solution[pblm.G_IDX], pblm.arm_palm_KDL)

            #while grasp_correction > 1e-3:
            #    solret.solution[pblm.G_IDX] = pblm.get_approach_head(solret.solution[pblm.G_IDX], -grasp_correction)
            #    oTg = pblm.joint_to_grasp(solret.solution, asmat=True)
            #    grasp_correction = compute_grasp_offset(oTg, obj_grid.true_size)
            
            #rospy.loginfo(f"Correction Sanity Check: {grasp_correction}")

            #pre_grasp_joints = pblm.get_approach_head(solret.solution[pblm.G_IDX])
            #pre_place_joints = pblm.get_approach_head(solret.solution[pblm.P_IDX])

            #drs_c = get_display_state(solret.solution[pblm.G_IDX], pblm.arm_palm_KDL)
            #drs = get_display_state(pre_grasp_joints, pblm.arm_palm_KDL)

            #drs_place = get_display_state(solret.solution[pblm.P_IDX], pblm.arm_palm_KDL)
            drs_place = get_display_state(solret.solution[pblm.P_IDX], pblm.arm_palm_KDL)

            dtraj = get_display_trajectory(solret.iterates[:, pblm.G_IDX, 0], pblm.arm_palm_KDL)
            dtraj_place = get_display_trajectory(solret.iterates[:, pblm.P_IDX, 0], pblm.arm_palm_KDL)

            for _ in range(100):
                display_robot_state(drs)
                #display_robot_state(drs_c, "grasp_corrected_state")
                display_robot_state(drs_place, "place_robot_state")
                display_robot_trajectory(dtraj, "grasp_joint_iterates")
                display_robot_trajectory(dtraj_place, "place_joint_iterates")
                rospy.sleep(0.01)

            hand.set_preshape(solret.solution[pblm.PRESHAPE])


            iter_range = range(len(solret.iterates))
            
            for key in pblm.LogDict.keys():
                if key in pblm.COST_TERMS and not key.startswith('reg'):
                    pyplot.plot(iter_range, pblm.extract_iters(key, solret.iterates), label=key)
            pyplot.legend()
            pyplot.ylim(-1,2)
            pyplot.show()

            pre_grasp_joints = pblm.get_approach_head(solret.solution[pblm.G_IDX])
            rospy.loginfo("Planning to get grasp...")
            executing = True
            if not planner.plan_and_exec_loop(pre_grasp_joints, rrt=True):
                executing = False
                #break

            if executing:
                grasp_q_vel = solret.solution[pblm.G_IDX] - pre_grasp_joints
                rospy.loginfo("Planning to approach...")
                grasp_traj = planner.q_vel_traj(pblm.arm_palm_KDL, grasp_q_vel)
                #uip=input('Approach?')
                #if uip.lower() in ['y']:
                planner.execute_plan(grasp_traj)
        
                hand.grasp()
                rospy.sleep(2)
                planner.remove_object(GRASP_OBJ_FN)

                lift_target = pblm.place_kdl.fk(solret.solution[pblm.P_IDX])
            
                #lift_traj = planner.plan_to_joint(pblm.get_lift_up_joints(solret.solution[pblm.G_IDX]))

                lift_traj = planner.lift_up(pblm.arm_palm_KDL, solret.solution[pblm.G_IDX], target = lift_target)

                uip = input('lift?')
                if uip.lower() == 'y':
                    planner.execute_plan(lift_traj)

            uip = input('proceed?')
            if not uip.lower() == 'y':
                executing = False

            if executing:
                grasp_trans = invert_trans(pblm.joint_to_grasp(solret.solution, asmat=True))
                grasp_pose = convert_array_to_pose(mat_to_vec(grasp_trans), 'reflex_palm_link')
                planner.attach_object(grasp_pose, obj_grid.true_size[[1,0,2]])

        
            gp = pblm.grasp_kdl.fk(solret.solution[pblm.G_IDX])
            grasp_mat = vector2mat(gp[:3], gp[3:])
            up = np.array([0, 0, 1])
            cos_dist = 0.0
            up_id = -1
            for i in range(3):
                if abs(np.dot(up, grasp_mat[:3, i])) > cos_dist:
                    cos_dist = abs(np.dot(up, grasp_mat[:3, i]))
                    up_id = i
            base_tol = 1
            orient_tols = [base_tol] * 3
            orient_tols[up_id] = np.pi * 2

            rospy.loginfo(f"Orientation tolerances set as {orient_tols}")

            place_pose = convert_array_to_pose(pblm.place_kdl.fk(solret.solution[pblm.P_IDX]),\
                                               'reflex_palm_link')

            pre_place_joints = pblm.get_lift_up_joints(solret.solution[pblm.P_IDX], dist=0.1)
        
            if not (executing and planner.plan_and_exec_loop(pre_place_joints, \
                                    place_pose.pose.orientation, orient_tols)):
                executing = False
                #break

            if executing:
                put_vel = [0.0] * 6
                put_vel[2] = -0.1
                
                put_q_vel = solret.solution[pblm.P_IDX] - pre_place_joints
            
                put_traj = planner.q_vel_traj(pblm.arm_palm_KDL, put_q_vel)

                planner.execute_plan(put_traj)

                rospy.sleep(0.5)
                hand.release()

                #lift_traj = planner.lift_up()
                lift_traj = planner.lift_up(pblm.arm_palm_KDL)#, solret.solution[pblm.P_IDX])
                #lift_traj = planner.plan_to_joint(pblm.get_lift_up_joints(solret.solution[pblm.P_IDX]))
                planner.execute_plan(lift_traj)
                hand.release()

            planner.detach_object()

            home_traj = planner.go_home()

            if input('execute?').lower() == 'y':
                planner.execute_plan(home_traj)

            if not input('Redo with next init?').lower() == 'y':
                break

        obj_grid.set_pose(pblm.object_pose_from_joints(solret.solution))
        #obj_grid.gen_sd_map()
        #input('check memory')
        #rospy.loginfo('SDF Generated')

        uip = input("Proceed with next object?")
        if uip.lower() not in ['y', 'yes']:
            rospy.loginfo("Aborting...")
            rospy.signal_shutdown("User aborted execution")
        
        obj_n = None
        while (not obj_n) or (obj_n in Env_List): 
            obj_n = input('Input object identifier: ')
        Env_List[obj_n] = obj_grid
        pblm.Env_List = Env_List
        pblm.publish_env_cloud()
        #add_object_meshes(Env_List, planner)
        planner.remove_object(GRASP_OBJ_FN)
    
    #while not rospy.is_shutdown():
        
        ##drs = get_display_state(solret.iterates[i][pblm.G_IDX], pblm.arm_palm_KDL)
        #display_robot_state(drs)
        #display_robot_state(drs_place, "place_robot_state")
        #rospy.sleep(0.01)
