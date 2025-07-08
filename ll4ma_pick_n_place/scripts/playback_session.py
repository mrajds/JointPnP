#!/usr/bin/env python3

import rospy
import os
import sys
import argparse
import pickle
from datetime import datetime
import trimesh
import numpy as np
import tf
import copy
import tf2_ros
import time
from ll4ma_pick_n_place.opt_problem import PnP_Problem, PnP_Problem_CS
from ll4ma_pick_n_place.gazebo_utils import *
from ll4ma_pick_n_place.planning_client import moveit_client, reflex_client, \
    get_display_state, display_robot_state, get_display_trajectory, display_robot_trajectory
from ll4ma_pick_n_place.segmentation_client import SegmentationClient
from ll4ma_pick_n_place.grasp_utils_client import GraspUtils

from ll4ma_opt.solvers.line_search import GradientDescent, GaussNewtonMethod, NewtonMethodWolfeLine
from ll4ma_opt.solvers.quasi_newton import BFGSMethod
from ll4ma_opt.solvers.penalty_method import PenaltyMethod, AugmentedLagranMethod
from ll4ma_opt.solvers.optimization_solver import OptimizationSolver, SolverReturn

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
    pose_to_6dof

from prob_grasp_planner.grasp_common_library.data_proc_lib import DataProcLib
from matplotlib import pyplot, cm
import matplotlib
from cycler import cycler

linestyle = ['--', '--', '--', '--'] + ['-'] * 50
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00'] + \
         ['#ff00ff', '#00ffff', '#000000', '#ff7f00', '#cca01d', '#00ff7f', '#660033'] * 7 + ['#c29367']
matplotlib.rcParams["axes.prop_cycle"] = cycler('linestyle', linestyle) + cycler('color', colors)

OBJ_DATASET_DIR_ = "/home/mohanraj/ycb" #Set this to the local location of YCB dataeset

#OBJ_NAME_ = ["025_mug", "024_bowl", "054_softball"]
OBJ_NAMES_ = ["003_cracker_box", "003_cracker_box", "025_mug", "006_mustard_bottle", "054_softball"]

SESSIONS_DIR = "Sessions/"

def get_object_name(obj_str):
    Obj_Names = [x for x in os.listdir(OBJ_DATASET_DIR_) if \
                 os.path.isdir(os.path.join(OBJ_DATASET_DIR_, x))]
    Matches = [x for x in Obj_Names if obj_str in x]
    if len(Matches) > 0:
        return Matches[0]
    else:
        return np.random.choice(OBJ_NAMES_)
    

def create_sdm_env_client(env_name, dims, resolution = 1000, world_trans = Pose()):
    '''
    Client to gpu_sdm_create_env service

    TODO:
    - move to dedicated library
    '''
    rospy.wait_for_service('/gpu_sdm_create_env')
    InitEnvVox = rospy.ServiceProxy('/gpu_sdm_create_env', CreateEnv)

    InitEnvReq = CreateEnvRequest()
    InitEnvReq.env_name = env_name
    InitEnvReq.dimension = dims
    InitEnvReq.resolution = resolution
    InitEnvReq.world_trans = world_trans
    if not InitEnvVox(InitEnvReq).success:
        rospy.logfatal("Couldn't create environment, Exitting")
        rospy.signal_shutdown("No Environment Buffer to Work on")

def add_sdm_object_request(env_name, OBJ_NAME_, obj_points, obj_dims, place_conf):
    Add2EnvReq = AddObj2EnvRequest()
    Add2EnvReq.env_name = env_name
    Add2EnvReq.obj_id = OBJ_NAME_
    Add2EnvReq.sparse_obj_grid = obj_points
    Add2EnvReq.obj_size = obj_dims
    Add2EnvReq.place_conf = place_conf
    return Add2EnvReq
        
def add_sdm_object_client(env_name, OBJ_NAME_, obj_points, obj_dims, place_conf):
    '''
    Client to add objects to sdm env

    TODO:
    - move to dedicated library
    '''
    rospy.wait_for_service('/gpu_sdm_add_obj')
    Add2EnvVox = rospy.ServiceProxy('/gpu_sdm_add_obj', AddObj2Env)

    Add2EnvReq = AddObj2EnvRequest()
    Add2EnvReq.env_name = env_name
    Add2EnvReq.obj_id = OBJ_NAME_
    Add2EnvReq.sparse_obj_grid = obj_points
    Add2EnvReq.obj_size = obj_dims
    Add2EnvReq.place_conf = place_conf
    if(Add2EnvVox(Add2EnvReq)):
        rospy.loginfo("Object Added Successfully")

def numpy2points(numpy_arr):
    points_arr = []
    for a_ in numpy_arr:
        points_arr.append(Point(a_[0], a_[1], a_[2]))
    return points_arr

def get_voxels_from_segmentation(segres):
    grasp_data_lib = DataProcLib()
    obj_world_pose_stamp = PoseStamped()
    obj_world_pose_stamp.header.frame_id = segres.header.frame_id
    obj_world_pose_stamp.pose = segres.pose
    grasp_data_lib.update_object_pose_client(obj_world_pose_stamp)
    sparse_grid, vox_size, vox_dim = grasp_data_lib.voxel_gen_client(segres)
    return sparse_grid, vox_dim

def position2list(point):
    return [point.x, point.y, point.z]

def pose2array(pose):
    arr = position2list(pose.position)
    qt = position2list(pose.orientation)
    qt.append(pose.orientation.w)
    return arr + list(euler_from_quaternion(qt))

def pose2cust(pose):
    qt = position2list(pose.orientation)
    qt.append(pose.orientation.w)
    r,p,y = euler_from_quaternion(qt)
    pose.orientation.x = r
    pose.orientation.y = p
    pose.orientation.z = y
    pose.orientation.w = 0.0
    return pose

def delete_model_client(obj_name):
    rospy.wait_for_service('/gazebo/delete_model')
    GzDelObj = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    GzDelObj(obj_name)

def sample_placement(obj, x=0.0, y = 0.0):
    place_pose = Pose()
    place_pose.position.x = x #+ np.random.uniform(-0.1, 0.1)
    place_pose.position.y = y #+ np.random.uniform(-0.1, 0.1)
    place_pose.position.z = obj.depth/2
    PlaceQuat = quaternion_from_euler(0,0,np.random.uniform(0,3.14))
    ObjQuat = obj.pose.orientation
    UpQuat = np.array([ObjQuat.x, ObjQuat.y, ObjQuat.z, ObjQuat.w])
    PlaceQuat = quaternion_multiply(PlaceQuat, UpQuat)
    place_pose.orientation.x = PlaceQuat[0]
    place_pose.orientation.y = PlaceQuat[1]
    place_pose.orientation.z = PlaceQuat[2]#np.random.uniform(0,3.14)
    place_pose.orientation.w = PlaceQuat[3]#PlaceQuat[3]
    return place_pose

def get_stationary_points(sdm2d, obj, pre=""):
    dx = gen_dsdf_x(sdm2d).detach().cpu().numpy()
    dy = gen_dsdf_y(sdm2d).detach().cpu().numpy()
    mag_d = np.sqrt(np.square(dx) + np.square(dy))[0][0]
    write_shaded_image(mag_d, None, None, pre+"Mag_d_Out", cm.seismic)
    sd_mask = sdm2d.detach().cpu().numpy() > 20
    #pyplot.imshow(sd_mask.astype(np.int), cmap="gray")
    #pyplot.show()
    write_shaded_image(sd_mask, None, None, pre+"sd_filter", cm.seismic)
    mag_mask = mag_d == 0
    mask = np.multiply(sd_mask, mag_mask)
    idx = np.argwhere(mask)
    if idx.shape[0] == 0:
        mag_mask = mag_d < 1
        mask = np.multiply(sd_mask, mag_mask)
        idx = np.argwhere(mask)
    #rospy.loginfo(mask)
    obj_grid = obj.flatten()
    obj_mask = obj_grid < 0.1
    #filter_mag_d = mag_d[mask]
    write_shaded_image(mask, None, None, pre+"FMag_d_Out", cm.seismic)
    for idi in idx:
        if check_coll(sd_mask, obj_mask, idi[0], idi[1]):
            return idi
    return None

def check_coll(sdm, obj, i, j):
    i_min = i - obj.shape[0] // 2
    i_max = i + obj.shape[0] // 2 + obj.shape[0] % 2
    j_min = j - obj.shape[1] // 2
    j_max = j + obj.shape[1] // 2 + obj.shape[1] % 2
    if i_min < 0 or j_min < 0:
        return False
    if i_max > sdm.shape[0] or j_max > sdm.shape[1]:
        return False
    assert sdm[i_min:i_max, j_min:j_max].shape == obj.shape , "Check indexing [{}:{} - {}:{}] with {} vs {}".\
    format(i_min, i_max, j_min, j_max, sdm[i_min:i_max, j_min:j_max].shape, obj.shape)
    mask = np.logical_or(sdm[i_min:i_max, j_min:j_max], obj)
    if np.all(mask==True):
        fig1 = pyplot.figure()
        pyplot.imshow(sdm.astype(np.int), cmap="gray")
        bbox_y = [i_min, i_max, i_max, i_min, i_min]
        bbox_x = [j_min, j_min, j_max, j_max, j_min]
        pyplot.plot(bbox_x, bbox_y)
        fig2 = pyplot.figure()
        pyplot.subplot(2,2,1)
        imms = pyplot.imshow(sdm[i_min:i_max, j_min:j_max].astype(np.int)*255, cmap='Greys', interpolation='nearest')
        pyplot.subplot(2,2,2)
        immo = pyplot.imshow(obj.astype(np.int)*255,cmap='Greys', interpolation='nearest')
        pyplot.subplot(2,2,3)
        imm = pyplot.imshow(mask.astype(np.int)*255,cmap='Greys', interpolation='nearest')
        pyplot.subplot(2,2,4)
        imf = pyplot.imshow(mask.astype(np.int)*255,cmap='Greys', interpolation='nearest')
        #pyplot.show()
        return True
    return False

def broadcast_poses(pose_dict, parent):
    for child_id in pose_dict:
        broadcast_pose(pose_dict[child_id], parent, child_id)

def broadcast_pose(pose, parent, child):
    br = tf.TransformBroadcaster()
    trans, rot = pose_to_array(pose)
    for i in range(10):
        br.sendTransform(trans, rot, rospy.Time.now(), child, parent)
        rospy.sleep(0.1)

def get_upright_config(obj):
    place_pose = Pose()
    place_pose.position.z = obj.depth/2
    ObjQuat = obj.pose.orientation
    UpQuat = np.array([ObjQuat.x, ObjQuat.y, ObjQuat.z, ObjQuat.w])
    place_pose.orientation.x = UpQuat[0]
    place_pose.orientation.y = UpQuat[1]
    place_pose.orientation.z = UpQuat[2]
    place_pose.orientation.w = UpQuat[3]
    return place_pose

def get_objgrid():
    Segmenter = SegmentationClient()
    GraspClient = GraspUtils()
    Obj = Segmenter.Call_Service(align=True)
    #grasp, preshape = GraspClient.Call_Service(Obj)
    #visualize_grasp(grasp, preshape, prefix)
    raw_grid, raw_dim = get_voxels_from_segmentation(Obj)
    obj_dim = [Obj.height, Obj.width, Obj.depth]
    place_conf = get_upright_config(Obj)#sample_placement(Obj, 0.2, 0.2)
    return ObjGrid(raw_grid, np.array(obj_dim), place_conf, grid_size = raw_dim, res = 1000)

def publish_jointstates(js_dict):
    for child_id in js_dict:
        js_pub = rospy.Publisher(child_id+'/joint_states', JointState, queue_size=1)
        for i in range(10):
            js_dict[child_id].header.stamp = rospy.Time.now()
            js_pub.publish(js_dict[child_id])
            rospy.sleep(0.1)

def visualize_grasp(hand_pose, preshape, prefix="", object_pose='object_pose'):
    def get_shell_trans(prefix=""):
        tf_b = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tf_b)
        T = tf_b.lookup_transform(prefix+'_tf/palm_shell_pose', object_pose,\
                                  rospy.Time(), rospy.Duration(5.0))
        trans = [T.transform.translation.x, T.transform.translation.y, T.transform.translation.z]
        rot = [T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z,\
               T.transform.rotation.w]
        return trans, rot

    hand_pose.header.frame_id = object_pose
    print(hand_pose)
    grasp_data_lib = DataProcLib()
    rospy.wait_for_service(prefix+'/update_palm_goal_pose')
    update_palm_goal_proxy = rospy.ServiceProxy(prefix+'/update_palm_goal_pose', UpdatePalmPose)
    update_palm_goal_request = UpdatePalmPoseRequest()
    update_palm_goal_request.palm_pose = hand_pose
    update_palm_goal_response = update_palm_goal_proxy(update_palm_goal_request)

    #trans, rot = get_shell_trans(prefix)
    #shell_array = [0.0] * 7
    #shell_array[:3] = trans
    #shell_array[3:] = rot
    #shell_pose = convert_array_to_pose(np.array(shell_array), object_pose)
    #grasp_data_lib.update_palm_pose_client(shell_pose, prefix)

    js_pub = rospy.Publisher(prefix+'/joint_states', JointState, queue_size=1)
    for i in range(100):
        preshape.header.stamp = rospy.Time.now()
        js_pub.publish(preshape)
        rospy.sleep(0.05)

def get_placement_pose(prefix, obj_pose, base='world', ee_link='palm_link'):
    tf_b = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_b)
    broadcast_pose(obj_pose, "place_table_corner", prefix+"/object_pose")
    T = tf_b.lookup_transform(prefix+'_tf/'+ee_link, base, \
                                  rospy.Time(), rospy.Duration(5.0))
    pose = Pose()
    pose.position = T.transform.translation
    pose.orientation = T.transform.rotation
    return pose

def plot_cost_joint_profile():
    seq = 629
    for l in range(7):
        break
    costs = [0.0] * seq
    tests = [0.0] * seq
    for i in range(seq):
        #pblm.initialize_grasp()
        x_curr = copy.deepcopy(x_test)
        tests[i] = x_test[l+7]*0 + (i - seq//2) * delt
        x_curr[7+l] = tests[i] + x_test[l+7]
        x_curr[7+l] = np.arctan2(np.sin(x_curr[7+l]), np.cos(x_curr[7+l]))
        costs[i] = pblm.cost(x_curr)
        #print(tests)
        #print(costs)
        #pyplot.yscale("log")
        pyplot.plot(tests, costs, label='Joint'+str(l+1))
        pyplot.legend()
        pyplot.xlabel("Joint Angles")
        pyplot.ylabel("Grasp Cost")
        pyplot.title("Solution Delta")
    #pyplot.show()

def restore_problem(pkl_f):
    with open(pkl_f, 'rb') as f:
        pblm_data = pickle.load(f)
    if isinstance(pblm_data[0], dict) and isinstance(pblm_data[1], ObjGrid):
        return pblm_data
    else:
        rospy.logwarn("Loaded problem is invalid, starting new session")
        return None

def save_problem(pblm, solret=None):
    pkl_f = SESSIONS_DIR + datetime.now().strftime("%h%d%y_%H%M.pkl")
    with open(pkl_f, 'wb') as f:
        data = (pblm.Env_List, pblm.Obj_Grid_raw, \
                pblm.obj_world_mat, pblm.x0, \
                pblm.get_pickle(), solret.get_pickle())
        pickle.dump(data, f) #Cannot pickle the entire class due to PyKDL
        #pickle.dump((), f)
    rospy.loginfo(f"Saved solved problem as {pkl_f}")

def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    #group.add_argument('--redo', action='store_false', dest='resume')
    #group.add_argument('--resume', action='store_true', dest='resume')
    parser.add_argument('object_name', type=str, default=None, nargs='?')
    parser.add_argument('session', type=str, default=None, nargs='?')
    #parser.add_argument('P_Wt', type=float, default=10.0, nargs='?')
    #parser.add_argument('G_Wt', type=float, default=1.0, nargs='?')
    args = parser.parse_args()
    return args

def query_torch_mdn(pblm, iterates):
    priors = []
    grasp_cli = GraspUtils()
    grasp_cli.init_mdn(pblm.object_sparse_grid, pblm.object_voxel_dims, pblm.object_size)
    for q in iterates:
        xg = pblm.joint_to_grasp(q)
        #print(xg, q[pblm.PRESHAPE])
        grasp_conf = np.asarray(list(xg)+list(q[pblm.PRESHAPE]), dtype=np.float64)
        priors.append(grasp_cli.get_grasp_prior(grasp_conf))
    return priors
    
        
if __name__ == '__main__':
    rospy.init_node('pnp_playback_node')
    old_Env_List = None
    session_mode = False
    args = parse_args()
    if args.session is None:
        rospy.logerror("Session not provided")
    else:
        pkl_f = args.session
        rospy.loginfo(f"Resuming session :{args.session}")
        old_Env_List, old_obj_grid, old_obj_trans, old_x0, iter_logs, solret = restore_problem(pkl_f)
        solret = SolverReturn(solret[0], solret[1], solret[2], solret[3], solret[4], solret[5], \
                              solret[6], solret[7])
        session_mode = True

    if args.object_name is None:
        rospy.loginfo("No object specified, choosing randomly from default list")
        obj_n = np.random.choice(OBJ_NAMES_)
    else:
        obj_n = get_object_name(args.object_name)

    spawn_model_client(obj_n)
    obj_grid = get_objgrid()
    pblm = PnP_Problem_CS(old_obj_grid)
    pblm.obj_world_mat = old_obj_trans
    pblm.x0 = old_x0
    obj_key = str(len(old_Env_List.keys()) - 1)
    #old_Env_List.pop(obj_key)
        
    pblm.Env_List = old_Env_List
    pblm.copy_log(iter_logs)    
    pblm.publish_env_cloud()
    
    print('Solution:', solret.solution,'\nCost:', solret.cost)
    print('Number of steps:', len(solret.iterates)-1)
    print('Solve Time:', solret.time)
    grasp_cost, place_cost = pblm.cost(solret.solution, split=True)
    print('Grasp cost:', grasp_cost, '\tPlace cost:', place_cost)

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

    #pblm.evaluate_constraints(solret.solution)

    solret.visualize_lambdas()
    solret.visualize_penalties()

    labels = pblm.get_constraint_labels()
    fitnesses = np.array([]).reshape(len(labels), 0).T
    #for i in range(len(solret.iterates)):
    #    fitnesses = np.vstack([fitnesses, pblm.evaluate_constraints(solret.iterates[i])])

    iter_range = range(len(solret.iterates))

    pyplot.plot(iter_range, query_torch_mdn(pblm, solret.iterates), label="torch")
    for key in pblm.LogDict.keys():
        if key == 'prior':
            pyplot.plot(iter_range, pblm.extract_iters(key, solret.iterates), label=key)
    pyplot.legend()
    pyplot.show()
    
    for key in pblm.LogDict.keys():
        if key in pblm.COST_TERMS:
            pyplot.plot(iter_range, pblm.extract_iters(key, solret.iterates), label=key)
    pyplot.legend()
    pyplot.show()

    for key in pblm.LogDict.keys():
        if key not in pblm.COST_TERMS and not key.endswith('grad'):
            print(key)
            pyplot.plot(iter_range, pblm.extract_iters(key, solret.iterates), label=key)
    pyplot.legend()
    pyplot.show()

    for key in pblm.LogDict.keys():
        if key not in pblm.COST_TERMS and key.endswith('grad'):
            pyplot.plot(iter_range, pblm.extract_iters(key, solret.iterates), label=key)
    pyplot.legend()
    pyplot.show()
    #for i in range(fitnesses.T.shape[0]):
    #    pyplot.plot(iter_range, fitnesses.T[i], label=labels[i])
    #pyplot.pause(0.1)

        #print(pblm.raw_prior)
    raw_pose = convert_array_to_pose(pblm.raw_prior, "object_pose")

    grasp_pose = pblm.xt
    grasp_pose = convert_array_to_pose(grasp_pose, "world")
        #broadcast_pose(grasp_pose.pose, "world", "grasp_pose")

    reach_pose = pblm.arm_palm_KDL.fk(solret.solution[pblm.G_IDX]*0.0)
    reach_pose = convert_array_to_pose(reach_pose, "world")
    broadcast_pose(reach_pose.pose, "world", "reach_pose")

    drs = get_display_state(solret.solution[pblm.G_IDX], pblm.arm_palm_KDL)

    drs_place = get_display_state(solret.solution[pblm.P_IDX], pblm.arm_palm_KDL)
    
    print(f"Iterates shape: {solret.iterates.shape}")
    dtraj = get_display_trajectory(solret.iterates[:, pblm.G_IDX, 0], pblm.arm_palm_KDL)
    dtraj_place = get_display_trajectory(solret.iterates[:, pblm.P_IDX, 0], pblm.arm_palm_KDL)

    for _ in range(100):
        display_robot_state(drs)
        display_robot_state(drs_place, "place_robot_state")
        display_robot_trajectory(dtraj, "grasp_joint_iterates")
        display_robot_trajectory(dtraj_place, "place_joint_iterates")
        rospy.sleep(0.01)

    print(f"Place config: {pblm.object_pose_from_joints(solret.solution)}")
    place_conf = convert_array_to_pose(pblm.object_pose_from_joints(solret.solution), "place_table_corner")
    #grasp_pose = convert_array_to_pose(solret.solution[pblm.Gx_IDX], "object_pose")
    obj_poses = pblm.get_env_poses()
    obj_grid.set_pose(pblm.object_pose_from_joints(solret.solution))
    obj_poses[obj_n + pblm.new_id()] = place_conf.pose

    #tg = pblm.obj_env_collision_test()
    #write_shaded_image(tg.min(2))
    #write_shaded_image(tg.min(1))
    #write_shaded_image(tg.min(0))
    

    #pyplot.legend()
    #pyplot.show(block=True)

    pblm.plot_joint_iterates(solret.iterates)
    
    while not rospy.is_shutdown():
        broadcast_poses(obj_poses, "place_table_corner")
        #broadcast_pose(grasp_pose.pose, "object_pose", "grasp_pose")

    #uip = input("Execute grasp?")
        
    #if uip.lower() not in ['y', 'yes']:
        #delete_model_client(obj_n)
        ##continue

    rospy.loginfo("Aborting...")
    rospy.signal_shutdown("User aborted execution")
    exit(0)

    with True:
        planner.execute_plan(plan_traj)
        hand.grasp()
        rospy.sleep(2)

        lift_traj = planner.lift_up()
        planner.execute_plan(lift_traj)
        #exit(0)

        plan_traj = planner.plan_to_joint(solret.solution[pblm.P_IDX])
        planner.execute_plan(plan_traj)
        hand.release()

        lift_traj = planner.lift_up()
        planner.execute_plan(lift_traj)

        i = 0
        while not rospy.is_shutdown():
            #broadcast_pose(grasp_pose.pose, "world", "grasp_pose")
            broadcast_pose(reach_pose.pose, "world", "reach_pose")
            broadcast_pose(raw_pose.pose, "object_pose", "raw_pose")

            #drs = get_display_state(solret.iterates[i][pblm.G_IDX], pblm.arm_palm_KDL)
            display_robot_state(drs)
            display_robot_state(drs_place, "place_robot_state")
            i += 1
            i = i % len(solret.iterates)
            rospy.sleep(0.01)
            
        delete_model_client(obj_n)
        home_traj = planner.go_home()
        planner.execute_plan(home_traj)
        
        exit(0)
        
        start = time.time()
        solret = solver1.optimize(pblm.x0, max_iterations=1000)
        print('Solution:', solret.solution,'\nCost:', solret.cost)
        print('Number of steps:', len(solret.iterates)-1)
        print('Solve Time:', solret.time)

        exit(0)
        start = time.time()
        x, y, converged, x_iterates = solver2.optimize(pblm.x0, max_iterations=100)
        print('Solution:',x,'\nCost:',y)
        print('Number of steps:', len(x_iterates)-1)
        print('Solve Time:', time.time()-start)

        #x, y, converged, x_iterates = solver3.optimize(x_test,max_iterations=50)
        #print('Solution:',x,'\nCost:',y)
        #print('Number of steps:', len(x_iterates)-1)
        exit(0)
        hand.set_preshape(preshape)
        grasp_preshapes[pref] = preshape
        plan_pose = TransformPose(grasp.pose, 'world', "object_pose")
        #plan_traj = planner.plan_to_pose(plan_pose)
        #planner.execute_plan(plan_traj)
        #rospy.loginfo("Execution Called")
        #hand.grasp()
        #lift_traj = planner.lift_up()
        #planner.execute_plan(lift_traj)
        #rospy.loginfo("Execution Called")
        #exit(0)
        #offset = obj_grid.recenter()
        #PartialEnv.add_object_handle(add_sdm_object_request("p", obj_n, numpy2points(raw_grid), raw_dim, place_conf))
        sdm3d, l, h = PartialEnv.gen_sd_map()
        sdm2d = min_from_sdm3d(sdm3d)
        write_shaded_image(sdm2d, l, h, out=str(i)+'min2d')
        idi = get_stationary_points(sdm2d, obj_grid, str(i))
        #idi = [10, 10]
        if idi is not None:
            rospy.loginfo("position found")
            obj_grid.pose.position.y = idi[1]/1000
            obj_grid.pose.position.x = idi[0]/1000
            PartialEnv.ObjList[obj_n+str(i)] = obj_grid
            PartialEnv.expand_grid()
            PartialEnv.publish_as_point_cloud("partial_env")
            obj_poses[pref+"/object_pose"] = obj_grid.pose
            broadcast_pose(obj_grid.pose, "place_table_corner", pref+"/object_pose")
            plan_pose = TransformPose(grasp.pose, 'world', "object_pose")
            plan_traj = planner.plan_to_pose(plan_pose)
            visualize_grasp(grasp, preshape, pref, pref+"/object_pose")
            planner.execute_plan(plan_traj)
            rospy.loginfo("Execution Called")
            hand.grasp()
            lift_traj = planner.lift_up()
            planner.execute_plan(lift_traj)
            rospy.loginfo("Execution Called")
            broadcast_pose(obj_grid.pose, "place_table_corner", pref+"/object_pose")
            place_pose = get_placement_pose(pref, obj_grid.pose)
            place_traj = planner.plan_to_pose(place_pose)
        else:
            rospy.logwarn("no solution")
        #delete_model_client(obj_n)
        rospy.sleep(2)

    while not rospy.is_shutdown():
        broadcast_poses(obj_poses, "place_table_corner")
        #publish_jointstates(grasp_preshapes)
        
    for i in range(sdm3d.shape[0]):
        if (i % 5) == 0:
            write_shaded_image(sdm3d[i], l, h, out='3ds'+str(i))
