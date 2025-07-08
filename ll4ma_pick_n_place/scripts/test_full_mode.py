#!/usr/bin/env python3

import rospy
import sys
import trimesh
import numpy as np
import tf
import tf2_ros
from ll4ma_pick_n_place.gazebo_utils import *
from ll4ma_pick_n_place.planning_client import moveit_client, reflex_client
from ll4ma_pick_n_place.segmentation_client import SegmentationClient
from ll4ma_pick_n_place.grasp_utils_client import GraspUtils
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
from gpu_sd_map.ros_transforms_lib import convert_array_to_pose, pose_to_array, TransformPose

from prob_grasp_planner.grasp_common_library.data_proc_lib import DataProcLib
from matplotlib import pyplot, cm

OBJ_DATASET_FOLDER_ = "/home/mohanraj/ycb" #Set this to the local location of YCB dataeset

#OBJ_NAME_ = ["025_mug", "024_bowl", "054_softball"]
OBJ_NAME_ = ["003_cracker_box" ,"019_pitcher_base", "006_mustard_bottle", "054_softball"]



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

def get_objgrid(prefix):
    Segmenter = SegmentationClient()
    GraspClient = GraspUtils()
    Obj = Segmenter.Call_Service(align=True)
    grasp, preshape = GraspClient.Call_Service(Obj)
    visualize_grasp(grasp, preshape, prefix)
    raw_grid, raw_dim = get_voxels_from_segmentation(Obj)
    raw_dim = [Obj.height, Obj.width, Obj.depth]
    place_conf = get_upright_config(Obj)#sample_placement(Obj, 0.2, 0.2)
    return ObjGrid(raw_grid, np.array(raw_dim), place_conf, res = 1000), grasp, preshape

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
        
if __name__ == '__main__':
    rospy.init_node('pnp_scene_test_node')

    hand = reflex_client()
    
    TABLE_DIM_ = [0.31, 0.5125, 0.5]

    dist = 0.2

    place = 1
    X = [0.1, 0.3, 0.5]
    Y = [0.1, 0.3, 0.5, 0.7]

    obj_poses = {}
    grasp_preshapes = {}

    planner = moveit_client()
    PartialEnv = EnvVox(TABLE_DIM_[0], TABLE_DIM_[1], TABLE_DIM_[2], 1000)

    for i in range(place):

        pref="ref"+str(i)
        obj_n = OBJ_NAME_[i%len(OBJ_NAME_)]

        hand.release()
        home_traj = planner.go_home()
        planner.execute_plan(home_traj)
        #obj_n = OBJ_NAME_[0]
        spawn_model_client(obj_n)
        #Segmenter = SegmentationClient()
        #Obj = Segmenter.Call_Service(align=True)
        #raw_grid, raw_dim = get_voxels_from_segmentation(Obj)
        #raw_dim = [Obj.height, Obj.width, Obj.depth]
        #place_conf = sample_placement(Obj, 0.2, 0.2)
        obj_grid, grasp, preshape = get_objgrid(pref)
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
        publish_jointstates(grasp_preshapes)
        
    for i in range(sdm3d.shape[0]):
        if (i % 5) == 0:
            write_shaded_image(sdm3d[i], l, h, out='3ds'+str(i))
