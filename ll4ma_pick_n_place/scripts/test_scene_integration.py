#!/usr/bin/env python3

import rospy
import sys
import trimesh
import numpy as np
from segmentation_client import SegmentationClient
from gazebo_msgs.srv import SpawnModel, DeleteModel
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from geometry_msgs.msg import Pose, WrenchStamped, Quaternion, PoseStamped, Point
import std_msgs.msg

from pointsdf_reconstruction.srv import Reconstruct, ReconstructRequest

from gpu_sd_map.srv import CreateEnv, CreateEnvRequest, AddObj2Env, AddObj2EnvRequest
from gpu_sd_map.environment_manager import EnvVox

from prob_grasp_planner.grasp_common_library.data_proc_lib import DataProcLib

OBJ_DATASET_FOLDER_ = "/home/mohanraj/ycb" #Set this to the local location of YCB dataset

OBJ_NAME_ = ["025_mug", "024_bowl", "054_softball"]
#OBJ_NAME_ = ["003_cracker_box" ,"025_mug" ,"019_pitcher_base", "006_mustard_bottle", "024_bowl", "054_softball"]

def GenURDFString(obj_name):
    object_pose = [0.0]*6;
    obj_mesh_path = OBJ_DATASET_FOLDER_ + '/' + obj_name + \
                           '/google_16k' + '/nontextured.stl'
    collision_mesh = OBJ_DATASET_FOLDER_ + '/' + obj_name + \
                           '/google_16k' + '/nontextured_proc.stl'

    obj_mesh = trimesh.load(obj_mesh_path)
    obj_mesh.density = 10
    object_mass, object_inertia = obj_mesh.mass, obj_mesh.moment_inertia
    object_rpy = str(object_pose[0]) + ' ' + str(object_pose[1]) + ' ' + str(object_pose[2])
    object_location = str(object_pose[3]) + ' ' + str(object_pose[4]) + ' ' + str(object_pose[5])
    r = np.random.randint(0,255)
    g = np.random.randint(0,255)
    b = np.random.randint(0,255)
    urdf_str = """
    <robot name=\"""" + obj_name + """\">
    <link name=\"""" + obj_name + """_link">
    <inertial>
    <origin xyz=\"""" + str(object_location) +"""\"  rpy=\"""" + str(object_rpy) +"""\"/>
    <mass value=\"""" + str(object_mass) + """\" />
    <inertia  ixx=\"""" + str(object_inertia[0][0]) + """\" ixy=\"""" + str(object_inertia[0][1]) + """\"  ixz=\"""" + \
        str(object_inertia[0][2]) + """\"  iyy=\"""" + str(object_inertia[1][1]) + """\"  iyz=\"""" + str(object_inertia[1][2]) + \
        """\"  izz=\"""" + str(object_inertia[2][2]) + """\" />
    </inertial>
    <visual>
    <origin xyz=\"""" + str(object_location) +"""\"  rpy=\"""" + str(object_rpy) +"""\"/>
    <geometry>
    <mesh filename=\"file://""" + obj_mesh_path + """\" />
    </geometry>
    <material name="rand_color">
    <color rbg=\""""+str(r)+" "+str(g)+" "+str(b)+"""\" />
    </material>
    </visual>
    <collision>
    <origin xyz=\"""" + str(object_location) +"""\"  rpy=\"""" + str(object_rpy) +"""\"/>
    <geometry>
    <mesh filename=\"file://""" + collision_mesh + """\" />
    </geometry>
    </collision>
    </link>
    <gazebo reference=\"""" + obj_name + """_link\">
    <mu1>10.0</mu1>
    <maxVel>0.0</maxVel>
    <minDepth>0.003</minDepth>
    <material>Gazebo/Red</material>
    </gazebo>
    </robot>
    """
    return urdf_str, obj_mesh.bounds

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

def spawn_client(obj_name):
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    GzSpawnObj= rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
    pick_table_height = 0.762
    Obj_urdf, obj_bounds = GenURDFString(obj_name)
    SpawnPose = Pose()
    SpawnPose.position.x = -0.2
    SpawnPose.position.y = -0.0
    SpawnPose.position.z = 1#pick_table_height/2.0 - obj_bounds[0][2] + 0.01
    SpawnQuat = quaternion_from_euler(0,0,np.random.uniform(0,3.14))
    SpawnPose.orientation.x=SpawnQuat[0]
    SpawnPose.orientation.y=SpawnQuat[1]
    SpawnPose.orientation.z=SpawnQuat[2]
    SpawnPose.orientation.w=SpawnQuat[3]
    GzSpawnObj(obj_name, Obj_urdf, 'Mutable_Objs', SpawnPose, 'ptable_link')
    rospy.sleep(2)

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

if __name__ == '__main__':
    rospy.init_node('pnp_scene_test_node')

    TABLE_DIM_ = [0.9125, 0.61, 0.5]

    dist = 0.2

    place = 3
    X = [0.1, 0.3, 0.5]
    Y = [0.1, 0.3, 0.5, 0.7]

    PartialEnv = EnvVox(TABLE_DIM_[0], TABLE_DIM_[1], TABLE_DIM_[2], 1000)
    
    env1 = "reconstructed"
    env2 = "partial"
    create_sdm_env_client(env1, TABLE_DIM_)
    create_sdm_env_client(env2, TABLE_DIM_)
    
    for i in range(place):
        obj_n = OBJ_NAME_[i%len(OBJ_NAME_)]
        #raw_input("Delete prev obj and press Enter to continue...")
    
        spawn_client(obj_n)
        #break
        #delete_model_client(obj_n)
        #continue
        Segmenter = SegmentationClient()
        Obj = Segmenter.Call_Service(align=True)
        #print(Obj.header.frame_id)
        rospy.wait_for_service('ll4ma_3d_reconstruct')
        Reconstruct_Srv = rospy.ServiceProxy('ll4ma_3d_reconstruct', Reconstruct)
        ReconReq = ReconstructRequest()
        ReconReq.grasp_obj = Obj
        rospy.loginfo("Calling Pointsdf_Reconstruction")
        ReconResp = Reconstruct_Srv(ReconReq)
        rospy.loginfo("Reconstuction successful")

        raw_grid, raw_dim = get_voxels_from_segmentation(Obj)
        #print(raw_grid.shape)
        #print(np.min(raw_grid, axis=0))
        #print(np.max(raw_grid, axis=0))
        raw_dim = [Obj.height, Obj.width, Obj.depth]
        print(raw_dim)
        #place_conf = [pose2cust(Obj.pose)]
        x_i = dist * ((i // 3) + 0.2)
        y_i = dist * ((i % 3) + 0.5)
        rospy.loginfo("Placing at {} , {}".format(x_i, y_i))
        place_conf = sample_placement(Obj, x_i, y_i)
        #invert = [False, False]
        #print(place_conf)
        add_sdm_object_client(env1, obj_n+str(i), ReconResp.sparse_points, ReconResp.true_dims, place_conf)
        add_sdm_object_client(env2, obj_n+str(i), numpy2points(raw_grid), raw_dim, place_conf)
        #sdm = env2.
        delete_model_client(obj_n)
        rospy.sleep(2)
