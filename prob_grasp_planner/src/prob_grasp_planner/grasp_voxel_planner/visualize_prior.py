#!/usr/bin/env python
import roslib
import rospy
from grasp_voxel_inference import GraspVoxelInference
import roslib.packages as rp
import h5py
import sys
import numpy as np
import tf
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import JointState
from data_proc_lib import DataProcLib
import grasp_common_functions as gcf
from prob_grasp_planner.srv import *
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
sys.path.append(pkg_path + '/src/grasp_common_library')
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped

def TransformPose(inpose,target_frame='palm_link',source_frame='shell'):
    tf_b = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_b)
    inpose_stamped = tf2_geometry_msgs.PoseStamped()
    inpose_stamped.pose = inpose
    inpose_stamped.header.frame_id = source_frame
    inpose_stamped.header.stamp = rospy.Time(0)

    opose = tf_b.transform(inpose_stamped,target_frame,rospy.Duration(5))
    return opose.pose

def broadcast_goals(tf_br, palm_goal_pose):
    palm_pose = palm_goal_pose
    tf_br.sendTransform((palm_pose.pose.position.x, palm_pose.pose.position.y, 
                    palm_pose.pose.position.z),
                    (palm_pose.pose.orientation.x, palm_pose.pose.orientation.y, 
                    palm_pose.pose.orientation.z, palm_pose.pose.orientation.w),
                             rospy.Time.now(), 'palm_goal_pose', 'object_pose')
    tf_br.sendTransform((-0.063, 0.000, -0.020),
                             (0.707, -0.0, 0.707, -0.0),
                             rospy.Time.now(), 'palm_shell_pose', 'palm_goal_pose')

def get_shell_trans():
    tf_b = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_b)
    T = tf_b.lookup_transform('palm_shell_pose', 'object_pose', rospy.Time(), rospy.Duration(5.0))
    trans = [T.transform.translation.x, T.transform.translation.y, T.transform.translation.z]
    rot = [T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w]
    print(trans, rot)
    return trans, rot
    

def pub_preshape_config(preshape, js_pub):
    js = JointState()
    js.name = ['preshape_1', 'proximal_joint_1', 'preshape_2', 'proximal_joint_2', 'proximal_joint_3']
    js.position = [0.0] * 5
    js.velocity = [0.0] * 5
    js.effort = [0.0] * 5
    #print('Setting Preshape: ', preshape[6])
    js.position[0] = preshape[6]
    js.position[2] = preshape[6]
    js.header.stamp = rospy.Time.now()
    js_pub.publish(js)
    rospy.sleep(0.5)

def convert_array_to_pose(pose_array, frame_id):
    '''
    Convert pose Quaternion array to ROS PoseStamped.

    Args:
        pose_array
    
    Returns:
        ROS pose.
    '''
    pose_stamp = PoseStamped()
    pose_stamp.header.frame_id = frame_id
    pose_stamp.pose.position.x, pose_stamp.pose.position.y, \
        pose_stamp.pose.position.z = pose_array[:3]    

    if len(pose_array) == 6:
        palm_quaternion = tf.transformations.quaternion_from_euler(
            pose_array[3], pose_array[4], pose_array[5])
    elif len(pose_array) == 7:
        palm_quaternion = pose_array[3:]
    
    pose_stamp.pose.orientation.x, pose_stamp.pose.orientation.y, \
        pose_stamp.pose.orientation.z, pose_stamp.pose.orientation.w = \
        palm_quaternion
    
    return pose_stamp
    

def get_grasp_data(raw_data_path, obj_id, grasp_id_cur_obj):
    grasp_file_path = raw_data_path + 'grasp_data.h5'
    grasp_data_file = h5py.File(grasp_file_path, 'r')
    grasps_number_key = 'grasps_number'
    grasp_no_obj_num_key = 'grasp_no_obj_num'
    grasps_num = grasp_data_file['total_grasps_num'][()]
    grasps_no_seg_num = proc_grasp_file[grasp_no_obj_num_key][()]
    max_object_id = grasp_data_file['max_object_id'][()]
    print grasps_num, max_object_id
    data_proc_lib = DataProcLib()
    object_id = 'object_' + str(obj_id)
    object_name = grasp_data_file[object_id + '_name'][()]
    object_grasp_id = object_id + '_grasp_' + str(grasp_id_cur_obj)
    object_world_seg_pose_key = object_grasp_id + '_object_world_seg_pose'
    object_world_seg_pose_array = grasp_data_file[object_world_seg_pose_key][()]
    pcd_file_name = find_pcd_file(data_path, grasp_data_file,
                                                    object_id, grasp_id_cur_obj, object_name)
    pcd_file_path = data_path + 'visual/pcd/' + pcd_file_name
    print pcd_file_path
    seg_obj_resp = gcf.seg_obj_from_file_client(pcd_file_path, data_proc_lib.listener)

if __name__ == '__main__':
    rospy.init_node('viz_prior_node')
    data_proc_lib = DataProcLib()
    #cvtf = gcf.ConfigConvertFunctions()
    grasp_net_model_path = pkg_path + '/models/reflex_grasp_inf_models/grasp_voxel_net/' + \
                           'grasp_voxel_net_reflex.ckpt'
    prior_model_path = pkg_path + '/models/reflex_grasp_inf_models/grasp_prior/' + \
                       'prior_net_reflex.ckpt'
    gmm_model_path = pkg_path + '/models/reflex_grasp_inf_models/grasp_prior/' + \
                     'failure_gmm_sets'
    ginf = GraspVoxelInference(grasp_net_model_path, prior_model_path, gmm_model_path)

    ginf.locs, ginf.scales, ginf.logits = [None] * 3

    pcd_dir = '/home/mohanraj/reflex_grasp_data/batch3/visual/pcd/'
    #pcd_file = 'object_6_crystal_hot_sauce_grasp_23.pcd'
    #pcd_file = 'object_116_haagen_dazs_cookie_dough_grasp_8.pcd'
    #pcd_file = 'object_47_vo5_tea_therapy_healthful_green_tea_smoothing_shampoo_grasp_4.pcd'
    pcd_file = 'object_77_canon_ack_e10_box_grasp_4.pcd'
    #pcd_file = 'object_94_quaker_chewy_chocolate_chip_grasp_6.pcd'
    
    pcd_path = pcd_dir + pcd_file

    seg_obj_resp = gcf.seg_obj_from_file_client(pcd_path, data_proc_lib.listener)

    obj_world_pose_stamp = PoseStamped()
    obj_world_pose_stamp.header.frame_id = seg_obj_resp.obj.header.frame_id
    obj_world_pose_stamp.pose = seg_obj_resp.obj.pose
    print(obj_world_pose_stamp.header.frame_id)
    data_proc_lib.update_object_pose_client(obj_world_pose_stamp)
    #exit(1)

    if not seg_obj_resp.object_found:
        print 'No object found for segmentation!'
    else:
        print 'Object Found'
    
    test_data_path = '/home/mohanraj/reflex_grasp_data/processed/batch3/' + \
                     'grasp_voxelized_data.h5'

    tf_br = tf.TransformBroadcaster()
    js_pub = rospy.Publisher('/joint_states', 
                                  JointState, queue_size=1)

    #Open and read from processed grasp dataset
    data_file = h5py.File(test_data_path, 'r')
    grasp_id = 100
    #grasp_config_obj_key = 'grasp_' + str(grasp_id) + '_config_obj'
    #grasp_full_config = data_file[grasp_config_obj_key][()] 
    #preshape_config_idx = list(xrange(7)) #+ [10, 11] + \
                           #[14, 15] + [18, 19]
    #grasp_preshape_config = grasp_full_config[preshape_config_idx]
    #grasp_sparse_voxel_key = 'grasp_' + str(grasp_id) + '_sparse_voxel'
    #sparse_voxel_grid = data_file[grasp_sparse_voxel_key][()]
    obj_dim_key = 'grasp_' + str(grasp_id) + '_dim_w_h_d'
    obj_size = data_file[obj_dim_key][()]
    #grasp_label_key = 'grasp_' + str(grasp_id) + '_label'
    #grasp_label = data_file[grasp_label_key][()]

    print('Requesting voxel')
    obj_dim = [seg_obj_resp.obj.width, seg_obj_resp.obj.height, seg_obj_resp.obj.depth]
    sparse_voxel_grid, voxel_size, voxel_grid_dim = data_proc_lib.voxel_gen_client(seg_obj_resp.obj)
    print('Voxel Loaded')
    print(voxel_grid_dim)
    
    #voxel_grid_full_dim = [32, 32, 32]
    voxel_grid = np.zeros(tuple(voxel_grid_dim))
    voxel_grid_index = sparse_voxel_grid.astype(int)
    voxel_grid[voxel_grid_index[:, 0], voxel_grid_index[:, 1],
                voxel_grid_index[:, 2]] = 1
    voxel_grid = np.expand_dims(voxel_grid, -1)
    #data_file.close()

    print(obj_dim, obj_size)

    ginf.get_prior_mixture(voxel_grid, obj_dim)
    samples =  ginf.sample_grasp_config()[0]
    #samples = ginf.mdn_mean_explicit(voxel_grid, obj_dim, 'side')#ginf.sample_grasp_config()

    print(samples[:7])
    
    hand_pose = convert_array_to_pose(np.array(samples[:6]), 'object_pose')

    rospy.wait_for_service('update_palm_goal_pose')
    update_palm_goal_proxy = rospy.ServiceProxy('update_palm_goal_pose', UpdatePalmPose)
    update_palm_goal_request = UpdatePalmPoseRequest()
    update_palm_goal_request.palm_pose = hand_pose
    update_palm_goal_response = update_palm_goal_proxy(update_palm_goal_request)

    if(update_palm_goal_response.success):
        rospy.loginfo('Goals Updated')
    
    #broadcast_goals(tf_br, hand_pose)
    trans, rot = get_shell_trans()
    shell_array = [0.0] * 7
    shell_array[:3] = trans
    shell_array[3:] = rot
    shell_pose = convert_array_to_pose(np.array(shell_array), 'object_pose')
    
    data_proc_lib.update_palm_pose_client(shell_pose)

    rate = rospy.Rate(1000)

    while not rospy.is_shutdown():
        pub_preshape_config(samples[:7], js_pub)
        rate.sleep()
