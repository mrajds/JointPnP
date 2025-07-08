#!/usr/bin/env python3
'''
Stuff related to segmentation goes here.
'''

import numpy as np

import rospy
from point_cloud_segmentation.srv import *
from point_cloud_segmentation.align_object_frame import align_object

from prob_grasp_planner.grasp_common_library.data_proc_lib import DataProcLib

from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import PointCloud2

from gpu_sd_map.environment_manager import ObjGrid
from gpu_sd_map.ros_transforms_lib import get_tf_mat, convert_array_to_pose, mat_to_vec, \
    rewire_pose, normalize_pose

from ll4ma_pick_n_place.visualizations import merge_pointcloud2s, publish_pointcloud2, broadcast_tf

class SegmentationClient:
    '''
    Simplified interface to call the point_cloud_segmentation service.
    This call handles all message processing needing moslty a single line call for usage.
    This is also a place to add and adapt other segmentation libraries.
    '''
    def __init__(self, init_node = False):
        if init_node:
            rospy.init_node('pnp_segementation_client_node')
        rospy.wait_for_service('/object_segmenter')
        self.Service = rospy.ServiceProxy('/object_segmenter', SegmentGraspObject)
        self.Multi_Seg_Service = rospy.ServiceProxy('/multi_object_segmenter', MultiGraspObjects)

    def Call_Service(self, align=True):
        '''
        Calls the object_segmenter service
        
        Params:
        None
        
        Output:
        SegmentedObject - Object containing segmented cloud, pose and dims. None if call fails. (point_cloud_segementation/GraspObject.msg)

        Status: Implemented

        Testing:
        '''
        try:
            seg_req_ = SegmentGraspObjectRequest()
            seg_res_ = self.Service(seg_req_)
            rospy.loginfo("Object raw_dim")
            rospy.loginfo([seg_res_.obj.width, seg_res_.obj.height, seg_res_.obj.depth])
        except:
            rospy.logerr('Segmentation Failed')
            return None
        if seg_res_.object_found:
            rospy.loginfo('Segmentation Successful')
            if align:
                seg_res_.obj = align_object(seg_res_.obj, world_z = False)
                rospy.loginfo("Object processed_dim")
                rospy.loginfo([seg_res_.obj.width, seg_res_.obj.height, seg_res_.obj.depth])
            return seg_res_.obj
        else:
            rospy.logerr('Segmentation Failed')
            return None

    def Call_Multi_Service(self, align=True):
        rospy.loginfo('Multi seg service')
        #pub = rospy.Publisher('/object_cloud', PointCloud2, queue_size=1)
        pub_pose = rospy.Publisher('/object_pose', PoseStamped, queue_size=1)
        try:
            seq_req_ = MultiGraspObjectsRequest()
            seg_res_ = self.Multi_Seg_Service(seq_req_)
        except:
            rospy.logerr('Segmentation Failed')
            return None
        if seg_res_.objects_found:
            rospy.loginfo('Segmentation Successful')
            if align:
                seg_res_.objs = [align_object(seg_obj) for seg_obj in seg_res_.objs]
            while True:
                ctr = 0
                for seg_obj in seg_res_.objs:
                    seg_obj.pose = normalize_pose(seg_obj.pose)
                    other_objs = [other for other in seg_res_.objs if other != seg_obj]
                    #rospy.loginfo("Object raw_dim")
                    #rospy.loginfo([seg_obj.width, seg_obj.height, seg_obj.depth])
                    #seg_obj = align_object(seg_obj)
                    #rospy.loginfo("Object processed_dim")
                    rospy.loginfo([seg_obj.width, seg_obj.height, seg_obj.depth])
                    merged = merge_pointcloud2s([obj.cloud for obj in other_objs])
                    publish_pointcloud2(merged, '/clutter_cloud')
                    publish_pointcloud2(seg_obj.cloud, '/object_cloud')
                    t_pose = PoseStamped()
                    t_pose.header = seg_obj.header
                    t_pose.pose = seg_obj.pose
                    pub_pose.publish(t_pose)
                    #broadcast_tf(str(ctr), t_pose)
                    ctr += 1
                    uip=input('Set as grasp object?')
                    if uip.lower() == 'y':
                        return seg_obj, other_objs

def rewire_seg_obj_poses(objs, parent_pose):
    poses = []
    for obj in objs:
        poses.append(rewire_pose(obj.pose, parent_pose))
    return poses

def get_upright_config(obj):
    place_pose = Pose()
    place_pose.position.z = obj.depth/2
    ObjQuat = obj.pose.orientation
    UpQuat = np.array([ObjQuat.x, ObjQuat.y, ObjQuat.z, ObjQuat.w])
    rospy.loginfo('UpQuat:' + str(UpQuat))
    rospy.loginfo(obj.pose)
    place_pose.orientation.x = UpQuat[0]
    place_pose.orientation.y = UpQuat[1]
    place_pose.orientation.z = UpQuat[2]
    place_pose.orientation.w = UpQuat[3]
    mat = get_tf_mat("world", "object_pose")
    arr = mat_to_vec(mat)
    return convert_array_to_pose(arr, 'world').pose


def get_voxels_from_segmentation(segres):
    grasp_data_lib = DataProcLib()
    obj_world_pose_stamp = PoseStamped()
    obj_world_pose_stamp.header.frame_id = segres.header.frame_id
    obj_world_pose_stamp.pose = segres.pose
    grasp_data_lib.update_object_pose_client(obj_world_pose_stamp)
    sparse_grid, vox_size, vox_dim = grasp_data_lib.voxel_gen_client(segres)
    return sparse_grid, vox_dim

def update_object_world_pose(pose):
    grasp_data_lib = DataProcLib()
    obj_world_pose_stamp = PoseStamped()
    obj_world_pose_stamp.header.frame_id = 'world'
    obj_world_pose_stamp.pose = pose
    grasp_data_lib.update_object_pose_client(obj_world_pose_stamp)

def get_objgrid(feedback=False):
    Segmenter = SegmentationClient()
    #GraspClient = GraspUtils()
    Obj = Segmenter.Call_Service(align=True)
    uip = lambda: input("Proceed with current segmentation?").lower()
    while feedback and uip() not in ['y', 'yes']:
        Obj = Segmenter.Call_Service(align=True)
    return seg_2_objgrid(Obj)

def seg_2_objgrid(seg_obj):
    raw_grid, raw_dim = get_voxels_from_segmentation(seg_obj)
    obj_dim = [seg_obj.width, seg_obj.height, seg_obj.depth]
    place_conf = seg_obj.pose#get_upright_config(seg_obj)
    return ObjGrid(raw_grid, np.array(obj_dim), place_conf, grid_size = raw_dim, res = 1000)

def get_multi_objgrid(feedback=False):
    Segmenter = SegmentationClient()
    Obj = Segmenter.Call_Service(align=True)
    uip = lambda: input("Proceed with current segmentation?").lower()
    while feedback and uip() not in ['y', 'yes']:
        res = Segmenter.Call_Multi_Service(align=True)
        if res is None:
            continue
        Obj, Other = res
    Other_poses = rewire_seg_obj_poses(Other, Obj.pose)
    Other = [seg_2_objgrid(obj) for obj in Other]
    return seg_2_objgrid(Obj), Other, Other_poses 
        
    #grasp, preshape = GraspClient.Call_Service(Obj)
    #visualize_grasp(grasp, preshape, prefix)
    
    #obj_dim = [Obj.height, Obj.width, Obj.depth]
    
