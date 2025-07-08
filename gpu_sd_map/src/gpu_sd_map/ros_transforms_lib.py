#!/usr/bin/env python

import rospy
import numpy as np
import tf
import tf2_ros
import tf2_geometry_msgs
import std_msgs.msg
import PyKDL

#Ros functions
from tf.transformations import quaternion_from_euler, euler_from_quaternion, euler_matrix, \
    is_same_transform, euler_from_matrix, quaternion_inverse, quaternion_multiply

#ROS messages:
from geometry_msgs.msg import Point, Pose, PoseStamped, Point32
from sensor_msgs.msg import PointCloud, ChannelFloat32, JointState

from .transforms_lib import vector2mat, vector2matinv

#tf_b = tf2_ros.Buffer()
#tf_listener = tf2_ros.TransformListener(tf_b)

def euler_diff(eulerA, eulerB):
        qA = quaternion_from_euler(eulerA[0], eulerA[1], eulerA[2])
        qB = quaternion_from_euler(eulerB[0], eulerB[1], eulerB[2])
        qAinv = quaternion_inverse(qA)
        qdiff = quaternion_multiply(qAinv, qB)
        return euler_from_quaternion(qdiff)

def publish_as_point_cloud(points, topic = 'env_points', colors=None, init_node = False, frame_id = "world" ):
        '''
        Publish the sparse grid as a point_cloud over ROS topic.

        Params:
        None

        Output:
        None - Publish ros topic over
        '''

        if init_node:
            rospy.init_node('env_grid_viz')
        pcl_pub_ = rospy.Publisher(topic, PointCloud, queue_size=10)
        point_cloud_ = PointCloud()
        point_cloud_.header = std_msgs.msg.Header()
        point_cloud_.header.stamp = rospy.Time.now()
        point_cloud_.header.frame_id = frame_id
        point_cloud_.channels.append(ChannelFloat32())
        point_cloud_.channels[0].name = "r"
        point_cloud_.channels.append(ChannelFloat32())
        point_cloud_.channels[1].name = "g"
        point_cloud_.channels.append(ChannelFloat32())
        point_cloud_.channels[2].name = "b"

        if colors is None:
            colors = np.tile(np.random.rand(1,3),(points.shape[0],1))
            
        for i in range(points.shape[0]):
            point_cloud_.points.append(Point32(points[i][0], points[i][1], points[i][2]))
            point_cloud_.channels[0].values.append(colors[i][0])
            point_cloud_.channels[1].values.append(colors[i][1])
            point_cloud_.channels[2].values.append(colors[i][2])

        rate = rospy.Rate(20)
        for i in range(10):
            pcl_pub_.publish(point_cloud_)
            rate.sleep()
        return

def position2list(point):
    return [point.x, point.y, point.z]

def point2numpy(points):
    '''
    Converts geometry_msgs/Point[] object to (N, 3) numpy array.

    Params:
    points - geometry_msgs/Point[] object with x, y, z vales.
    
    Output:
    points_arr - (N, 3) numpy array 

    Status: Implemented
    '''
    points_arr = np.array([],dtype=np.float64).reshape(0,3)
    if isinstance(points, list):
        for p_ in points:
            points_arr = np.vstack([points_arr, np.array([p_.x, p_.y, p_.z])])
    assert points_arr.shape[1] == 3, "Output dims incorrect check ros_transforms_lib/Point2Numpy"
    return points_arr

def numpy2points(numpy_arr):
    points_arr = []
    for a_ in numpy_arr:
        points_arr.append(Point(a_[0], a_[1], a_[2]))
    return points_arr

def pose2array(pose):
    arr = position2list(pose.position)
    qt = position2list(pose.orientation)
    #qt.append(pose.orientation.w)
    return arr + list(qt)

def TransformPoseStamped(inpose, target_frame, tf_b = None, listener = None):
    if tf_b is None:
        tf_b = tf2_ros.Buffer()
    if listener is None:
        listener = tf2_ros.TransformListener(tf_b)
    opose = tf_b.transform(inpose,target_frame,rospy.Duration(1))
    return opose.pose

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

def broadcast_pose(inpose_arr, base_frame, target_frame):
    inpose = convert_array_to_pose(inpose_arr, base_frame).pose
    br = tf.TransformBroadcaster()
    for i in range(10):
        positiontuple = (inpose.position.x, inpose.position.y, inpose.position.z)
        quattuple = (inpose.orientation.x, \
                     inpose.orientation.y, \
                     inpose.orientation.z, \
                     inpose.orientation.w)
        br.sendTransform(positiontuple,quattuple,rospy.Time.now(),target_frame,base_frame)
        rospy.sleep(0.1)

def get_shell_trans(prefix=""):
    tf_b = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_b)
    T = tf_b.lookup_transform(prefix+'_tf/palm_shell_pose', 'object_pose', rospy.Time(), rospy.Duration(5.0))
    trans = [T.transform.translation.x, T.transform.translation.y, T.transform.translation.z]
    rot = [T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w]
    print(trans, rot)
    return trans, rot

def get_tf_mat(source_frame, target_frame, tf_b = None, listener = None):
    if tf_b is None:
        tf_b = tf2_ros.Buffer()
    if listener is None:
        listener = tf2_ros.TransformListener(tf_b)
    T = tf_b.lookup_transform(source_frame, target_frame, rospy.Time(), rospy.Duration(1.0))
    trans = [T.transform.translation.x, T.transform.translation.y, T.transform.translation.z]
    rot = [T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z, T.transform.rotation.w]
    #broadcast_pose(trans+rot, source_frame, "get_mat_test")
    rot = euler_from_quaternion(rot)
    return vector2mat(trans, rot)

def mat_to_vec(mat):
    pos = mat[:3, 3]
    rot = tf.transformations.euler_from_matrix(mat)
    return list(pos) + list(rot)

def vector_transform(vec, source, target, tf_b = None, listener = None):
    mat_ = get_tf_mat(source, target, tf_b, listener) 
    adj_ = get_adjoint_transform(mat_)
    return adj_ @ vec

def vector_inverse(vec):
    mat_ = vector2matinv(vec[:3], vec[3:])
    return mat_to_vec(mat_)

def get_space_jacobian(mat):
    rot_ = mat[:3, :3]
    trans_ = mat[:3, 3]
    sj = np.zeros((6,6))
    sj[:3, :3] = rot_
    sj[3:, 3:] = rot_
    return sj

def get_adjoint_transform(mat):
    '''
    Unverified, ref Tucker's notes on rendundancy resolution for math.
    '''
    rot_ = mat[:3, :3]
    trans_ = mat[:3, 3]
    adj = np.zeros((6,6))
    adj[:3, :3] = rot_
    adj[3:, 3:] = rot_
    tr_sk_ = np.zeros((3,3))
    tr_sk_[0,2] = trans_[1]
    tr_sk_[2,0] = -trans_[1]
    tr_sk_[1,0] = trans_[2]
    tr_sk_[0,1] = -trans_[2]
    tr_sk_[2,1] = trans_[0]
    tr_sk_[1,2] = -trans_[0]
    adj[:3, 3:] = tr_sk_ @ rot_ #np.cross(trans_, rot_)
    return adj

def skew_symmetric_mat(vec):
    sk_ = np.zeros((3,3))
    sk_[0,2] = vec[1]
    sk_[2,0] = -sk_[0,2]
    sk_[1,0] = vec[2]
    sk_[0,1] = -sk_[1,0]
    sk_[2,1] = vec[0]
    sk_[1,2] = -sk_[2,1]
    return sk_

def skew_symmetric_mat_einsum(vec):
    a = np.eye(3)
    r = np.array([[0,1,0],\
                  [0,0,1],\
                  [1,0,0]])
    a = np.einsum('ij,i->ij', a, vec)
    a = np.einsum('ij,jk -> ki',a,r )
    a = np.einsum('ij,jk -> ki',a,r.T )
    return a - a.T

def batch_vector_skew_mat(vec):
    A = np.eye(3)
    R = np.array([[0,1,0],\
                  [0,0,1],\
                  [1,0,0]])

    assert vec.shape[0] == 3 or vec.shape[1] == 3, \
        "Expected input batch of vectors of shape 3xN "
    if vec.shape[0] != 3:
        vec = vec.T
    vec = vec.reshape(3,-1)
    
    SV = np.einsum('ij,ik -> kij', A, vec)
    SV = np.einsum('aij,jk -> aki', SV, R)
    SV = np.einsum('aij,jk -> aki', SV, R.T)
    return SV - np.einsum('ijk->ikj', SV)
    
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
    rospy.sleep(0.05)

def convert_array_to_pose(pose_array, frame_id, offset=[0.0]*3):
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
    pose_stamp.pose.position.x += offset[0]
    pose_stamp.pose.position.y += offset[1]
    pose_stamp.pose.position.z += offset[2]

    if len(pose_array) == 6:
        palm_quaternion = tf.transformations.quaternion_from_euler(
            pose_array[3], pose_array[4], pose_array[5])
    elif len(pose_array) == 7:
        palm_quaternion = pose_array[3:]
    
    pose_stamp.pose.orientation.x, pose_stamp.pose.orientation.y, \
        pose_stamp.pose.orientation.z, pose_stamp.pose.orientation.w = \
        palm_quaternion
    
    return pose_stamp

def pose_to_array(pose):
    position = [pose.position.x, pose.position.y, pose.position.z]
    orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    return position, orientation

def pose_to_6dof(pose):
    position = [pose.position.x, pose.position.y, pose.position.z]
    orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    euler_angles = list(euler_from_quaternion(orientation))
    #euler_angles.reverse()
    return position + euler_angles

def pose2cust(pose):
    qt = position2list(pose.orientation)
    qt.append(pose.orientation.w)
    r,p,y = euler_from_quaternion(qt)
    pose.orientation.x = r
    pose.orientation.y = p
    pose.orientation.z = y
    pose.orientation.w = 0.0
    return pose

def dof6_to_dof7_array(pose):
    quat  = quaternion_from_euler(pose[3], pose[4], pose[5])
    orientation = list(quat)
    return pose[:3] + orientation

def normalize_pose(pose):
        q = [pose.orientation.x, \
             pose.orientation.y, \
             pose.orientation.z, \
             pose.orientation.w]
        norm = np.linalg.norm(q)
        pose.orientation.x = q[0]/norm
        pose.orientation.y = q[1]/norm
        pose.orientation.z = q[2]/norm
        pose.orientation.w = q[3]/norm
        return pose

def rewire_pose(child, parent):
        w_x_c = pose_to_6dof(child)
        wTc = vector2mat(w_x_c[:3], w_x_c[3:])
        w_x_p = pose_to_6dof(parent)
        pTw = vector2matinv(w_x_p[:3], w_x_p[3:])
        p_x_c = mat_to_vec(pTw @ wTc)
        return p_x_c#convert_array_to_pose(p_x_c, parent_id)


def test_3d_rotation_equivalence(r1, r2):
    m1 = euler_matrix(r1[0], r1[1], r1[2])
    m2 = euler_matrix(r2[0], r2[1], r2[2])
    return is_same_transform(m1, m2)

def euler_from_mat(T):
    R = T
    if T.shape == (3,3):
        R = T[:3,:3]
    return euler_from_matrix(R, 'rxyz')

def get_twist(T1, T2):
        R1 = PyKDL.Rotation(*T1[:3,:3].reshape(-1))
        p1 = PyKDL.Vector(*T1[:3, 3])
        f1 = PyKDL.Frame(R1, p1)
        R2 = PyKDL.Rotation(*T2[:3,:3].reshape(-1))
        p2 = PyKDL.Vector(*T2[:3, 3])
        f2 = PyKDL.Frame(R2, p2)
        return PyKDL.diff(f1,f2,1)
        
