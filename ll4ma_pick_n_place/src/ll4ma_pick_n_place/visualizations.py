import numpy as np

import tf
import rospy
import tf2_ros
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs

from geometry_msgs.msg import Point32
from visualization_msgs.msg import Marker, MarkerArray
from gpu_sd_map.ros_transforms_lib import pose_to_array
from matplotlib import pyplot, cm

from prob_grasp_planner.srv import UpdateTF, UpdateTFRequest

BROADCAST_SERVICE_ = '/update_broadcast_tf_list'

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

def broadcast_tf(frame_id, pose_stamped):
    rospy.wait_for_service(BROADCAST_SERVICE_)
    tf_server = rospy.ServiceProxy(BROADCAST_SERVICE_, UpdateTF)
    req = UpdateTFRequest()
    req.frame_id = frame_id
    req.pose = pose_stamped
    tf_server(req)

def broadcast_poses(pose_dict, parent):
    for child_id in pose_dict:
        broadcast_pose(pose_dict[child_id], parent, child_id)

def broadcast_pose(pose, parent, child):
    br = tf.TransformBroadcaster()
    trans, rot = pose_to_array(pose)
    for i in range(10):
        br.sendTransform(trans, rot, rospy.Time.now(), child, parent)
        rospy.sleep(0.1)

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

        
def map_rgb(scalars, hi=None, lo=None):
    pos = scalars > 1
    neg = scalars < -1
    if hi is None:
        hi = np.max(scalars)
    if lo is None:
        lo = np.min(scalars)
    scalars[(scalars >= -1) & (scalars <=1)] = 0.5
    scalars[pos] *= 0.25/hi
    scalars[pos] += 0.75
    scalars[neg] -= lo
    scalars[neg] *= 0.25/(-lo)
    return cm.rainbow(scalars)

def numpy_to_pointcloud2(points, colors, frame):
    N = points.shape[0] #Number of points
    assert colors.shape[0] == N, f"Expected colors dim 0 to be same as number of points ({N})," + \
        f"but got colors of dim (colors.shape)"
    assert points.shape[1] == 3, "points are not 3 dimensional"
    assert colors.shape[1] >= 3 and colors.shape[1] <= 4, "Expected colors to be rgb or rgba"
    if colors.shape[1] == 3:
        colors = np.hstack([colors, np.ones((N,1))])

    #https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0#file-dragon_pointcloud-py-L20-L41
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
    data = np.hstack([points, colors]).astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = std_msgs.Header(frame_id=frame, stamp=rospy.Time.now())

    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=N,
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * N),
        data=data
    )

def merge_pointcloud2s(clouds):
    header = std_msgs.Header(frame_id=clouds[0].header.frame_id, stamp=rospy.Time.now())

    data = []
    point_step = clouds[0].point_step
    width = 0
    for cloud in clouds:
        data += cloud.data
        width += cloud.width 
    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=width,
        is_dense=False,
        is_bigendian=False,
        fields=clouds[0].fields,
        point_step=point_step,
        row_step=(point_step * width),
        data=data
    )
    

def publish_pointcloud2(pc2, topic):
    pcl_pub = rospy.Publisher(topic, sensor_msgs.PointCloud2, queue_size=10)
    rate = rospy.Rate(100)
    pcl_pub.publish(pc2)
    rate.sleep()


def publish_2point_vectors(start_pts, end_pts, colors, frame, topic="/Gradient_Vectors", step=1):
    '''
    Publish rviz Arror marker array representing vectors between 2 points
    Copied from gpu_conv_sdm/gripper_augment

    Params:
    start_pts - N x 3 array of 3d points (float)
    end_pts - N x 3 array of 3d points (float)
    colors - N x 3 vector of 0.0 to 1.0 RGB values (float)
    topic - topic to publish (str)
    step - sample interval (int)

    Output:
    None - rostopic is published.

    Status: Working

    Testing: Visual Verified
    '''
    assert start_pts.shape[0] == end_pts.shape[0], "Start, End points size mismatch ({},{}) received".format(start_pts.shape[0], end_pts.shape[0])
    assert start_pts.shape[0] == colors.shape[0], "colors must be same shape as points"
    assert start_pts.shape[1] == 3, "Start point are not 3D, {} dim(s) received".format(start_pts.shape[1])
    assert end_pts.shape[1] == 3, "Start point are not 3D, {} dim(s) received".format(end_pts.shape[1]) 
    marray_pub_ = rospy.Publisher(topic, MarkerArray, queue_size=10)
    marray = MarkerArray()
    ctr = 0
    for i in range(0, start_pts.shape[0], step):
        ctr += 1
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = ctr
        marker.ns = "grads"
        marker.scale.x = 0.001
        marker.scale.y = 0.004
        marker.color.a = 1.0
        marker.color.r = colors[i][0]
        marker.color.g = colors[i][1]
        marker.color.b = colors[i][2]
        marker.points.append(Point32(start_pts[i][0], start_pts[i][1], start_pts[i][2]))
        marker.points.append(Point32(end_pts[i][0], end_pts[i][1], end_pts[i][2]))
        marray.markers.append(marker)
    rospy.loginfo("Total {} vectors on topic \"{}\"".format(ctr, topic))
    rate = rospy.Rate(100)
    for i in range(100):
        marray_pub_.publish(marray)
        rate.sleep()
