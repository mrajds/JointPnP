#!/usr/bin/env python

import trimesh
import numpy as np
import copy
import sys
import time

#Ros Stuff
import rospy
import std_msgs.msg
from gpu_sd_map.transforms_lib import *
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray

class GripperVox:
    def __init__(self, obj):
        if isinstance(obj, str):
            try:
                file_ = open(obj, 'rb')
            except FileNotFoundError:
                print("File \'%s\' cannot be opened", obj)
                exit(1)
            self.VoxelGrid_ = trimesh.exchange.binvox.load_binvox(file_)
        elif isinstance(obj, GripperVox):
            self.VoxelGrid_ = copy.deepcopy(obj.VoxelGrid_)
        else:
            print("Error unknow type for parameter \'obj\'")
            exit(1)
        self.VoxelGrid_.hollow()

    def print_metadata(self):
        print("Num Voxels: %d", len(self.VoxelGrid_.points))
        print("Size per Voxel (Pitch): %f", self.VoxelGrid_.pitch)

    def set_trans_(self, transform):
        assert transform.shape == (4,4), "Not a transform matrix, check shape"
        self.VoxelGrid_.apply_transform(transform)

class ReflexAug:
    def __init__(self, Base, Finger):
        '''
        Initializes the articulated reflex voxel grid using the Base and Finger voxel grids.
        Calls configure_() to initialize precomputed link values.
        
        Params:
        Base - voxel grid of reflex shell + pad (GripperVox)
        Finger - voxel grid of reflex proximal + distal + flex (GripperVox)

        Output:
        None - this is a constructor

        Status: Working

        TODO:
        - Save only points at zero not GripperVox object.
        - Verify compute efficiency on transforming only points.
        - Make this compatible/parameterized for reflex2.
        - Implement finger closure.
        '''
        if not isinstance(Base, GripperVox) or not isinstance(Finger, GripperVox):
            print("Error Args \'Base\' \(or\) \'Finger\' are not of type 'GripperVox'")
            exit(1)
        self.Base = Base
        self.F1 = GripperVox(Finger)
        self.F2 = GripperVox(Finger)
        self.F3 = GripperVox(Finger)
        #self.Base.VoxelGrid_.points = self.bin_points(self.Base.VoxelGrid_.points)
        #self.F1.VoxelGrid_.points = self.bin_points(self.F1.VoxelGrid_.points)
        #self.F2.VoxelGrid_.points = self.bin_points(self.F2.VoxelGrid_.points)
        #self.F3.VoxelGrid_.points = self.bin_points(self.F3.VoxelGrid_.points)
        self.F1_Mat = np.eye(4)
        self.F2_Mat = np.eye(4)
        self.F3_Mat = np.eye(4)
        self.Base_Mat = np.eye(4)
        self.configure_()
        
        Base_max_dim = max(np.amax(Base.VoxelGrid_.points, axis = 0) - np.amin(Base.VoxelGrid_.points, axis = 0))
        Finger_max_dim = max(np.amax(Finger.VoxelGrid_.points, axis = 0) - np.amin(Finger.VoxelGrid_.points, axis = 0))
        self.Kernel_Size = int((Base_max_dim/2 + 2*Finger_max_dim) * 1000 + 10)
        self.Kernel_Size += 1 - self.Kernel_Size % 2
        self.atZero_ = True
        print("Setting Kernel Size as", self.Kernel_Size)
        self.Base_Offset = vector2mat([0.00, 0.01, 0.0])
        self.Finger_Offset = vector2mat([0.0, 0.00815, 0.0])
        return

    def copy(self):
        return copy.deepcopy(self)

    def Reset(self):
        '''
        Resets palm pose and preshapes to zero angle.

        Params:
        None
        
        Output:
        None - sets instance transforms to identity

        Status: Working
        
        Testing: Visual Verified
        '''
        self.F1.set_trans_(invert_trans(self.F1_Mat))
        self.F2.set_trans_(invert_trans(self.F2_Mat))
        self.F3.set_trans_(invert_trans(self.F3_Mat))
        self.Base.set_trans_(invert_trans(self.Base_Mat))
        self.F1_Mat = self.F1_Mat @ invert_trans(self.F1_Mat)
        self.F2_Mat = self.F2_Mat @ invert_trans(self.F2_Mat)
        self.F3_Mat = self.F3_Mat @ invert_trans(self.F3_Mat)
        self.Base_Mat = self.Base_Mat @ invert_trans(self.Base_Mat)
        self.atZero_ = True
        

    def configure_(self):
        '''
        Configures the transforms for each part of the hand based on fixed link parameters from the URDF.
        
        Params:
        None - values hardcoded based on URDF params

        Output:
        None - sets class instances with precomputed values

        Status: Working

        Testing: Visual Verified
        
        TODO:
        - Read these values directly from URDF, making Reflex2 dev easy.
        - Make this reflex2 compatible.
        '''
        pi = 22.0 / 7.0
        #TODO: Get this values directly from URDF / robot_description?
        #Swivel Joint Offsets (Values from URDF)
        Sw1_origin = [0.0503973683071414, -0.026, 0.063]
        Sh2Sw1_Mat = vector2mat(Sw1_origin)
        Sw2_origin = [0.0503973683071414, 0.026, 0.063]
        Sh2Sw2_Mat = vector2mat(Sw2_origin)

        #Finger Offsets (Values from URDF)
        F1_origin = [0.01, 0.0, 0.0186]
        F1_rpy = [0.0, 0.28, 0.0]
        self.Sw2F1_Mat = vector2mat(F1_origin, F1_rpy)
        F2_origin = F1_origin
        F2_rpy = F1_rpy
        self.Sw2F2_Mat = self.Sw2F1_Mat
        F3_origin = [-0.03, 0.0, 0.0816]
        F3_rpy = [0.0, 0.28, pi]
        Sh2F3_Mat = vector2mat(F3_origin, F3_rpy)

        #Offset Value to Move Finger to Pad (from Shell)
        Sh2P_Trans = [0.02, 0.0, 0.063]
        Sh2P_Rot = [pi/2.0, 0.0, -pi/2.0]
        P2Sh_Mat = vector2matinv(Sh2P_Trans, Sh2P_Rot)

        P2Pm_Rot = [pi, 0.0, pi/2.0]
        self.Pm2P_Mat = vector2matinv(rpy = P2Pm_Rot)

        self.P2Sw1_Mat = P2Sh_Mat @ Sh2Sw1_Mat
        self.P2Sw2_Mat = P2Sh_Mat @ Sh2Sw2_Mat
        P2F3_Mat = P2Sh_Mat @ Sh2F3_Mat

        self.Pm2F3_Mat = self.Pm2P_Mat @ P2F3_Mat #For fingers 1 & 2 this is done in set_preshape_pose based on preshape value

    def apply_transforms_(self):
        '''
        Applies the configration transfroms to the voxel grid

        Params:
        None - Reads transforms from instance.

        Output:
        None - Invokes call to apply transforms.

        Status: Working

        Testing: Visual Verified
        '''
        self.F1.set_trans_(self.F1_Mat)
        self.F2.set_trans_(self.F2_Mat)
        self.F3.set_trans_(self.F3_Mat) #F3 Matches Shell here (F3 has no Swivel to match)
        self.Base.set_trans_(self.Base_Mat)
        self.atZero_ = False
        

    def Set_Pose_Preshape(self, rpy=[0.0, 0.0, 0.0], preshape = 0.0, render = False, ignore_fingers=False, centered=True):
        '''
        Sets the 3d orientation and preshape of the hand.

        Params:
        rpy - 1 x 3 vector of euler orientations (radians)
        preshape - preshape joint angles (radians), affects finger 1 and 2.
        render - if the gripper needs to be visualized (bool)

        Output:
        None - sets transforms for members F1, F2, F3 & Base

        Status: Working

        Testing: Visual Verified
        '''
        if not self.atZero_:
            self.Reset()

        F1_Preshape_Mat = vector2mat(rpy=[0.0, 0.0,-preshape])
        F2_Preshape_Mat = vector2mat(rpy=[0.0, 0.0, preshape])

        #Config_Mat
        Cnf2Pm = vector2mat(rpy=rpy)

        P2F1_Mat = self.P2Sw1_Mat @ F1_Preshape_Mat @ self.Sw2F1_Mat
        P2F2_Mat = self.P2Sw2_Mat @ F2_Preshape_Mat @ self.Sw2F2_Mat

        Pm2F1_Mat = self.Pm2P_Mat @ P2F1_Mat
        Pm2F2_Mat = self.Pm2P_Mat @ P2F2_Mat
        # self.Pm2F3_Mat used fixed params hence precomputed in configure_()

        self.F1_Mat = Cnf2Pm @ Pm2F1_Mat
        self.F2_Mat = Cnf2Pm @ Pm2F2_Mat
        self.F3_Mat = Cnf2Pm @ self.Pm2F3_Mat
        self.Base_Mat = Cnf2Pm

        if centered:
            self.F1_Mat = self.F1_Mat @ self.Finger_Offset
            self.F2_Mat = self.F2_Mat @ self.Finger_Offset
            self.F3_Mat = self.F3_Mat @ self.Finger_Offset
            #self.Base_Mat = self.Base_Mat @ self.Base_Offset 
        
        self.apply_transforms_()
        
        if render:
            self.build_gripper_()
        return self.extract_points_(ignore_fingers=ignore_fingers, centered=centered)

    def Get_Derivate_Preshape(self, rpy, preshape, ignore_fingers=False):
        '''
        Computes the partial derivative of the points on voxel grid w.r.t. preshape values.

        Params:
        rpy - 1 x 3 vector of euler orientations (radians)
        preshape - preshape angle to compute the derivate at (radians)

        Output:
        N x 3 Matrix (N number of points, 3 dof for point representation)
        
        Status: Working

        Testing: Visual & Cross Verified (against finite differencing)

        TODO:
        - The transforms are extracted apply them to a copy of the voxel grid and extract the points
        '''

        st = time.time()
        
        dRF1_preshape_ = get_derivate_elementary_rotation(-preshape, 'z')
        dRF2_preshape_ = get_derivate_elementary_rotation(preshape, 'z')

        dF1_preshape_Mat_ = rot2mat(dRF1_preshape_)
        dF1_preshape_Mat_[3][3] = 0 #For derivate of homogenrous transform
        dF2_preshape_Mat_ = rot2mat(dRF2_preshape_)
        dF2_preshape_Mat_[3][3] = 0 #For derivate of homogenrous transform

        dP2F1_Mat_ = self.P2Sw1_Mat @ dF1_preshape_Mat_ @ self.Sw2F1_Mat
        dP2F2_Mat_ = self.P2Sw2_Mat @ dF2_preshape_Mat_ @ self.Sw2F2_Mat

        dPm2F1_Mat_ = self.Pm2P_Mat @ dP2F1_Mat_
        dPm2F2_Mat_ = self.Pm2P_Mat @ dP2F2_Mat_

        Cnf2Pm_ = vector2mat(rpy=rpy)

        dF1_Mat = Cnf2Pm_ @ dPm2F1_Mat_
        dF2_Mat = Cnf2Pm_ @ dPm2F2_Mat_

        # Finger 3 and Base are unaffected by preshape changes hence zeros
        dF3_Mat = np.zeros((4,4))
        dF3_Mat[3][3] = 1
        dBase_Mat = np.zeros((4,4))
        dBase_Mat[3][3] = 1

        dReflex = self.copy()
        dReflex.Reset()
        dReflex.F1_Mat = dF1_Mat
        dReflex.F2_Mat = dF2_Mat
        dReflex.F3_Mat = dF3_Mat
        dReflex.Base_Mat = dBase_Mat

        p1 = time.time()
        #print("Derivate precompute took: {} secs".format(p1 - st))

        dReflex.apply_transforms_()
        #print("Apply transforms took: {} secs".format(time.time() - p1))
        #print("Total time took: {} secs".format(time.time() - st))
        dpoints = dReflex.extract_points_(True, ignore_fingers=ignore_fingers)
        return dpoints

    def Get_Derivate_Orientation(self, rpy, preshape):
        '''
        Computed the partial derivative of the points on voxel grid w.r.t. palm orientation.

        Params:
        rpy - 1 x 3 vector of euler orientations (radians)
        preshape - preshape angle (radians)
        
        Output:
        (N x 3) x 3 Matrix (N number of points, 3 dof for point representation, 3 euler angles)
       
        Status: Working

        Testing: Visual & Cross Verified (against finite differencing)
        '''

        r, p, y = rpy
        
        F1_Preshape_Mat = vector2mat(rpy=[0.0, 0.0,-preshape])
        F2_Preshape_Mat = vector2mat(rpy=[0.0, 0.0, preshape])

        P2F1_Mat = self.P2Sw1_Mat @ F1_Preshape_Mat @ self.Sw2F1_Mat
        P2F2_Mat = self.P2Sw2_Mat @ F2_Preshape_Mat @ self.Sw2F2_Mat

        Pm2F1_Mat = self.Pm2P_Mat @ P2F1_Mat
        Pm2F2_Mat = self.Pm2P_Mat @ P2F2_Mat
        # self.Pm2F3_Mat used fixed params hence precomputed in configure_()

        dRPY_rot_ = get_rotation_jacobian(r, p, y)

        dCnf2Pm_x = rot2mat(dRPY_rot_[0])
        dCnf2Pm_x[3][3] = 0
        dCnf2Pm_y = rot2mat(dRPY_rot_[1])
        dCnf2Pm_y[3][3] = 0
        dCnf2Pm_z = rot2mat(dRPY_rot_[2])
        dCnf2Pm_z[3][3] = 0

        def apply_conf_(dCnf2Pm):
            dF1_Mat = dCnf2Pm @ Pm2F1_Mat
            dF2_Mat = dCnf2Pm @ Pm2F2_Mat
            dF3_Mat = dCnf2Pm @ self.Pm2F3_Mat
            dBase_Mat = dCnf2Pm
            dReflex = self.copy()
            dReflex.Reset()
            dReflex.F1_Mat = dF1_Mat
            dReflex.F2_Mat = dF2_Mat
            dReflex.F3_Mat = dF3_Mat
            dReflex.Base_Mat = dBase_Mat
            dReflex.apply_transforms_()
            return dReflex.extract_points_()

        points_dx = apply_conf_(dCnf2Pm_x)
        points_dy = apply_conf_(dCnf2Pm_y)
        points_dz = apply_conf_(dCnf2Pm_z)

        return points_dx, points_dy, points_dz

    def extract_points_(self, neg_f1 = False, ignore_fingers=False, centered=False):
        '''
        Extract points from the voxel grid
        
        Params:
        None 

        Output:
        N x 3 matrix - points of the voxel grip (float)

        Status: Working

        Testing: Visual Verified
        '''
        st = time.time()
        Base_points = self.Base.VoxelGrid_.points
        F1_points = self.F1.VoxelGrid_.points
        if neg_f1:
            F1_points = -F1_points
        F2_points = self.F2.VoxelGrid_.points
        F3_points = self.F3.VoxelGrid_.points
        #if centered:
        #    Base_points += (self.Base_Mat[:3, :3] @ self.Base_Offset.T[:3]).T[0, :3]
        #    F1_points -= (self.F1_Mat[:3, :3] @ self.Finger_Offset.T[:3]).T[0, :3]
        #    F2_points -= (self.F2_Mat[:3, :3] @ self.Finger_Offset.T[:3]).T[0, :3]
        #    F3_points += (self.F3_Mat[:3, :3] @ self.Finger_Offset.T[:3]).T[0, :3]
        end = time.time() - st
        #print("Extraction took {} secs".format(end))
        if not ignore_fingers:
            points = np.vstack((Base_points, F1_points, F2_points, F3_points))
        else:
            points = Base_points
        return points

    def build_kernel_(self):
        Kernel = np.zeros((self.Kernel_Size, self.Kernel_Size, self.Kernel_Size))
        points = self.extract_points_()
        indices = points * 1000
        indices_L = np.floor(indices).astype(np.int)
        indices_H = np.ceil(indices).astype(np.int)
        indices = np.vstack((indices_L, indices_H))
        #indices = trimesh.voxel.ops.points_to_indices(points, 1e-3)
        indices += int(self.Kernel_Size / 2) + 1 #To offset negative indices
        #start = time.time()
        Kernel[tuple(indices.T)] = 1
        #end = time.time() - start
        #print(sys.getsizeof(Kernel))
        #print("Took %d secs", end)
        return Kernel
        
    def build_gripper_(self):
        points = self.extract_points_()
        mesh = trimesh.voxel.ops.points_to_marching_cubes(points, 1e-3)
        mesh.show()

    def bin_points(self, points, binsize = 0.01):
        binnedpts = []
        binnedpts2 = []
        for i in range(3):
            bins = np.arange(points.min(axis = 0)[i], \
                             points.max(axis = 0)[i] + binsize*2, \
                             binsize)
            binnedpts.append(bins[np.digitize(points[:,i], bins, right=False)])
            binnedpts2.append(bins[np.digitize(points[:,i], bins, right=True)])
        binnedpts = np.vstack(binnedpts).T
        binnedpts2 = np.vstack(binnedpts2).T
        binned = np.unique(np.vstack([binnedpts, binnedpts2]), axis = 0) - binsize/2
        return binned
    
def visualize_points(points):
    print(points.shape)
    mesh = trimesh.voxel.ops.points_to_marching_cubes(points)
    mesh.show()

def Publish_Points_As_Cloud(points, topic = "Gripper_Voxelized"):
    '''
    Publish a list of points as pointcloud ros topic
    
    Params:
    points - N x 3 array of points in 3d (float).
    
    Output:
    None - rostopic "Gripper_Voxelized" is published.

    Status: Working, Update

    Testing: Visual Verified

    TODO:
    - Move this to a common library.
    '''
    pcl_pub_ = rospy.Publisher(topic, PointCloud, queue_size=10)
    gripper_cloud_ = PointCloud()
    gripper_cloud_.header = std_msgs.msg.Header()
    gripper_cloud_.header.stamp = rospy.Time.now()
    gripper_cloud_.header.frame_id = "reflex_palm_link"

    for i in range(points.shape[0]):
        gripper_cloud_.points.append(Point32(points[i][0], points[i][1], points[i][2]))

    rate = rospy.Rate(10)
    for i in range(10):
        pcl_pub_.publish(gripper_cloud_)
        rate.sleep()

def Publish_2Point_Vectors(start_pts, end_pts, color = [1.0, 0.0, 0.0], topic="/Gradient_Vectors", step=100):
    '''
    Publish rviz Arror marker array representing vectors between 2 points

    Params:
    start_pts - N x 3 array of 3d points (float)
    end_pts - N x 3 array of 3d points (float)
    color - 1 x 3 vector of 0.0 to 1.0 RGB values (float)
    topic - topic to publish (str)
    step - sample interval (int)

    Output:
    None - rostopic is published.

    Status: Working

    Testing: Visual Verified
    '''
    assert start_pts.shape[0] == end_pts.shape[0], "Start, End points size mismatch ({},{}) received".format(start_pts.shape[0], end_pts.shape[0])
    assert start_pts.shape[1] == 3, "Start point are not 3D, {} dim(s) received".format(start_pts.shape[1])
    assert end_pts.shape[1] == 3, "Start point are not 3D, {} dim(s) received".format(end_pts.shape[1]) 
    marray_pub_ = rospy.Publisher(topic, MarkerArray, queue_size=10)
    marray = MarkerArray()
    ctr = 0
    for i in range(0, start_pts.shape[0], step):
        ctr += 1
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = ctr
        marker.ns = "grads"
        marker.scale.x = 0.001
        marker.scale.y = 0.004
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.points.append(Point32(start_pts[i][0], start_pts[i][1], start_pts[i][2]))
        marker.points.append(Point32(end_pts[i][0], end_pts[i][1], end_pts[i][2]))
        marray.markers.append(marker)
    rospy.loginfo("Total {} vectors on topic \"{}\"".format(ctr, topic))
    rate = rospy.Rate(100)
    for i in range(100):
        marray_pub_.publish(marray)
        rate.sleep()
    
if __name__ == '__main__':
    Base_BV_Path = "../gripper_augmentation/base_6.binvox" #Use rospack find to generate the full path
    Finger_BV_Path = "../gripper_augmentation/finger_sweep.binvox"
    Base = GripperVox(Base_BV_Path)
    Finger1 = GripperVox(Finger_BV_Path)
    Finger2 = GripperVox(Finger1)
    Base.print_metadata()
    Finger2.print_metadata()
    Gripper = ReflexAug(Base, Finger1)
    Gripper.Base.print_metadata()
    Gripper.F1.print_metadata()

    set_preshape = 0.0
    rpy = [0.0, 0.0, 0.0]
    Gripper.Set_Pose_Preshape(rpy, preshape = set_preshape, render = False)
    points = Gripper.extract_points_()

    start_d = time.time()
    dpoints_dp = Gripper.Get_Derivate_Preshape(rpy, set_preshape)
    dpoints_do = Gripper.Get_Derivate_Orientation(rpy, set_preshape)
    tot_d = time.time() - start_d

    rospy.init_node("Gripper_Aug_Visualize")
    rospy.loginfo("Gradient computation took {} secs".format(tot_d))
    Publish_Points_As_Cloud(points, '/left/reflex_binned')
    exit(0)
    Publish_2Point_Vectors(points, points + dpoints_dp, [1.0, 1.0, 0.0], "preshape_grads")
    Publish_2Point_Vectors(points, points + dpoints_do[0], [1.0, 0.0, 0.0], "roll_grads")
    Publish_2Point_Vectors(points, points + dpoints_do[1], [0.0, 1.0, 0.0], "pitch_grads")
    Publish_2Point_Vectors(points, points + dpoints_do[2], [0.0, 0.0, 1.0], "yaw_grads")

    st_fd = time.time()
    delt = 0.001
    rospy.loginfo("FD Preshape")
    Gripper.Set_Pose_Preshape(rpy, preshape = set_preshape + delt, render = False)
    points1 = Gripper.extract_points_()
    points1 = (points1 - points)/delt
    Publish_2Point_Vectors(points, points + points1, [0.0, 1.0, 1.0], "fd_grads")
    for i in range(3):
        rospy.loginfo("FD Orientation axis {}".format(i+1))
        rpy[i] += delt
        Gripper.Set_Pose_Preshape(rpy, preshape = set_preshape, render = False)
        points1 = Gripper.extract_points_()
        points1 = (points1 - points)/delt
        rpy[i] -= delt
    tot_fd = time.time() - st_fd
    rospy.loginfo("FD gradient computation took {} secs".format(tot_fd))
    Publish_2Point_Vectors(points, points + points1, [0.0, 1.0, 1.0], "fd_grads")
    Publish_Points_As_Cloud(points + points1, "Projected_Step")
    rospy.spin()
