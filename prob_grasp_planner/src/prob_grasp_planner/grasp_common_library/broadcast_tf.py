#!/usr/bin/env python
import rospy
import tf
import numpy as np
from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
from prob_grasp_planner.srv import *

class BroadcastTf:
    '''
    The class to 1. create ROS services to update the grasp palm tf;
    2. broadcast the updated grasp palm tf; 3. update and broadcast the object tf.
    '''
    def __init__(self):
        rospy.init_node('grasp_palm_tf_br')
        # Grasp pose in blensor camera frame.
        self.tf_dict = {}
        self.palm_pose = None #PoseStamped()
        #self.palm_pose.header.frame_id = 'object_pose'
        #self.palm_pose.pose.orientation.w = 1.0
        self.object_pose_world = None #PoseStamped()
        #self.object_pose_world.header.frame_id = 'world'
        #self.object_pose_world.pose.orientation.w = 1.0
        self.palm_goal_pose = None #PoseStamped()
        #self.palm_goal_pose.pose.orientation.w = 1.0
        self.tf_br = tf.TransformBroadcaster()
        ns = rospy.get_namespace()
        self.tf_prefix = ns[0:-1]+"_tf/"
        if ns == "":
            self.tf_prefix = ""
        
    def broadcast_tf(self):
        '''
        Broadcast the grasp palm tf.
        '''
        if self.palm_pose is not None:
            #rospy.loginfo('Publishing pose from grasp_palm_pose '+self.palm_pose.header.frame_id )
            self.tf_br.sendTransform((self.palm_pose.pose.position.x, self.palm_pose.pose.position.y, 
                    self.palm_pose.pose.position.z),
                    (self.palm_pose.pose.orientation.x, self.palm_pose.pose.orientation.y, 
                    self.palm_pose.pose.orientation.z, self.palm_pose.pose.orientation.w),
                    rospy.Time.now(), self.tf_prefix+'grasp_palm_pose', self.palm_pose.header.frame_id)

        if self.object_pose_world is not None:
            #rospy.loginfo('Publishing pose from '+self.object_pose_world.header.frame_id+' object_pose' )
            self.tf_br.sendTransform((self.object_pose_world.pose.position.x, self.object_pose_world.pose.position.y, 
                    self.object_pose_world.pose.position.z),
                    (self.object_pose_world.pose.orientation.x, self.object_pose_world.pose.orientation.y, 
                    self.object_pose_world.pose.orientation.z, self.object_pose_world.pose.orientation.w),
                    rospy.Time.now(), 'object_pose', self.object_pose_world.header.frame_id)

        for key in self.tf_dict:
            pose = self.tf_dict[key]
            self.tf_br.sendTransform((pose.pose.position.x, pose.pose.position.y, 
                    pose.pose.position.z), (pose.pose.orientation.x, pose.pose.orientation.y, 
                    pose.pose.orientation.z, pose.pose.orientation.w),
                    rospy.Time.now(), key, pose.header.frame_id)

    def broadcast_goals(self):
        if self.palm_goal_pose is not None:
            palm_pose = self.palm_goal_pose
            tf_br = self.tf_br
            tf_br.sendTransform((palm_pose.pose.position.x, palm_pose.pose.position.y, 
                                 palm_pose.pose.position.z),
                                (palm_pose.pose.orientation.x, palm_pose.pose.orientation.y, 
                                 palm_pose.pose.orientation.z, palm_pose.pose.orientation.w),
                                rospy.Time.now(), self.tf_prefix+'palm_goal_pose',\
                                self.palm_goal_pose.header.frame_id)
            tf_br.sendTransform((-0.063, 0.000, -0.020),
                                (0.707, -0.0, 0.707, -0.0),
                                rospy.Time.now(), self.tf_prefix+'palm_shell_pose', self.tf_prefix+'palm_goal_pose')

    def handle_update_palm_goal(self, req):
        '''
        Handler to update the palm pose tf.
        '''
        self.palm_goal_pose = req.palm_pose
        response = UpdatePalmPoseResponse()
        response.success = True
        return response

    def update_palm_goal_server(self):
        '''
        Create the ROS server to update the palm tf.
        '''
        rospy.Service('update_palm_goal_pose', UpdatePalmPose, self.handle_update_palm_goal) 
        rospy.loginfo('Service update_palm_goal_pose:')
        rospy.loginfo('Ready to update palm goal pose:')

    def handle_update_palm_pose(self, req):
        '''
        Handler to update the palm pose tf.
        '''
        self.palm_pose = req.palm_pose
        response = UpdatePalmPoseResponse()
        response.success = True
        return response

    def update_palm_pose_server(self):
        '''
        Create the ROS server to update the palm tf.
        '''
        rospy.Service('update_grasp_palm_pose', UpdatePalmPose, self.handle_update_palm_pose) 
        rospy.loginfo('Service update_grasp_palm_pose:')
        rospy.loginfo('Ready to update grasp palm pose:')

    def handle_update_object_pose(self, req):
        '''
        Handler to update the object pose tf.
        '''
        self.object_pose_world = req.object_pose_world
        response = UpdateObjectPoseResponse()
        response.success = True
        return response

    def update_object_pose_server(self):
        '''
        Create the ROS server to update the object tf.
        '''
        rospy.Service('update_grasp_object_pose', UpdateObjectPose, self.handle_update_object_pose) 
        rospy.loginfo('Service update_grasp_object_pose:')
        rospy.loginfo('Ready to update grasp object pose:')


    def handle_update(self, req):
        self.tf_dict[req.frame_id] = req.pose
        response = UpdateTFResponse()
        response.success = True
        return response
    
    def update_tf_dict_server(self):
        rospy.Service('update_broadcast_tf_list', UpdateTF, self.handle_update)
        rospy.loginfo('Ready to update Broadcast TF List')
if __name__ == '__main__':
    broadcast_tf = BroadcastTf() 
    broadcast_tf.update_palm_pose_server()
    broadcast_tf.update_object_pose_server()
    broadcast_tf.update_palm_goal_server()
    broadcast_tf.update_tf_dict_server()
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        broadcast_tf.broadcast_tf()
        broadcast_tf.broadcast_goals()
        rate.sleep()


