#!/usr/bin/env python
'''
This module contains motion planning libraries for pick and place
The libraries planned to be supported:
- moveit
- ll4ma_planner
'''

import moveit_msgs.msg
import moveit_commander
import trajectory_msgs.msg
from ll4ma_pick_n_place.srv import Plan2Pose, Plan2PoseRequest, LiftUp, LiftUpRequest, \
    ExecPlan, ExecPlanRequest, Plan2Joint, Plan2JointRequest, GetIK, GetIKRequest, \
    CartPlan, CartPlanRequest
import copy
import rospy
import numpy as np
from reflex_msgs.srv import ReflexSetPreshape, ReflexSetPreshapeRequest
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Pose
from moveit_server import add_box_to_scene, remove_scene_object, StraightLinePlanner, add_mesh_to_scene, attach_object, detach_object

from reflex import ReflexGraspInterface
from matplotlib import pyplot, cm

def get_display_trajectory(joint_iterates, kdl_chain):
    dtraj = moveit_msgs.msg.DisplayTrajectory()
    dtraj.model_id = "iiwa"
    jn = kdl_chain.get_joint_names()
    dtraj.trajectory_start.joint_state.name = jn
    dtraj.trajectory_start.joint_state.position = joint_iterates[0]
    jtraj = trajectory_msgs.msg.JointTrajectory()
    jtraj.joint_names = jn
    #rospy.loginfo(f"{joint_iterates.shape[0]} points in output trajectory")
    for i in range(joint_iterates.shape[0]):
        traj_i = trajectory_msgs.msg.JointTrajectoryPoint()
        traj_i.positions = joint_iterates[i]
        traj_i.time_from_start = rospy.Duration.from_sec(i)
        jtraj.points.append(traj_i)
    dtraj.trajectory.append(moveit_msgs.msg.RobotTrajectory())
    dtraj.trajectory[0].joint_trajectory = jtraj
    return dtraj

def display_robot_trajectory(dtraj, topic):
    pub = rospy.Publisher(topic, moveit_msgs.msg.DisplayTrajectory, queue_size = 20)
    #drs.state.joint_state.header.stamp = rospy.Time.now()
    pub.publish(dtraj)

def get_limit_joints(jangles, kdl_chain):
    epsilon = 1e-6
    lim_links = []
    jlo_, jhi_ = kdl_chain.get_joint_limits()
    jn_ = kdl_chain.get_joint_names()
    for id_, j_ in enumerate(jangles):
        if (j_ <= jlo_[id_] + epsilon) or (j_ >= jhi_[id_] - epsilon):
            joint_ = kdl_chain.get_joint_by_name(jn_[id_])
            lim_links.append(joint_.child)
    return lim_links

def get_display_state(joint_array, kdl_chain):
    jn_ = kdl_chain.get_joint_names()
    drs = moveit_msgs.msg.DisplayRobotState()
    drs.state.joint_state.name = jn_
    drs.state.joint_state.position = joint_array
    
    lim_links_ = get_limit_joints(joint_array, kdl_chain)
    if len(lim_links_) > 0:
        drs.highlight_links = []
    for link_ in lim_links_:
        color_obj_ = moveit_msgs.msg.ObjectColor()
        color_obj_.id = link_
        color_obj_.color.r = 1.0
        color_obj_.color.b = 1.0
        color_obj_.color.a = 1.0
        drs.highlight_links.append(color_obj_)
    return drs

def display_robot_state(drs, topic = "display_robot_state"):
    pub = rospy.Publisher(topic, moveit_msgs.msg.DisplayRobotState, queue_size = 20)
    drs.state.joint_state.header.stamp = rospy.Time.now()
    pub.publish(drs)

def get_IK(pose, ns):
    get_ik_service_topic_ = process_namespace(ns) + 'pnp_get_IK_service'
    get_IK_client = rospy.ServiceProxy(get_ik_service_topic_, GetIK)
    req = GetIKRequest()
    req.pose = pose
    resp = get_IK_client(req)
    if resp.success:
        return resp.joint_angles
    else:
        return None

def process_namespace(ns):
    out = ns
    if not out.startswith('/'):
        out = '/' + out
    if not out.endswith('/'):
        out = out + '/'
    return out

    
    
class moveit_client:
    def __init__(self, init_node = False, arm_name = 'kuka_arm', planning_frame='world', goal_frame='palm_link', ns='/'):
        '''
        TODO description
        params:
        
        arm_name - planning group name (string)
        planning_frame - frame if of input goal poses (string)
        goal_frame - robot link to move to the goal (string)
        '''
        if init_node:
            rospy.init_node('pnp_moveit_planning_node')
        self.planning_frame = planning_frame
        self.goal_frame = goal_frame
        self.arm_name = arm_name
        #self.move_group = moveit_commander.MoveGroupCommander('kuka_arm')
        self.traj_disp_pub = rospy.Publisher('/pnp/planned_path',
                                             moveit_msgs.msg.DisplayTrajectory,
                                             queue_size=20)
        self.ns = ns
        if not self.ns.startswith('/'):
            self.ns = '/' + self.ns
        if not self.ns.endswith('/'):
            self.ns = self.ns + '/'
        plan_to_pose_service_topic_ = self.ns + 'pnp_plan_to_pose_service'
        plan_to_joint_service_topic_ = self.ns +'pnp_plan_to_joint_service'
        lift_up_service_topic_ = self.ns +'pnp_liftup_service'
        execute_plan_service_topic_ = self.ns +'pnp_execute_plan_service'
        go_home_service_topic_ = self.ns +'pnp_go_home_plan_service'
        get_ik_service_topic_ = self.ns + 'pnp_get_IK_service'
        cart_plan_service_topic_ = self.ns + 'pnp_cartesian_plan_service'
        rospy.wait_for_service(plan_to_pose_service_topic_)
        rospy.wait_for_service(lift_up_service_topic_)
        self.plan_to_pose_client = rospy.ServiceProxy(plan_to_pose_service_topic_, Plan2Pose)
        self.plan_to_joint_client = rospy.ServiceProxy(plan_to_joint_service_topic_, Plan2Joint)
        self.lift_up_client = rospy.ServiceProxy(lift_up_service_topic_, LiftUp)
        self.execute_plan_client = rospy.ServiceProxy(execute_plan_service_topic_, ExecPlan)
        self.go_home_client = rospy.ServiceProxy(go_home_service_topic_, LiftUp)
        self.get_IK_client = rospy.ServiceProxy(get_ik_service_topic_, GetIK)
        self.cart_plan_client = rospy.ServiceProxy(cart_plan_service_topic_, CartPlan)
        self.GRASP_OBJ_NAME = 'grasp_object'
        self.coll_obj_names = []

    def go_home(self):
        plan_req_ = LiftUpRequest()
        plan_resp_ = self.go_home_client(plan_req_)
        return plan_resp_.plan
        
    def plan_to_pose(self, pose):
        plan_req_ = Plan2PoseRequest()
        plan_req_.pose = pose
        plan_resp_ = self.plan_to_pose_client(plan_req_)
        #if plan_resp_.success:
            #self.display_plan(plan_resp_.plan)
        return plan_resp_.plan

    def plan_to_joint(self, thetas, target_orientation = None, orientation_tols=None, rrt=False):
        plan_req_ = Plan2JointRequest()
        plan_req_.joint_target = thetas
        plan_req_.constrained = False
        plan_req_.rrt = rrt
        if target_orientation is not None:
            plan_req_.constrained = True
            plan_req_.target = target_orientation
            plan_req_.tolerances = orientation_tols
        plan_resp_ = self.plan_to_joint_client(plan_req_)
        return plan_resp_.plan

    def get_IK(self, pose):
        req = GetIKRequest()
        req.pose = pose
        resp = self.get_IK_client(req)
        if resp.success:
            return resp.joint_angles
        else:
            return None

    def lift_up(self, kdl, q = None, velocity = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0], target = None):
        #velocity[3:] = omegas
        #print(velocity)
        tplanner = StraightLinePlanner(kdl)
        plan = tplanner.plan(velocity, q, target)
        
        #plan_req_ = LiftUpRequest()
        #plan_resp_ = self.lift_up_client(plan_req_)
        #if plan_resp_.success:
            #self.display_plan(plan_resp_.plan)
        return plan

    def cart_plan(self, distance):
        plan_req_ = CartPlanRequest()
        plan_req_.distance = distance
        plan_resp_ = self.cart_plan_client(plan_req_)
        return plan_resp_.plan

    def q_vel_traj(self, kdl, q_vel):
        tplanner = StraightLinePlanner(kdl)
        plan = tplanner.plan_direct_joint_vel(q_vel)
        return plan
        
    def display_plan(self, jtraj):
        dtraj = moveit_msgs.msg.DisplayTrajectory()
        dtraj.model_id = "iiwa"
        dtraj.trajectory_start.joint_state.name = jtraj.joint_names
        dtraj.trajectory_start.joint_state.position = jtraj.points[0].positions
        dtraj.trajectory_start.joint_state.velocity = jtraj.points[0].velocities
        dtraj.trajectory.append(moveit_msgs.msg.RobotTrajectory())
        dtraj.trajectory[0].joint_trajectory = jtraj
        #disp_traj.trajectory.append(jtraj)
        for i in range(100):
            self.traj_disp_pub.publish(dtraj)
            rospy.sleep(0.01)

    def execute_plan(self, arm_plan):
        if(arm_plan is None):
            return False
        return self.execute_plan_client(arm_plan)

    def add_object_bb(self, obj_size, name=None, pose=Pose(), frame='object_pose'):
        bb_pose = PoseStamped()
        bb_pose.header.frame_id = frame
        bb_pose.pose = pose
        if name==None:
            name = self.GRASP_OBJ_NAME
        add_box_to_scene(name, bb_pose, obj_size)
    def add_object_mesh(self, name, frame_id = 'object_pose',  pose = Pose()):
        obj_pose = PoseStamped()
        pose.orientation.w = 1
        obj_pose.pose = pose
        obj_pose.header.frame_id = frame_id
        rospy.loginfo('calling moveit server')
        self.coll_obj_names.append(name)
        add_mesh_to_scene(name, obj_pose, name + '.stl')
        
    def remove_object(self, name=None):
        if not name:
            name = self.GRASP_OBJ_NAME
        remove_scene_object(name)

    def remove_objects(self):
        for name in self.coll_obj_names:
            remove_scene_object(name)


    def attach_object(self, pose_stamped, size):
        attach_object(pose_stamped, size)

    def detach_object(self):
        detach_object()


    def plan_and_exec_loop(self, joint_target, target_orientation = None, tols=None, plot=False, rrt=False):
        try:
            traj = self.plan_to_joint(joint_target, target_orientation, tols)
        except:
            rospy.logwarn("Planning Failed")

        set_to = target_orientation
        set_tols = tols
        uip = 'r'
        while uip.lower() not in ['y', 'yes']:
            if plot:
                self.plot_traj(traj)
            uip = input("Execute trajectory?\ty - Execute\tq - Quit\nDefault - Replan:")
            if uip.lower() in ['q', 'quit']:
                rospy.loginfo("Aborting...")
                #rospy.signal_shutdown("User aborted execution")
                return False
            elif uip.lower() in ['c']:
                if set_to is None:
                    rospy.loginfo("Setting Constraints")
                    set_to = target_orientation
                    set_tols = tols
                else:
                    rospy.loginfo("Unsetting Constraints")
                    set_to = None
                    set_tols = None
            elif uip.lower() not in ['y', 'yes']:
                try:
                    traj = self.plan_to_joint(joint_target, set_to, set_tols, rrt=rrt)
                except:
                    rospy.logwarn("Planning Failed")
        self.execute_plan(traj)
        return True

    def plot_traj(self, traj):
        jtraj = traj.joint_trajectory
        all_positions = [point.positions for point in jtraj.points]
        all_positions = np.array(all_positions)
        labels = jtraj.joint_names
        for i in range(all_positions.shape[1]):
            pyplot.plot(range(all_positions.shape[0]), all_positions[:,i], label=labels[i])
        pyplot.legend()
        pyplot.show()
        
        
class reflex_client:
    def __init__(self):
        #rospy.loginfo("waiting for reflex services")
        #rospy.wait_for_service('/reflex2_grasp_interface/open_hand')
        #self.hand_open_call = rospy.ServiceProxy("/reflex2_grasp_interface/open_hand", Trigger)
        #rospy.wait_for_service("/reflex2_grasp_interface/grasp")
        #self.hand_grasp_call = rospy.ServiceProxy("/reflex2_grasp_interface/grasp", Trigger)
        #rospy.wait_for_service("/reflex2_grasp_interface/set_preshape")
        #self.hand_preshape_call = rospy.ServiceProxy("/reflex2_grasp_interface/set_preshape", ReflexSetPreshape)

        self.POSE_CMD_TOPIC = "reflex_takktile2/command_position"
        self.FORCE_CMD_TOPIC = "reflex_takktile2/command_motor_force"

        self.hand = ReflexGraspInterface(namespace="left/reflex_takktile2")

    
    def release(self, full=True):
        #self.hand_open_call()
        if full:
            self.hand.open_hand()
        else:
            self.hand.release()

    def set_preshape(self, preshape):
        #PreshapeReq = ReflexSetPreshapeRequest()
        #PreshapeReq.preshape = preshape
        #self.hand_preshape_call(PreshapeReq)
        self.hand.set_preshape(preshape)

    def grasp(self, velocity=1.0, force=100, rate=100):
        
        #self.hand_grasp_call()
        self.hand.grasp(force=force, tighten_increment=0.1)
