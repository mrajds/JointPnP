#!/usr/bin/env python

import numpy as np
import moveit_msgs.msg
import shape_msgs.msg
import moveit_commander
import copy
import rospy
import sys
from ll4ma_pick_n_place.srv import Plan2Pose, Plan2PoseResponse, LiftUp, LiftUpResponse, \
    ExecPlan, ExecPlanResponse, Plan2Joint, Plan2JointResponse, GetIK, GetIKResponse, \
    CartPlan, CartPlanResponse
from trac_ik_python.trac_ik import IK
from moveit_interface.iiwa_planner import IiwaPlanner
#from grasp_pipeline.robot_traj_interface import robotTrajInterface
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ll4ma_util import ros_util

from ll4ma_pick_n_place.kdl_utils import ManipulatorKDL
from ll4ma_pick_n_place.data_utils import load_env_yaml

from gpu_sd_map.ros_transforms_lib import euler_diff, get_twist
from gpu_sd_map.transforms_lib import euler_to_angular_velocity, vector2mat

MAX_VEL_FACTOR_ = 0.2
MAX_ACC_FACTOR_ = 0.1

ARM_MOVE_GROUP_ = 'iiwa_arm'

ROBOT_DESCRIPTION = rospy.get_namespace() +  'robot_description' 

class IIWA_Reflex_State:
    def __init__(self):
        self.robot_name = 'iiwa'
        self.hand_name = 'reflex_takktile2'
        self.joint_states_dict = {self.robot_name:None, self.hand_name:None}
        self.full_js = None
        self.pub_full_js = rospy.Publisher("joint_states", JointState, queue_size=1)
        for aname in self.joint_states_dict:
            rospy.Subscriber("{}/joint_states".format(aname), JointState, \
                             self._joint_state_cb, aname)
    def _joint_state_cb(self, joint_state, robot_key):
        self.joint_states_dict[robot_key] = joint_state

    def joint_states_merger(self):
        if None not in self.joint_states_dict.values():
            new_js = JointState()
            times = []
            for js in self.joint_states_dict.values():
                times.append(js.header.stamp)
                new_js.name +=  js.name
                new_js.position += js.position
                new_js.velocity += js.velocity
                new_js.effort += js.effort
            new_js.header.stamp = min(times)
            return new_js
        return None

    def publish_merged_states(self):
        self.full_js = self.joint_states_merger()
        if self.full_js is not None:
            self.pub_full_js.publish(self.full_js)
            
        
class StraightLinePlanner:

    def __init__(self, kdl, jacobian=None):
        self.kdl = kdl
        self.jacobian = jacobian
        self.dq = np.zeros(self.kdl.get_dof()) + 1e-5
        if not jacobian:
            self.jacobian = kdl.jacobian

    def damped_inv(self, A, rho=0.017):
        AA_T = A @ A.T
        damping = np.eye(A.shape[0]) * rho**2
        inv = np.linalg.inv(AA_T + damping)
        d_pinv = A.T @ inv
        return d_pinv

    def compute_manipulability(self, q, J):
        m_score = np.sqrt(np.linalg.det(J @ J.T))
        J_prime = self.jacobian(q + self.dq)
        m_prime = np.sqrt(np.linalg.det(J_prime @ J_prime.T))
        q_null = (m_prime - m_score) / self.dq
        return q_null

    def null_space_projection(self, q, q_vel, J, J_inv):
        q_null = self.compute_manipulability(q, J)
        identity = np.identity(self.kdl.get_dof())
        q_vel_constraint = (identity - J_inv @ J) @ q_null
        q_vel_proj = q_vel + q_vel_constraint
        return q_vel_proj

    def velocity_limit_projection(self, q_vel):
        limits = np.ones(self.kdl.get_dof()) * 0.2
        limit_check =np.max(np.absolute(q_vel) - limits)
        if limit_check > 0:
            idx = np.argmax(np.absolute(q_vel) - limits)
            alpha = limits[idx]/(np.absolute(q_vel)[idx])
            return alpha * q_vel
        return q_vel
        

    def plan(self, velocity, start=None, target=None, time=2, dt=0.1):
        if not isinstance(velocity, np.ndarray):
            velocity = np.array(velocity)
        times = np.arange(0.0, time, dt)
        q = start
        moveit_commander.roscpp_initialize(sys.argv)
        if start is None:
            move_group = moveit_commander.MoveGroupCommander(ARM_MOVE_GROUP_)
            q = move_group.get_current_joint_values()
        if target is not None:
            tposition = np.array(self.kdl.fk(q)) + velocity
            Ttarget = vector2mat(tposition[:3], target[3:]) #Target position based on linvel
        joint_pos = [q]
        joint_vel = [np.zeros(self.kdl.get_dof())]
        for t in times[1:]:
            if target is not None:
                curr = self.kdl.fk(q)
                #diff = euler_diff(curr[3:], target[3:])
                #omega = euler_to_angular_velocity(diff)
                Tcurr = vector2mat(curr[:3], curr[3:])
                vel = get_twist(Tcurr, Ttarget)
                vel = np.array([*vel])
                #rospy.loginfo(f'vel: {vel}')
                if np.all(np.abs(vel[3:]) < 1e-5):
                    rospy.loginfo('Target Orientation Achieved')
                    break
                velocity = vel
            J = self.jacobian(q)
            J_inv = self.damped_inv(J)
            q_vel = J_inv @ velocity
            q_vel = self.null_space_projection(q, q_vel, J, J_inv)
            q_vel = self.velocity_limit_projection(q_vel)
            d_q = q_vel * dt
            q = q + d_q
            joint_lows, joint_highs = self.kdl.get_joint_limits()
            joint_lows = np.array(joint_lows) * 0.95
            joint_highs = np.array(joint_highs) * 0.95
            if not np.all([q >= joint_lows, q<= joint_highs]):
                    rospy.loginfo('All motions stopped')
                    break
                #else:
                    #velocity[:3] = 0 # Stop linear motion & continue angular motion
                    #rospy.loginfo('Stopping Linear Motion')
            joint_pos.append(q)
            joint_vel.append(q_vel)
        jtraj = self.make_trajectory(joint_pos, joint_vel, times)
        dtraj = self.get_display_trajectory(jtraj)
        pub = rospy.Publisher('move_group/display_planned_path',\
                moveit_msgs.msg.DisplayTrajectory, queue_size = 20)
        rate = rospy.Rate(100)
        for i in range(100):
            if pub.get_num_connections() > 1:
                break
            pub.publish(dtraj)
            rate.sleep()
        return dtraj.trajectory[0]

    def plan_direct_joint_vel(self, q_vel, time=1, dt=0.1):
        times = np.arange(0.0, time, dt)
        moveit_commander.roscpp_initialize(sys.argv)
        move_group = moveit_commander.MoveGroupCommander(ARM_MOVE_GROUP_)
        q = move_group.get_current_joint_values()
        joint_pos = [q]
        joint_vel = [np.zeros(self.kdl.get_dof())]
        for t in times[1:]:
            J = self.jacobian(q)
            J_inv = self.damped_inv(J)
            #q_vel = self.null_space_projection(q, q_vel, J, J_inv)
            q_vel = self.velocity_limit_projection(q_vel)
            d_q = q_vel * dt
            q = q + d_q
            joint_lows, joint_highs = self.kdl.get_joint_limits()
            joint_lows = np.array(joint_lows) * 0.95
            joint_highs = np.array(joint_highs) * 0.95
            if not np.all([q >= joint_lows, q<= joint_highs]):
                    rospy.loginfo('All motions stopped')
                    break
            joint_pos.append(q)
            joint_vel.append(q_vel)
        jtraj = self.make_trajectory(joint_pos, joint_vel, times)
        dtraj = self.get_display_trajectory(jtraj)
        pub = rospy.Publisher('move_group/display_planned_path',\
                moveit_msgs.msg.DisplayTrajectory, queue_size = 20)
        rate = rospy.Rate(100)
        for i in range(100):
            if pub.get_num_connections() > 1:
                break
            pub.publish(dtraj)
            rate.sleep()
        return dtraj.trajectory[0]
        
            
    def make_trajectory(self, positions, velocities, times):
        jtraj = JointTrajectory()
        jtraj.header.frame_id = 'world'
        jtraj.joint_names = self.kdl.get_joint_names()
        for i in range(len(positions)):
            tpoint = JointTrajectoryPoint()
            tpoint.positions = positions[i]
            tpoint.velocities = velocities[i]
            tpoint.accelerations = velocities[i]
            tpoint.time_from_start = rospy.Duration.from_sec(times[i])
            if i:
                tpoint.accelerations -= velocities[i-1]
            jtraj.points.append(tpoint)
        return jtraj

    def get_display_trajectory(self, jtraj):
        dtraj = moveit_msgs.msg.DisplayTrajectory()
        dtraj.model_id = "iiwa"
        dtraj.trajectory_start.joint_state.name = self.kdl.get_joint_names()
        dtraj.trajectory_start.joint_state.position = jtraj.points[0].positions
        dtraj.trajectory_start.joint_state.velocity = jtraj.points[0].velocities
        dtraj.trajectory.append(moveit_msgs.msg.RobotTrajectory())
        dtraj.trajectory[0].joint_trajectory = jtraj
        return dtraj

def plan_to_pose_server(req):
    moveit_commander.roscpp_initialize(sys.argv)
    move_group = moveit_commander.MoveGroupCommander(ARM_MOVE_GROUP_)
    move_group.set_max_velocity_scaling_factor(MAX_VEL_FACTOR_)
    move_group.set_max_acceleration_scaling_factor(MAX_ACC_FACTOR_)
    move_group.set_planning_time(10)
    move_group.set_num_planning_attempts(3)
    move_group.clear_pose_targets()
    ik_solver = IK('world', 'reflex_palm_link', urdf_string=rospy.get_param(ROBOT_DESCRIPTION))
    seed_state = [0.0] * ik_solver.number_of_joints
    ik_js = ik_solver.get_ik(seed_state, req.pose.position.x, req.pose.position.y, req.pose.position.z,
                             req.pose.orientation.x, req.pose.orientation.y, req.pose.orientation.z, 
                             req.pose.orientation.w)
    #move_group.set_pose_target(req.pose)
    if req.constrained:
        constraints = get_orientation_constraint(move_group, req.target, req.tolerances)
        #constraints.position_constraints.append(get_position_constraint(move_group))
        move_group.set_path_constraints(constraints)
        move_group.set_planning_time(120)
        move_group.set_num_planning_attempts(12)
    move_group.set_joint_value_target(np.array(ik_js))
    plan = move_group.plan()
    rospy.loginfo("Found a plan")
    resp = Plan2PoseResponse()
    return _process_plan_response(plan, resp)

def get_IK(req):
    ik_solver = IK('world', 'reflex_palm_link', urdf_string=rospy.get_param(ROBOT_DESCRIPTION))
    seed_state = [0.0] * ik_solver.number_of_joints
    ik_js = ik_solver.get_ik(seed_state, req.pose.position.x, req.pose.position.y, req.pose.position.z,
                             req.pose.orientation.x, req.pose.orientation.y, req.pose.orientation.z, 
                             req.pose.orientation.w)
    resp = GetIKResponse()
    resp.success = False
    if ik_js is not None:
        resp.success = True
        resp.joint_angles = ik_js
    return resp

def _process_plan_response(plan, resp):
    resp.success = False
    if plan[0] and len(plan[1].joint_trajectory.points) > 0:
        resp.success = True
        resp.plan = plan[1]
        rospy.loginfo("Plan valid")
    return resp
    

def plan_to_joint_server(req):
    moveit_commander.roscpp_initialize(sys.argv)
    move_group = moveit_commander.MoveGroupCommander(ARM_MOVE_GROUP_)
    move_group.clear_pose_targets()
    move_group.set_max_velocity_scaling_factor(MAX_VEL_FACTOR_)
    move_group.set_max_acceleration_scaling_factor(MAX_ACC_FACTOR_)
    move_group.set_planning_time(30)
    move_group.set_num_planning_attempts(4)
    if req.constrained:
        #detach_object()
        #ik_solver = IK('world', 'iiwa_link_7')
        #seed_state = [0.0] * ik_solver.number_of_joints
        #curr_pose = move_group.get_current_pose().pose
        #ik_js = ik_solver.get_ik(seed_state, curr_pose.position.x, curr_pose.position.y, curr_pose.position.z, curr_pose.orientation.x, curr_pose.orientation.y, curr_pose.orientation.z, curr_pose.orientation.w)

        #curr_st = move_group.get_current_state()
        #rospy.loginfo(curr_st)
        #rospy.loginfo(ik_js)
        #curr_st.joint_state.position = list(ik_js) + [0.0] * 5
        #move_group.set_start_state(curr_st)
        
        constraints = get_orientation_constraint(move_group, req.target, req.tolerances)
        #constraints.position_constraints.append(get_position_constraint(move_group, req.tolerances))
        move_group.set_path_constraints(constraints)
        move_group.set_planning_time(60)
        move_group.set_num_planning_attempts(12)
    if req.rrt:
        move_group.set_planner_id("pRRT")
    rospy.loginfo(np.array(req.joint_target))
    move_group.set_joint_value_target(np.array(req.joint_target))
    plan = move_group.plan()
    rospy.loginfo("Found a plan")
    resp = Plan2JointResponse()
    return _process_plan_response(plan, resp)

def get_orientation_constraint(move_group, orientation, tols):
    oc = moveit_msgs.msg.OrientationConstraint()
    oc.header.frame_id = move_group.get_pose_reference_frame()
    oc.link_name = 'reflex_palm_link'#move_group.get_end_effector_link()
    oc.orientation = orientation#move_group.get_current_pose().pose.orientation
    oc.absolute_x_axis_tolerance = tols[0]
    oc.absolute_y_axis_tolerance = tols[1]
    oc.absolute_z_axis_tolerance = tols[2]
    oc.weight=1.0
    constraint = moveit_msgs.msg.Constraints()
    constraint.orientation_constraints.append(oc)
    return constraint

def get_position_constraint(move_group, tols):
    pc = moveit_msgs.msg.PositionConstraint()
    pc.header.frame_id = move_group.get_pose_reference_frame()
    curr_pose = move_group.get_current_pose()
    to_table = curr_pose.pose.position.z - 0.6
    pc.link_name = 'reflex_palm_link'
    box = shape_msgs.msg.SolidPrimitive()
    box.type = shape_msgs.msg.SolidPrimitive.BOX
    box.dimensions = [4, 4, 4]
    pc.constraint_region.primitives.append(box)
    box_pose = PoseStamped()
    up_id = np.argmax(tols)
    box_pose.pose.position.x = 0.0 #-(2-to_table)
    box_pose.pose.position.y = 0.0
    box_pose.pose.position.z = 2 + 0.6 #0.0 # 1.5/2# + 0.59
    rospy.loginfo(box_pose)
    box_pose.pose.orientation.w = 1.0
    pc.constraint_region.primitive_poses.append(box_pose.pose)
    pc.weight = 1.0
    return pc


def cartesian_planner(req):
    kdl = ManipulatorKDL()
    vel_planner = StraightLinePlanner(kdl)
    distance = np.array(req.distance)
    MAX_VEL = 0.1
    time = max(np.abs(distance/MAX_VEL))
    rospy.loginfo(f"commanding velocity: {distance/time} for time: {time} secs")
    plan = vel_planner.plan(distance/time, time=time)
    resp = CartPlanResponse()
    resp.success = True
    resp.plan = plan
    return resp

def lift_up_server(req):
    waypoints = []
    move_group = moveit_commander.MoveGroupCommander(ARM_MOVE_GROUP_)
    move_group.set_max_velocity_scaling_factor(MAX_VEL_FACTOR_)
    move_group.set_max_acceleration_scaling_factor(MAX_ACC_FACTOR_)
    start_pose = move_group.get_current_pose().pose
    rospy.loginfo("Current robot pose")
    rospy.loginfo(move_group.get_current_pose())
    #waypoints.append(copy.deepcopy(start_pose))
    Zs = np.arange(0, 0.25, 0.05)
    for z_inc in Zs:
        new_pose = copy.deepcopy(start_pose)
        new_pose.position.z += z_inc
        waypoints.append(new_pose)
    (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
    resp = LiftUpResponse()
    return _process_plan_response((True,plan), resp)

def go_home_server(req):
    JOINT_0_ = np.array([0.0] * 7)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    move_group = moveit_commander.MoveGroupCommander(ARM_MOVE_GROUP_)
    move_group.set_max_velocity_scaling_factor(MAX_VEL_FACTOR_)
    move_group.set_max_acceleration_scaling_factor(MAX_ACC_FACTOR_)
    move_group.set_joint_value_target(JOINT_0_)
    plan = move_group.plan()
    resp = LiftUpResponse()
    return _process_plan_response(plan, resp)

def execute_plan_server_real_iiwa(req):
    rate = 100
    iiwa = IiwaPlanner(rate)
    resp = ExecPlanResponse()
    if iiwa.wait_for_joint_state(timeout=2):
        try:
            proc_traj = ros_util.interpolate_joint_trajectory(req.plan.joint_trajectory, iiwa.rate.sleep_dur.to_sec())
            exec_res = iiwa.command_trajectory(proc_traj)
        except:
            rospy.logwarn('Execution failed, Proceed with caution')
            exec_res = False
        resp.success = exec_res
    else:
        resp.success = False
    return resp

def execute_plan_server_moveit(req):
    move_group = moveit_commander.MoveGroupCommander(ARM_MOVE_GROUP_)
    move_group.execute(req.plan, wait=True)
    move_group.stop()
    resp = ExecPlanResponse()
    resp.success = True
    return resp

def execute_plan_server(req):
    TrajInterface = robotTrajInterface(arm_prefix = '/iiwa', hand_prefix='/reflex', init_node=False)
    resp = ExecPlanResponse()
    resp.success = True
    if(req.plan is None) or (len(req.plan.joint_trajectory.points) <= 1):
        resp.success=False
    else:
        TrajInterface.send_jtraj(req.plan.joint_trajectory)
    return resp

def create_pnp_scene():
    scene = moveit_commander.PlanningSceneInterface()
    rospy.sleep(1)
    place_table_size = [0.65, 0.455, 0.755]
    pick_table_size = [0.9125, 0.61, 0.60]

    place_table_pose = PoseStamped()
    place_table_pose.header.frame_id = 'world'
    place_table_pose.pose.position.x = 0.0
    place_table_pose.pose.position.y = (pick_table_size[0] + place_table_size[1])/2#-(0.45625 + 0.61/2)
    place_table_pose.pose.position.z = place_table_size[2] / 2
    place_table_pose.pose.orientation.x = 0.0
    place_table_pose.pose.orientation.y = 0.0
    place_table_pose.pose.orientation.z = 0.0 # 0.7071068
    place_table_pose.pose.orientation.w = 1.0 # 0.7071068
    

    pick_table_pose = PoseStamped()
    pick_table_pose.header.frame_id = 'world'
    pick_table_pose.pose.position.x = 0.0 #0.88/2 + 0.61/2 + 0.1
    pick_table_pose.pose.position.y = -(0.45625 + 0.61/2)
    pick_table_pose.pose.position.z = 0.3
    pick_table_pose.pose.orientation.x = 0.0
    pick_table_pose.pose.orientation.y = 0.0
    pick_table_pose.pose.orientation.z = 0.0
    pick_table_pose.pose.orientation.w = 1.0
    #[0.88, 1.85, 0.74] #0.762]

    #no_fly_zone = [1.0, 1.4, 0.59]
    no_fly_zone = [0.1, 0.4, 0.59]
    no_fly_pose = PoseStamped()
    no_fly_pose.header.frame_id = 'world'
    no_fly_pose.pose.position.x = 0.0#-(0.305 + 0.3/2)
    no_fly_pose.pose.position.y = 0.0
    no_fly_pose.pose.position.z = 0.294916
    no_fly_pose.pose.orientation.w = 1.0
    
    scene.add_box('place_table', place_table_pose, place_table_size)
    scene.add_box('pick_table', pick_table_pose, pick_table_size)
    #scene.add_box('no_fly_zone', no_fly_pose, no_fly_zone)

def create_clutter_pnp_scene():
    scene = moveit_commander.PlanningSceneInterface()
    rospy.sleep(1)
    table_size = [0.75, 1.85, 0.735]

    table_pose = PoseStamped()
    table_pose.header.frame_id = 'world'
    table_pose.pose.position.x = 0.305 + table_size[0]/2
    table_pose.pose.position.y = 0.0
    table_pose.pose.position.z = table_size[2] / 2
    table_pose.pose.orientation.x = 0.0
    table_pose.pose.orientation.y = 0.0
    table_pose.pose.orientation.z = 0.0 # 0.7071068
    table_pose.pose.orientation.w = 1.0 # 0.7071068
    
    scene.add_box('foldable_table', table_pose, table_size)

def create_shelf_scene():
    scene = moveit_commander.PlanningSceneInterface()
    rospy.sleep(1)

    slab_size = [0.30, 0.86, 0.02]

    shelf1_pose = PoseStamped()

    shelf1_pose.header.frame_id = 'world'
    shelf1_pose.pose.position.x = 0.3 + 0.325 + 0.15
    shelf1_pose.pose.position.y = 0.0
    shelf1_pose.pose.position.z = 1.24
    shelf1_pose.pose.orientation.w = 1.0


    shelf2_pose = PoseStamped()

    shelf2_pose.header.frame_id = 'world'
    shelf2_pose.pose.position.x = 0.3 + 0.325 + 0.15
    shelf2_pose.pose.position.y = 0.0
    shelf2_pose.pose.position.z = 0.75
    shelf2_pose.pose.orientation.w = 1.0

    scene.add_box('shelf1', shelf1_pose, slab_size)

    scene.add_box('shelf2', shelf2_pose, slab_size)

    

def add_box_to_scene(name, pose_stamped, size):
    size[2] *= 0.9
    scene = moveit_commander.PlanningSceneInterface()
    scene.add_box(name, pose_stamped, size)

def add_mesh_to_scene(name,  pose_stamped, file_name):
    scene = moveit_commander.PlanningSceneInterface()
    rospy.loginfo('Sending to scene interface')
    scene.add_mesh(name, pose_stamped, file_name)
    rospy.loginfo('scene returned')

def add_object_meshes(env_dict):
    for key in env_dict.keys():
        VoxGrid = env_dict[key]
        VoxGrid.get_mesh(key+'.stl')
        rospy.loginfo(f'adding mesh {key}.stl....')
        pose = VoxGrid.get_pose()
        add_mesh_to_scene(key, pose, key+'.stl')
        rospy.loginfo(f'added')

def remove_scene_object(name):
    scene = moveit_commander.PlanningSceneInterface()
    scene.remove_world_object(name)

def remove_object_meshes(env_dict):
    for key in env_dict.keys():
        remove_scene_object(key)
        os.remove(key+'.stl')


def attach_object(pose_stamped, size):
    contact_links_ = ['reflex_distal_link_1', 'reflex_distal_pad_link_1', \
                      'reflex_distal_link_2', 'reflex_distal_pad_link_2', \
                      'reflex_distal_link_3', 'reflex_distal_pad_link_3', \
                      'reflex_proximal_link_1', 'reflex_proximal_pad_link_1', \
                      'reflex_proximal_link_2', 'reflex_proximal_pad_link_2', \
                      'reflex_proximal_link_3', 'reflex_proximal_pad_link_3', \
                      'reflex_flex_link_1', 'reflex_flex_link_2', 'reflex_flex_link_3', \
                      'reflex_palm_link', 'reflex_shell_link', 'reflex_pad_link', \
                      'reflex_swivel_link_1', 'reflex_swivel_link_2', 'pick_table']
    #The above list can be obtained from moveit, if we create a move_group for it.
    obj_name = 'grasp_obj'
    add_box_to_scene(obj_name, pose_stamped, size)
    scene = moveit_commander.PlanningSceneInterface()
    scene.attach_box('reflex_palm_link', obj_name, touch_links=contact_links_)

def detach_object():
    obj_name = 'grasp_obj'
    scene = moveit_commander.PlanningSceneInterface()
    scene.remove_attached_object('reflex_palm_link', name=obj_name)
    scene.remove_world_object(obj_name)

if __name__ == '__main__':
    ns = rospy.get_namespace()
    args = sys.argv + ['joint_states:='+ns+'iiwa/joint_states']
    moveit_commander.roscpp_initialize(args)
    rospy.init_node('pnp_moveit_server_node', anonymous=False)
    #create_pnp_scene()
    create_clutter_pnp_scene()
    #create_shelf_scene()
    js_manager = IIWA_Reflex_State()
    rate = rospy.Rate(100)
    env_yaml = rospy.get_param('env_yaml', None)
    env_dict = {}

    def shutdown():
        remove_object_meshes(env_dict)
    rospy.on_shutdown(shutdown)
    
    if env_yaml is not None:
        env_dict = load_env_yaml(env_yaml)
    rospy.Service('pnp_plan_to_pose_service', Plan2Pose, plan_to_pose_server)
    rospy.Service('pnp_plan_to_joint_service', Plan2Joint, plan_to_joint_server)
    rospy.Service('pnp_liftup_service', LiftUp, lift_up_server)
    rospy.Service('pnp_execute_plan_service', ExecPlan, execute_plan_server_real_iiwa)
    rospy.Service('pnp_go_home_plan_service', LiftUp, go_home_server)
    rospy.Service('pnp_get_IK_service', GetIK, get_IK)
    rospy.Service('pnp_cartesian_plan_service', CartPlan, cartesian_planner)
    rospy.loginfo('Services Advertised')
    add_object_meshes(env_dict)
    while not rospy.is_shutdown():
        js_manager.publish_merged_states()
        rate.sleep()
    rospy.spin()
