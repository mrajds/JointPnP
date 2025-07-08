#!/usr/bin/env python
import rospy
import sys
import tf
import tf2_ros
import tf2_geometry_msgs
import trimesh
import math
import numpy as np
import tf_conversions.posemath as pm
from std_srvs.srv import Empty, Trigger, TriggerResponse
from tf import TransformerROS
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply
from geometry_msgs.msg import Pose, WrenchStamped, Quaternion, PoseStamped
from sensor_msgs.msg import JointState, PointCloud2
from point_cloud_segmentation.srv import *
from ll4ma_opt_utils.srv import UpdateWorld, AddObject
from gazebo_msgs.srv import GetModelState,SpawnModel
from grasp_pipeline.srv import *
from prob_grasp_planner.srv import GraspVoxelInfer, GraspVoxelInferRequest
from ll4ma_robot_control.srv import TaskPosition
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rospkg import RosPack
from sklearn.decomposition import PCA
rp=RosPack()
rp.list()
path=rp.get_path('ll4ma_planner')+'/scripts'
sys.path.insert(0,path)
path=rp.get_path('ll4ma_opt_utils')+'/scripts'
sys.path.insert(0,path)
from plan_client import *
from robot_class import robotInterface

class ll4ma_pick_n_place:
    def __init__(self, sim = True):
        self.sim = sim
        self.contact = False
        self.object_name = '003_cracker_box'
        self.robot_base_link = 'lbr4_base_link'
        self.ee_frame = 'pad'
        self.object_config_frame = self.object_name + "_0"
        self.object_mesh_folder = '/home/mohanraj/ycb/'
        self.object_mesh_path = self.getObjMeshPath()
        self.object_mesh = trimesh.load(self.object_mesh_path)
        self.ri=robotInterface(init_node=False)
        self.pub=rospy.Publisher('/lbr4/joint_cmd',JointState,queue_size=1)
        self.taskpub = rospy.Publisher('/lbr4/task_position_cmd',PoseStamped,queue_size=1)
        self.object_frame = self.object_name+'__'+self.object_name+'_link'
        self.loop_rate=rospy.Rate(100)
        self.reflex_pub = rospy.Publisher('/reflex/joint_cmd',JointState,queue_size=1)
        self.currJS = JointState()
        self.reflexJS = JointState()
        rospy.Subscriber('/ft_sensor_neg/wrench',WrenchStamped,self.contactCB)
        #rospy.wait_for_service('/lbr4/task_position_service')   ### Need this later
        self.TaskPositionCall = rospy.ServiceProxy('/lbr4/task_position_service', TaskPosition)
        rospy.Subscriber('/lbr4/joint_states',JointState,self.stateCB)
        rospy.Subscriber('/reflex/joint_states', JointState, self.reflexStateCB)
        rospy.wait_for_service('/object_segmenter')
        self.object_seg_call = rospy.ServiceProxy('/object_segmenter', SegmentGraspObject)
        rospy.wait_for_service('/gen_grasp_preshape')
        self.gen_heuristic_grasp = rospy.ServiceProxy('/gen_grasp_preshape', GraspPreshape)
        rospy.wait_for_service('/reflex_grasp_interface/open_hand')
        self.hand_open_call = rospy.ServiceProxy("/reflex_grasp_interface/open_hand", Trigger)
        rospy.wait_for_service("/reflex_grasp_interface/grasp")
        self.hand_grasp_call = rospy.ServiceProxy("/reflex_grasp_interface/grasp", Trigger)
        rospy.wait_for_service("/reflex_grasp_interface/release")
        self.hand_release_call = rospy.ServiceProxy("/reflex_grasp_interface/release", Trigger)

    def reconfigure(self, obj_name, obj_id):
        self.object_name = obj_name
        self.object_config_frame = self.object_name + "_" + str(obj_id)
        self.object_mesh_path = self.getObjMeshPath()
        self.object_mesh = trimesh.load(self.object_mesh_path)
        self.object_frame = self.object_name+'__'+self.object_name+'_link'

    def getCollisionMeshPath(self):
        object_mesh_path = self.object_mesh_folder + '/' + self.object_name + \
                           '/' + 'google_16k' + '/nontextured_proc.stl'
        return object_mesh_path
        
    def getObjMeshPath(self):
        object_mesh_path = self.object_mesh_folder + '/' + self.object_name + \
                           '/' + 'google_16k' + '/nontextured.stl'
        return object_mesh_path

    def get_grasp_yaw(self):
        major_axis = self.get_major_axis()
        CosT = np.dot(np.array(major_axis),np.array([1,0]))
        SinT = np.dot(np.array(major_axis),np.array([0,1]))
        oT = -math.atan2(SinT,CosT)
        print(oT*180/3.14)
        #oT*=0
        tf_b = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tf_b)
        source_frame = self.robot_base_link
        target_frame = self.object_frame
        Trans = tf_b.lookup_transform(target_frame,source_frame,rospy.Time(0),rospy.Duration(5.0))
        QuatArray = np.array([Trans.transform.rotation.x,Trans.transform.rotation.y,Trans.transform.rotation.z,Trans.transform.rotation.w])
        (r,p,y) = euler_from_quaternion(QuatArray)
        oT += y
        #NOTE: The Frame of reference for reflex is perfectly aligned with gripper x as minor-axis and object x as major-axis for other grippers this may not be the case and add GripperOffset accordingly.
        return oT
    
    def get_major_axis(self):
        mesh = self.object_mesh
        pts = np.array(mesh.vertices)
        pca = PCA(n_components=2)
        pca.fit(pts[:,:2])
        return pca.components_[0]

    def get_object_mass_and_inertia(self):
        self.object_mesh.density = 10
        return self.object_mesh.mass, self.object_mesh.moment_inertia

    def GenURDFString(self):
        object_pose = [0.0]*6;
        collision_mesh = self.getCollisionMeshPath()
        object_mass, object_inertia = self.get_object_mass_and_inertia()
        object_rpy = str(object_pose[0]) + ' ' + str(object_pose[1]) + ' ' + str(object_pose[2])
        object_location = str(object_pose[3]) + ' ' + str(object_pose[4]) + ' ' + str(object_pose[5])
        urdf_str = """
        <robot name=\"""" + self.object_name + """\">
        <link name=\"""" + self.object_name + """_link">
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
        <mesh filename=\"file://""" + self.object_mesh_path + """\" />
        </geometry>
        </visual>
        <collision>
        <origin xyz=\"""" + str(object_location) +"""\"  rpy=\"""" + str(object_rpy) +"""\"/>
        <geometry>
        <mesh filename=\"file://""" + collision_mesh + """\" />
        </geometry>
        </collision>
        </link>
        <gazebo reference=\"""" + self.object_name + """_link\">
        <mu1>10.0</mu1>
        <maxVel>0.0</maxVel>
        <minDepth>0.003</minDepth>
        <material>Gazebo/Red</material>
        </gazebo>
        </robot>
        """
        return urdf_str

    def get_mesh_dims(self):
        return self.object_mesh.bounds

    def contactCB(self, ft_dat):
        forces = ft_dat.wrench.force
        force_vec = np.array([forces.x, forces.y, forces.z])
        rosprint("Force Mag: " + str(np.linalg.norm(force_vec)))
        #rosprint(str(forces.x)+"-x\n"+str(forces.y)+"-y\n"+str(forces.z)+"-z\n")
        if((abs(ft_dat.wrench.force.x)>20)or(abs(ft_dat.wrench.force.y)>20)or(abs(ft_dat.wrench.force.z)>12)):
            self.contact = True
        else:
            self.contact = False

    def objectsegCB(self, dat):
        self.object_cloud = dat
        self.object_sense = True
        return
    
    def tablesegCB(self, dat):
        self.table_cloud = dat
        self.table_sense = True
        return

    def broadcast_ee_pose(self, GraspPose):
        br = tf.TransformBroadcaster()
        for i in range(10):
            positiontuple = (GraspPose.position.x, GraspPose.position.y, GraspPose.position.z)
            quattuple = (GraspPose.orientation.x, \
                         GraspPose.orientation.y, \
                         GraspPose.orientation.z, \
                         GraspPose.orientation.w)
            br.sendTransform(positiontuple,quattuple,rospy.Time.now(),"PlannedEEPose",self.robot_base_link)
            rospy.sleep(0.1)

    def get_ee_top_grasp_pose(self):
        meshBB = self.get_mesh_dims()
        centroid = meshBB[1][:]/2 + meshBB[0][:]/2
        OffsetPose = Pose()
        OffsetPose.position.x = centroid[0]
        OffsetPose.position.y = centroid[1]
        OffsetPose.position.z = meshBB[1][2] + 0.30
        source_frame = self.object_frame
        target_frame = self.robot_base_link
        OffsetPose = TransformPose(OffsetPose,target_frame,source_frame)
        yaw = self.get_grasp_yaw()
        print('yaw_angle='+str(yaw*180/3.14))
        GraspPose = OffsetPose
        GraspQuat = quaternion_from_euler(-1.57,0,-yaw)
        GraspPose.orientation.x=GraspQuat[0]
        GraspPose.orientation.y=GraspQuat[1]
        GraspPose.orientation.z=GraspQuat[2]
        GraspPose.orientation.w=GraspQuat[3]
        return GraspPose

    def PlanTraj(self, GraspPose, ObjPose):
        g_plan=ll4maPlanner('/lbr4_reflex')
        ri=robotInterface(init_node=False)
        rate=rospy.Rate(10)
        while(not g_plan.got_state):
            rate.sleep()
        robot_js=JointState()
        robot_js.name=['lbr4_j0','lbr4_j1','lbr4_j2','lbr4_j3','lbr4_j4','lbr4_j5','lbr4_j6']
        robot_js.position=g_plan.joint_state.position[0:7]
        hand_js=JointState()
        hand_js.name=['preshape_1','preshape_2', 'proximal_joint_1',
                      'proximal_joint_2','proximal_joint_3']
        hand_js.position=g_plan.joint_state.position[7:]
        hand_preshape=copy.deepcopy(hand_js)
        robot_traj, palm_pose=g_plan.get_palm_traj(robot_js,hand_js, \
                            hand_preshape,GraspPose,JointTrajectory(),ObjPose,T=3)
        robot_traj = ri.get_smooth_traj(robot_traj)
        g_plan.viz_traj(robot_traj,t_name='ll4ma_planner/traj')
        return robot_traj
    
    def Approach_Pose(self, GraspPose, ObjPose):
        meshBB = self.get_mesh_dims()
        centroid = meshBB[1][:]/2 + meshBB[0][:]/2
        OffsetPose = Pose()
        OffsetPose.position.x = centroid[0]
        OffsetPose.position.y = centroid[1]
        OffsetPose.position.z = meshBB[1][2]
        source_frame = self.object_frame
        target_frame = self.robot_base_link
        OffsetPose = TransformPose(OffsetPose,target_frame,source_frame)
        GraspPose.position = OffsetPose.position
        #GraspPose.position.y = -GraspPose.position.y
        self.broadcast_ee_pose(GraspPose)
        j_traj = self.PlanTraj(GraspPose, ObjPose)
        return j_traj

    def Lift_Object(self, GraspPose, ObjPose):
        GraspPose.position.z += 0.2
        self.broadcast_ee_pose(GraspPose)
        j_traj = self.PlanTraj(GraspPose, ObjPose)
        return j_traj

    def stop_exec(self, jc):
        for i in range(10):
            jc.velocity *= 0
            jc.effort *= 0
            self.pub.publish(jc)
            self.loop_rate.sleep()
        return

    def execute_traj(self, j_traj, check_contact=False):
        new_jc=JointState()
        for i in range(len(j_traj.points)):
            if self.contact and check_contact:
                rosprint("Gripper in Contact")
                self.stop_exec(new_jc)
                break
            new_jc=JointState()
            new_jc.name=j_traj.joint_names
            new_jc.position=j_traj.points[i].positions
            new_jc.velocity=j_traj.points[i].velocities
            new_jc.effort=j_traj.points[i].accelerations
            self.pub.publish(new_jc)
            #print i
            self.loop_rate.sleep() 
        return

    def genUpDownTraj(self, z):
        resp = self.straight_rel_z(z)
        ri=robotInterface(init_node=False)
        j_traj = JointTrajectory()
        FirstPt = JointTrajectoryPoint()
        FirstPt.positions = self.currJS.position
        j_traj.points.append(FirstPt)
        j_traj.points = j_traj.points + resp.plan_traj.points
        j_traj.points[-1].time_from_start.secs=1
        smooth_traj = ri.get_smooth_traj(j_traj)
        #print(smooth_traj)
        return resp.plan_traj
        
    def stateCB(self, JS):
        self.currJS = JS

    def reflexStateCB(self, JS):
        self.reflexJS = JS

    def command_reflex_joints(self, jcmd):
        ctr = 0
        while ctr<100:
            self.reflex_pub.publish(jcmd)
            self.loop_rate.sleep()
            ctr+=1

    def command_reflex(self,Close=True):
        jcmd = JointState()
        jcmd.name = ['preshape_1','proximal_joint_1','preshape_2',
                     'proximal_joint_2','proximal_joint_3']
        jcmd.velocity = [0.0]*5
        jcmd.effort = [0.0]*5
        if Close:
            jcmd.position=np.array([0.0,0.0,3.0,3.0,3.0])
        else:
            jcmd.position=np.array([0.0,0.0,1.3,1.3,1.3])*0.
        ctr = 0
        while ctr<100:
            self.reflex_pub.publish(jcmd)
            self.loop_rate.sleep()
            ctr+=1
        #jcmd = self.reflexJS
        #jcmd.position += np.array([0.0,0.0,0.3,0.3,0.3])
        #while ctr<200:
            #self.reflex_pub.publish(jcmd)
            #self.loop_rate.sleep()
            #ctr+=1
            
    def get_object_pose(self):
        TrROS = TransformerROS()
        tf_b = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tf_b)
        source_frame = self.robot_base_link
        target_frame = self.object_config_frame
        print(self.object_config_frame)
        B_Trans_O = tf_b.lookup_transform(source_frame,target_frame,rospy.Time(0),rospy.Duration(5.0))
        trans = [B_Trans_O.transform.translation.x,B_Trans_O.transform.translation.y,B_Trans_O.transform.translation.z]
        rot = [B_Trans_O.transform.rotation.x,B_Trans_O.transform.rotation.y,B_Trans_O.transform.rotation.z,B_Trans_O.transform.rotation.w]
        B_Trans_O_Mat = TrROS.fromTranslationRotation(trans, rot)
        source_frame = self.ee_frame
        target_frame = self.object_frame
        print(self.object_frame)
        O_Trans_E = tf_b.lookup_transform(target_frame,source_frame,rospy.Time(0),rospy.Duration(5.0))
        trans = [O_Trans_E.transform.translation.x,O_Trans_E.transform.translation.y,O_Trans_E.transform.translation.z]
        rot = [O_Trans_E.transform.rotation.x,O_Trans_E.transform.rotation.y,O_Trans_E.transform.rotation.z,O_Trans_E.transform.rotation.w]
        O_Trans_E_Mat = TrROS.fromTranslationRotation(trans, rot)
        B_Trans_E_Mat = np.matmul(B_Trans_O_Mat, O_Trans_E_Mat)
        print(B_Trans_E_Mat.shape)
        retpose = Mat_to_Pose(B_Trans_E_Mat)
        retpose.position.z += 0.1
        self.broadcast_ee_pose(retpose)
        return retpose
        
def TransformPose(inpose,target_frame='lbr4_base_link',source_frame='world'):
    tf_b = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_b)
    inpose_stamped = tf2_geometry_msgs.PoseStamped()
    inpose_stamped.pose = inpose
    inpose_stamped.header.frame_id = source_frame
    inpose_stamped.header.stamp = rospy.Time(0)

    opose = tf_b.transform(inpose_stamped,target_frame,rospy.Duration(5))
    return opose.pose

def Mat_to_Pose(Mat):
    q = quaternion_from_matrix(Mat)
    pose = Pose(position=Point(*Mat[:3,3]),orientation=Quaternion(*q))
    return pose

def RecordTraj(Traj):
    f = open("SampleTraj.txt","w")
    for i in range(len(Traj.points)):
        f.write(str(Traj.points[i].positions))
        f.write("\n")
        f.write(str(Traj.points[i].velocities))
        f.write("\n")
        f.write(str(Traj.points[i].accelerations))
        f.write("\n")
    f.close()
    return
        

def GzPoseToTfPose(GzPose):
    retPose = Pose()
    retPose.position.x = float(GzPose.position.x)
    retPose.position.y = float(GzPose.position.y)
    retPose.position.z = float(GzPose.position.z)
    retPose.orientation.x = float(GzPose.orientation.x)
    retPose.orientation.y = float(GzPose.orientation.y)
    retPose.orientation.z = float(GzPose.orientation.z)
    retPose.orientation.w = float(GzPose.orientation.w)
    return retPose

def getObjNames(ObjFile):
    f = open(ObjFile,"r")
    text = f.read()
    lines = text.splitlines()
    return lines

def GraspVoxelInfClient(segObj):
    rospy.loginfo('Waiting for service grasp_voxel_infer')
    rospy.wait_for_service('grasp_voxel_infer')
    VoxelInf = rospy.ServiceProxy('grasp_voxel_infer', GraspVoxelInfer)
    rospy.loginfo('Requesting Service...')
    req = GraspVoxelInferRequest()
    req.seg_obj = segObj
    req.prior_name = 'MDN'
    req.grasp_type = 'overhead'
    res = VoxelInf(req)
    palm_pose = res.full_inf_config.palm_pose
    preshape_js = res.full_inf_config.hand_joint_state
    rospy.loginfo('Received Palm Pose: ')
    rospy.loginfo(palm_pose)
    rospy.loginfo('Received Preshape: ')
    rospy.loginfo(preshape_js)
    palm_pose_base_link = TransformPose(palm_pose.pose, source_frame=palm_pose.header.frame_id)
    return palm_pose_base_link, None

if __name__ == '__main__':
    simmode = True
    if(len(sys.argv) != 2):
        rosprint('Error: Input Object Sequence File')
        exit(1)
    ObjList=getObjNames(sys.argv[1])

    rospy.init_node('pick_n_place_main')
    pnp = ll4ma_pick_n_place(sim=simmode)
    rosprint = rospy.loginfo
    rosprint('Initialized LL4MA Pick and Place Pipeline')
    rospy.Subscriber('/object_cloud',PointCloud2,pnp.objectsegCB)
    rospy.Subscriber('/table_cloud',PointCloud2,pnp.tablesegCB)
    if simmode:
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        GetGazeboModelPos = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        GzSpawnObj = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        rosprint('Gazebo is running. Waiting for ll4ma_planner')
        
    rospy.wait_for_service('/ll4ma_planner/update_world')
    UpdatePlannerEnv = rospy.ServiceProxy('/ll4ma_planner/update_world', UpdateWorld)
    AddObject_ToPlanner = rospy.ServiceProxy('/ll4ma_planner/add_object', AddObject)
    rosprint('LL4MA planner is running. Updating Environment')
    EnvPoses = []
    ObjPose = Pose()
    if simmode:
        GzModelState = GetGazeboModelPos('object_table','world')
        EnvPoses.append(TransformPose(GzPoseToTfPose(GzModelState.pose)))
        GzModelState = GetGazeboModelPos('pick_table','world')
        EnvPoses.append(TransformPose(GzPoseToTfPose(GzModelState.pose)))
    else:
        # Add Real Env Segmentation
        pass
    
    if(UpdatePlannerEnv(ObjPose,EnvPoses)):
        rosprint('Planner Environment Updated')
        
    for i,obj_name in enumerate(ObjList):
        pnp.reconfigure(obj_name,i)

        if simmode:
            obj_urdf = pnp.GenURDFString()
            SpawnPose = Pose()
            SpawnPose.position.x = -0.1
            SpawnPose.position.y = -0.1
            SpawnPose.position.z = 1.0
            SpawnQuat = quaternion_from_euler(0,0,np.random.uniform(0,3.14))
            SpawnPose.orientation.x=SpawnQuat[0]
            SpawnPose.orientation.y=SpawnQuat[1]
            SpawnPose.orientation.z=SpawnQuat[2]
            SpawnPose.orientation.w=SpawnQuat[3]
            GzSpawnObj(pnp.object_name,obj_urdf,'Mutable_Objs',SpawnPose,'table_link')
            rosprint('Spawnned Test Object In Gazebo')
            rospy.sleep(2)
            segreq = SegmentGraspObjectRequest()
            segres = pnp.object_seg_call(segreq)
            if segres.object_found:
                rosprint('There is a response')
            else:
                rosprint('ERROR!!! Segmentation failed aborting')
                exit(0)
            #while(not(pnp.object_sense and pnp.table_sense)):
                #rosprint('Looking for Object, Is the camera on??')
                #rospy.sleep(2)
            rosprint('Object Point Cloud Segmented')
            #PreShapeReq = GraspPreshapeRequest()
            #PreShapeReq.obj = segres.obj
            #PreShapeRes = pnp.gen_heuristic_grasp(PreShapeReq)
            GzModelState = GetGazeboModelPos(pnp.object_name,'world')
            ObjPose = TransformPose(GzPoseToTfPose(GzModelState.pose))
            
            #rosprint('Found '+ str(len(PreShapeRes.palm_goal_pose_world)) + ' Preshapes')
            ##rand_id =np.random.randint(0,len(PreShapeRes.palm_goal_pose_world))
            #rosprint('Randomly Choosing Grasp ' + str(rand_id) + " of " + str(len(PreShapeRes.palm_goal_pose_world)))
            #if(PreShapeRes.is_top_grasp[rand_id]):
                #rosprint('Top Grasp Selected')
            GraspPose, Preshape = GraspVoxelInfClient(segres.obj)
            if(GraspPose is None):
                rospy.loginfo("Error Voxel Client incomplete (Server responded, need to transform and plan)")
                exit(1)
        else:
            # Add Real Obj Segmentation
            pass
        
        if(AddObject_ToPlanner(pnp.getObjMeshPath(),Pose(),ObjPose)):
            rosprint('Added Object Mesh to Planner')
            rosprint('Tracked Object')
            #GraspPose = pnp.get_ee_top_grasp_pose()
            #GraspPoseStamped = PreShapeRes.palm_goal_pose_world[rand_id]
            ##GraspPose = TransformPose(GraspPoseStamped.pose)
            pnp.broadcast_ee_pose(GraspPose)
            Exec=False
            SomePose = Pose()
            SomePose.position.x = 0
            SomePose.position.y = 0
            SomePose.position.z = -1

            #if simmode:
                #pnp.command_reflex_joints(PreShapeRes.allegro_joint_state[rand_id])
                #pnp.command_reflex(False)
            #else:
            pnp.hand_open_call()
    
            AddObject_ToPlanner(pnp.getObjMeshPath(),Pose(),SomePose)
            while(not Exec):
                robot_traj = pnp.PlanTraj(GraspPose,SomePose)
                RecordTraj(robot_traj)
                #Uip = raw_input('Execute? y/n')
                #rosprint(robot_traj)
                Uip ='y'
                if (str(Uip) == str('s')) or (str(Uip) == str('y')):
                    pnp.execute_traj(robot_traj, False)
                    #ri.send_jtraj(robot_traj)
                    Exec=True
                #raw_input('Continue?')
                #while(not pnp.contact):
            ReachPose = copy.deepcopy(GraspPose)
            #ReachPose.position.z -= 0.2
            ReachCmd = PoseStamped()
            ReachCmd.header.stamp = rospy.Time.now()
            ReachCmd.header.frame_id = 'lbr4_base_link'
            ReachCmd.pose = ReachPose
            #pnp.TaskPositionCall(ReachCmd)
            #if simmode:
                #pnp.command_reflex()
            #else:
            pnp.hand_grasp_call()
                
            #ReachCmd.pose.position.x += 0.2
            ReachCmd.pose.position.z += 0.005
            pnp.TaskPositionCall(ReachCmd)
            #Uip = raw_input('Break off Here for Task Space Test')
            #down_traj = pnp.genUpDownTraj(-0.2)
            #pnp.execute_traj(down_traj, True)
            #down_traj = pnp.genUpDownTraj(0.01)
            #pnp.execute_traj(down_traj, False)
            #pnp.command_reflex()
            #down_traj = pnp.genUpDownTraj(0.5)
            #pnp.execute_traj(down_traj, False)
            MidPose = copy.deepcopy(GraspPose)
            MidPose.position.x = -0.5
            MidPose.position.y = 0
            MidPose.position.z = 0.5
            robot_traj = pnp.PlanTraj(MidPose,SomePose)
            pnp.execute_traj(robot_traj, False)
            PlacePose = GraspPose #pnp.get_object_pose()
            PlacePose.position.x += np.random.uniform(-0.05, 0.05)
            PlacePose.position.z += 0.05
            RandRot = quaternion_from_euler(0., 0., np.random.uniform(0.0, 3.14))
            RotQuat = [PlacePose.orientation.x, PlacePose.orientation.y,
                                        PlacePose.orientation.z, PlacePose.orientation.w]
            PlaceQuat = quaternion_multiply(RandRot,  RotQuat) #quaternion_from_euler()
            PlacePose.orientation.x, PlacePose.orientation.y, \
                PlacePose.orientation.z, PlacePose.orientation.w = PlaceQuat
            pnp.broadcast_ee_pose(PlacePose)
            robot_traj = pnp.PlanTraj(PlacePose,SomePose)
            pnp.execute_traj(robot_traj, False)

            #if simmode:
                #pnp.command_reflex(False)
            #else:
            pnp.hand_open_call()

            MidCmd = PoseStamped()
            MidCmd.header.stamp = rospy.Time.now()
            MidCmd.header.frame_id = 'lbr4_base_link'
            MidCmd.pose = MidPose
            
            pnp.TaskPositionCall(MidCmd)
            #up_traj = pnp.genUpDownTraj(0.1)
            #pnp.execute_traj(up_traj, False)
            #robot_traj = pnp.PlanTraj(MidPose,SomePose)
            #pnp.execute_traj(robot_traj, False)
            #if(AddObject_ToPlanner(pnp.getObjMeshPath(),Pose(),SomePose)):
            #robot_traj = pnp.Approach_Pose(GraspPose,SomePose)
            #raw_input('Continue?')
            #pnp.execute_traj(robot_traj, True)
            #pnp.command_reflex()
            MidPose.position.x *= -1 
            robot_traj = pnp.Lift_Object(MidPose,SomePose)
            #raw_input('Continue?')
            pnp.execute_traj(robot_traj)
            #if(AddObject_ToPlanner(pnp.getObjMeshPath(),Pose(),ObjPose)):
    rosprint('Execution Successful')
            #pnp.command_reflex(False)
