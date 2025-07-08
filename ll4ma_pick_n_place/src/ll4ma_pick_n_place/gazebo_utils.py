import trimesh
import rospy
import numpy as np
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler

OBJ_DATASET_FOLDER_ = "/home/mohanraj/ycb" #Set this to the local location of YCB dataeset

def gen_urdf_string(obj_name):
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
    urdf_str = \
    """
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

def spawn_model_client(obj_name):
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    GzSpawnObj= rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
    pick_table_height = 0.762
    Obj_urdf, obj_bounds = gen_urdf_string(obj_name)
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
