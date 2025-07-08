#!/usr/bin/env python
'''
Adapter for gpu_sd_map functions to constraints and gradients.

TODO:
- This might be tricky as the gpu_sd_map library uses conda venv.
  So do this as service calls may be?
'''

def create_sdm_env_client(env_name, dims, resolution = 1000, world_trans = Pose()):
    '''
    Client to gpu_sdm_create_env service

    TODO:
    - move to dedicated library
    '''
    rospy.wait_for_service('/gpu_sdm_create_env')
    InitEnvVox = rospy.ServiceProxy('/gpu_sdm_create_env', CreateEnv)

    InitEnvReq = CreateEnvRequest()
    InitEnvReq.env_name = env_name
    InitEnvReq.dimension = dims
    InitEnvReq.resolution = resolution
    InitEnvReq.world_trans = world_trans
    if not InitEnvVox(InitEnvReq).success:
        rospy.logfatal("Couldn't create environment, Exitting")
        rospy.signal_shutdown("No Environment Buffer to Work on")


def add_sdm_object_request(env_name, OBJ_NAME_, obj_points, obj_dims, place_conf):
    Add2EnvReq = AddObj2EnvRequest()
    Add2EnvReq.env_name = env_name
    Add2EnvReq.obj_id = OBJ_NAME_
    Add2EnvReq.sparse_obj_grid = obj_points
    Add2EnvReq.obj_size = obj_dims
    Add2EnvReq.place_conf = place_conf
    return Add2EnvReq

def add_sdm_object_client(env_name, OBJ_NAME_, obj_points, obj_dims, place_conf):
    '''
    Client to add objects to sdm env

    TODO:
    - move to dedicated library
    '''
    rospy.wait_for_service('/gpu_sdm_add_obj')
    Add2EnvVox = rospy.ServiceProxy('/gpu_sdm_add_obj', AddObj2Env)

    Add2EnvReq = AddObj2EnvRequest()
    Add2EnvReq.env_name = env_name
    Add2EnvReq.obj_id = OBJ_NAME_
    Add2EnvReq.sparse_obj_grid = obj_points
    Add2EnvReq.obj_size = obj_dims
    Add2EnvReq.place_conf = place_conf
    if(Add2EnvVox(Add2EnvReq)):
        rospy.loginfo("Object Added Successfully")
