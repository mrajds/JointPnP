import os
import rospy
import pickle
import yaml
import numpy as np
from datetime import datetime
from gpu_sd_map.environment_manager import EnvVox, ObjGrid

SDM_PICKLES_DIR_ = "/home/mohanraj/obj_sdms"

OBJ_DATASET_DIR_ = "/home/mohanraj/ycb"

SESSIONS_DIR = "Sessions/"

INIT_DATA_DIR = "/home/mohanraj/pnp/init_data/"

def get_object_name(obj_str):
    Obj_Names = [x for x in os.listdir(OBJ_DATASET_DIR_) if \
                 os.path.isdir(os.path.join(OBJ_DATASET_DIR_, x))]
    Matches = [x for x in Obj_Names if obj_str in x]
    if len(Matches) > 0:
        return Matches[0]
    else:
        return np.random.choice(OBJ_NAMES_)
        
def save_problem(pblm, solret=None):
    pkl_f = SESSIONS_DIR + datetime.now().strftime("%h%d%y_%H%M.pkl")
    with open(pkl_f, 'wb') as f:
        data = (pblm.Env_List, pblm.Obj_Grid_raw, \
                pblm.obj_world_mat, pblm.x0, \
                pblm.get_pickle(), solret.get_pickle())
        pickle.dump(data, f) #Cannot pickle the entire class due to PyKDL
        #pickle.dump((), f)
        rospy.loginfo(f"Saved solved problem as {pkl_f}")

def restore_problem(pkl_f):
    with open(pkl_f, 'rb') as f:
        pblm_data = pickle.load(f)
    if isinstance(pblm_data[0], dict) and isinstance(pblm_data[1], ObjGrid):
        return pblm_data
    else:
        rospy.logwarn("Loaded problem is invalid, starting new session")
        return None

def load_sdm_pickle(obj_n):
    pkl_f = os.path.join(SDM_PICKLES_DIR_, obj_n+'.pkl')
    with open(pkl_f, 'rb') as f:
        sdm = pickle.load(f)
    if isinstance(sdm, ObjGrid):
        return sdm
    else:
        return None

def load_env_yaml(pkl_f):
    env_list = {}
    def get_pose_array(data, z):
        position = [data['position']['x'], \
                    data['position']['y'], \
                    z]
        orientation = [data['orientation']['x'], \
                       data['orientation']['y'], \
                       data['orientation']['z']]
        return position + orientation
    
    with open(pkl_f, 'r') as f:
        data = yaml.safe_load(f)
    obj_list = data['objects']
    rospy.loginfo(f'Loading environments with objects: \n{obj_list}')
    for obj_n in obj_list:
        obj_grid = load_sdm_pickle(obj_n)
        pose_raw = data[obj_n]
        pose_arr = get_pose_array(pose_raw, obj_grid.true_size[2]/2)
        print(pose_arr)
        obj_grid.set_pose(pose_arr, preserve=True)
        env_list[obj_n] = obj_grid
    return env_list

def write_exp(init_data):
    fn = datetime.now().strftime("%h%d%y_%H%M.yaml")
    with open(INIT_DATA_DIR+fn, 'w') as ymlf:
        yaml.dump(init_data, ymlf, default_flow_style=False)
    rospy.loginfo(f"Initializations saved as {fn}")

def read_exp(fn):
    rospy.loginfo(f"Reading: {fn}")
    with open(INIT_DATA_DIR+fn+'.yaml', 'r') as ymlf:
        data = yaml.safe_load(ymlf)
    return data
