#!/usr/bin/env python3

from ll4ma_pick_n_place.data_utils import load_env_yaml
from ll4ma_pick_n_place.opt_problem import publish_env_cloud
import rospy
import os
import sys
from rospkg import RosPack

rp = RosPack()
pkg_path = rp.get_path('ll4ma_pick_n_place')

ENVS_DIR = os.path.join(pkg_path, 'envs')


if __name__ == '__main__':
    rospy.init_node('scene_viz_node')
    ENV_PATH = os.path.join(ENVS_DIR, sys.argv[1]+'.yaml')
    env_list = load_env_yaml(ENV_PATH)
    publish_env_cloud(env_list)
    
