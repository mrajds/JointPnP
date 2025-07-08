#!/usr/bin/env python3

import rospy
import pickle
from ll4ma_pick_n_place.segmentation_client import SegmentationClient, get_objgrid

EXT = '.pkl'
SAVE_DIR = '/home/mohanraj/obj_sdms/'

if __name__ == '__main__':
    rospy.init_node('objgrid_pickler_node')
    obj_grid = get_objgrid(True)
    obj_grid.gen_sd_map()
    uip = input('Input object name: ')
    uip = uip.lower()
    if uip == 'q':
        rospy.loginfo('Cancelled')
        rospy.signal_shutdown('User aborted execution')
        exit(0)
    if not uip.endswith(EXT):
        uip = uip + EXT
        
    pkl_f = SAVE_DIR + uip
    rospy.loginfo(f'Saving file: {pkl_f}')
    with open(pkl_f, 'wb') as f:
        pickle.dump(obj_grid, f)
