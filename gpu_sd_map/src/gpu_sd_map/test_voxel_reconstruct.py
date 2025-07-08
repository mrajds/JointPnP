#!/usr/bin/env python
#import rospy
import sys
import pypcd
import csv
import roslib.packages as rp
#from rospkg import RosPack
#rp=RosPack()
#rp.list()
sys.path.append(rp.get_pkg_dir("pointsdf_reconstruction") + "/src")
sys.path.append(rp.get_pkg_dir("pointsdf_reconstruction") + "/src/PointConv")
sys.path.append(rp.get_pkg_dir("prob_grasp_planner") + "/src/grasp_common_library")
from api import MeshReconstructor
from voxel_visualization_tools import visualize_voxel, convert_to_sparse_voxel_grid, convert_to_dense_voxel_grid
from visualization import plot_voxel


PCD_DIR="/home/mohanraj/ll4ma_prime_WS/src/ll4ma_3d_reconstruction/test_data/pcd/Depth/"
PCD_FILE="006_mustard_bottle_40.pcd"
MODEL_PATH="/home/mohanraj/ll4ma_prime_WS/src/ll4ma_3d_reconstruction/full_model"

def write_voxels(FileName, voxels):
    with open(FileName, 'w') as f:
        csv.writer(f, delimiter=' ').writerows(voxels)

if __name__ == '__main__':
    reconstructor = MeshReconstructor(MODEL_PATH)

    try:
        cloud = pypcd.PointCloud.from_path(PCD_DIR + PCD_FILE)
    except IOError:
        print("File, " + str(PCD_DIR + PCD_FILE) + " doesn't exist.")
        exit(1)
    voxels = reconstructor.reconstruct(cloud)
    sparse_voxel_grid = convert_to_sparse_voxel_grid(voxels)
    #dense_voxel_grid = convert_to_dense_voxel_grid(sparse_voxel_grid, 512)
    print(sparse_voxel_grid)
    visualize_voxel(voxels, as_cubes=True, img_path="test.png")
    write_voxels("vox3.dat", sparse_voxel_grid)
    print("Voxels written to file")
    #raw_input()
