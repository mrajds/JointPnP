#!/usr/bin/env python3

import rospy
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import matplotlib
#import pptk
import time
import os

from .gripper_augment import GripperVox, ReflexAug, visualize_points
from PIL import Image
from matplotlib import pyplot, cm
import pylab
from mpl_toolkits.mplot3d import Axes3D

FIGURE_DUMP_PATH_ = "/home/mohanraj/ll4ma_prime_WS/src/gpu_sd_map/Figures/"

class ConvNet(nn.Module):
    def __init__(self, Obj_shape):
        super(ConvNet, self).__init__()
        # In production version set these variables based on the object voxel size
        Z_env, _, O_w, O_h = Obj_shape
        ####################
        self.conv1 = torch.nn.Conv2d(Z_env, Z_env, (O_w, O_h), groups=Z_env, bias=False) #groups=Z_env eliminates summation of conv2d
        self.dilator3d = torch.nn.Conv3d(1, 1, 3, padding=1, bias=False)
        self.dilator = torch.nn.Conv2d(1, 1, (3,3), padding=(1,1), bias=False)
        self.conv_dx = torch.nn.Conv2d(1, 1, (1,3), padding=(0,1), bias=False)
        self.conv_dy = torch.nn.Conv2d(1, 1, (3,1), padding=(1,0), bias=False)
        self.cuda1 = torch.device('cuda')

    def forward(self, env, obj, contained=True):
        t_start = time.time()
        one = torch.tensor(1.0, device=self.cuda1, dtype=torch.half)
        w = torch.tensor(obj, device=self.cuda1, dtype=torch.half)
        x = torch.tensor(env, device=self.cuda1, dtype=torch.half)
        padding = (math.floor(w.shape[2]/2), math.ceil(w.shape[2]/2), \
                   math.floor(w.shape[3]/2), math.ceil(w.shape[3]/2))
        pv = 0
        if contained:
            pv = 1
        x = F.pad(x, padding, mode='constant', value=pv)
        x = 1 - x
        self.conv1.weight = nn.Parameter(w)
        x = x.unsqueeze_(0)
        x2 = self.conv1(x)
        w_sums = torch.sum(w, (2,3,1))
        inv_sums = 1.0/w_sums
        inv_sums = torch.where(torch.isinf(inv_sums), one, inv_sums)
        #x2[0] = x2[0] * inv_sums[:, None, None]
        t_sum = time.time() - t_start
        #rospy.loginfo("Time for Forward Pass: %f", t_sum)
        return x2[0].detach().cpu().numpy()

    def gen_3d_sd_map(self, vox):
        Tstart = time.time()
        w = torch.ones((1,1,3,3,3), device=self.cuda1, dtype=torch.half)
        x = torch.tensor(vox, device=self.cuda1, dtype=torch.half)
        x = x.unsqueeze_(0)
        x = x.unsqueeze_(0)
        #print("Vox Grid size: {}".format(x.element_size() * x.nelement()))
        k = 0
        j = -1
        #x = 1 - x
        x = x.half()
        sdm = torch.zeros(vox.shape, device=self.cuda1)
        curr_pts = x[0][0].nonzero(as_tuple = False)
        sdm[tuple(curr_pts.T)] = 0.0
        tresh = 1.0/9.0
        self.dilator3d.weight = nn.Parameter(w)
        TP1 = time.time() - Tstart
        #rospy.loginfo(f"Initialized in {TP1} sec(s)")
        DTime = 0
        PTime = 0
        while not torch.all(x == 1.0):
            Dst = time.time()
            x_new = self.dilator3d(x.half())
            Den = time.time()
            x_new = torch.where(x_new < tresh, 0.0, 1.0)
            x_diff = x_new - x
            x_diff = torch.where(x_diff < tresh, 0.0, 1.0)
            Fen = time.time()
            #new_pts = x_diff[0][0].nonzero(as_tuple = False)
            x_diff = x_diff.bool()
            sdm[x_diff[0][0]] = k
            Hen = time.time()
            #sdm[tuple(new_pts.T)] = k
            x = x_new
            k += 1
            #Fen = time.time()
            DTime += Den - Dst
            PTime += Hen - Fen
        #rospy.loginfo(f"Full Dilate at {k} iters")
        x = torch.tensor(vox, device=self.cuda1, dtype=torch.half)
        x = x.unsqueeze_(0)
        x = x.unsqueeze_(0)
        x = 1 - x
        x = x.half()
        while not torch.all(x == 1.0):
            Dst = time.time()
            x_new = self.dilator3d(x.half())
            Den = time.time()
            x_new = torch.where(x_new < tresh, 0.0, 1.0)
            x_diff = x_new - x
            x_diff = torch.where(x_diff < tresh, 0.0, 1.0)
            Fen = time.time()
            #new_pts = x_diff[0][0].nonzero(as_tuple = False)
            x_diff = x_diff.bool()
            sdm[x_diff[0][0]] = j
            Hen = time.time()
            #sdm[tuple(new_pts.T)] = j
            x = x_new
            j -= 1
            #Fen = time.time()
            DTime += Den - Dst
            PTime += Hen - Fen
        #rospy.loginfo(f"Full Dilate at {j} iters")
        TTot = time.time() - Tstart
        #rospy.loginfo(f"Generated 3D-SDf Map in {TTot} sec(s)")
        #rospy.loginfo(f"Total {DTime}s for Dilate, Avg {DTime/(k-j)}s per iter")
        #rospy.loginfo(f"Total {PTime}s for Processing, Avg {PTime/(k-j)}s per iter")
        return sdm, j, k
    
    def gen_sd_map(self, img):
        Tstart = time.time()
        w = torch.ones((1,1,3,3), device=self.cuda1, dtype=torch.half)
        x = img.clone()
        nx = img.clone()
        nx = nx.half()
        x = x.unsqueeze_(0)
        x = x.unsqueeze_(0)
        nx = nx.unsqueeze_(0)
        nx = nx.unsqueeze_(0)
        x = 1 - x
        x = x.half()
        torch.cuda.synchronize()
        k = 0
        j = -1 
        sdm = torch.zeros(img.shape, device=self.cuda1)
        curr_pts = x[0][0].nonzero(as_tuple = False)
        #print(curr_pts)
        sdm[tuple(curr_pts.T)] = 0.0
        tresh = 1.0/9.0
        self.dilator.weight = nn.Parameter(w)
        while not torch.all(x == 1.0):
            x_new = self.dilator(x.half())
            x_new = torch.where(x_new < tresh, 0.0, 1.0)
            #x_new[x_new < tresh] = 0.0
            #x_new[x_new >= tresh] = 1.0
            x_diff = x_new - x
            x_diff = torch.where(x_diff < tresh, 0.0, 1.0)
            #x_diff[x_diff < tresh] = 0.0
            #x_diff[x_diff >= tresh] = 1.0
            new_pts = x_diff[0][0].nonzero(as_tuple = False)
            sdm[tuple(new_pts.T)] = k
            x = x_new
            k += 1
        while not torch.all(nx == 1.0):
            nx_new = self.dilator(nx.half())
            #nx_new[nx_new < tresh] = 0.0
            #nx_new[nx_new >= tresh] = 1.0
            nx_new = torch.where(nx_new < tresh, 0.0, 1.0)
            nx_diff = nx_new - nx
            #nx_diff[nx_diff < tresh] = 0.0
            #nx_diff[nx_diff >= tresh] = 1.0
            nx_diff = torch.where(nx_diff < tresh, 0.0, 1.0)
            new_pts = nx_diff[0][0].nonzero(as_tuple = False)
            sdm[tuple(new_pts.T)] = j
            nx = nx_new
            j -= 1
        #rospy.loginfo(f"Full Dilate at {k, j} iters")
        write_shaded_image(sdm, j, k)
        TTot = time.time() - Tstart
        #rospy.loginfo(f"Generated SDf Map in {TTot} sec(s)")
        return sdm

    def clean_2d(self, img):
        w = torch.ones((1,1,3,3), device=self.cuda1, dtype=torch.half)
        x = 1 - torch.from_numpy(img).to(self.cuda1)
        x = x.unsqueeze_(0)
        x = x.unsqueeze_(0)
        self.dilator.weight =nn.Parameter(w)
        x = self.dilator(x.half())
        x = self.dilator(1-x)
        #x = 1-x
        return x[0][0].detach().cpu().numpy()
    
    def gen_dsdf_x(self, sdm):
        Tstart = time.time()
        w = torch.zeros((1,1,1,3), device=self.cuda1, dtype=torch.float)
        x = sdm.clone()
        x = x.unsqueeze_(0)
        x = x.unsqueeze_(0)
        w[0][0][0][0] = -1.0
        w[0][0][0][2] = 1.0
        with torch.no_grad():
            self.conv_dx.weight = nn.Parameter(w)
            dx = self.conv_dx(x)
            dx /= 2.0
        write_shaded_image(dx[0][0], None, None, "Dx_Out", cm.seismic)
        return dx

    def gen_dsdf_y(self, sdm):
        Tstart = time.time()
        w = torch.zeros((1,1,3,1), device=self.cuda1, dtype=torch.float)
        y = sdm.clone()
        y = y.unsqueeze_(0)
        y = y.unsqueeze_(0)
        w[0][0][0] = -1.0
        w[0][0][2] = 1.0
        with torch.no_grad():
            self.conv_dy.weight = nn.Parameter(w)
            dy = self.conv_dy(y)
            dy /= 2.0
        write_shaded_image(dy[0][0], None, None, "Dy_Out", cm.seismic)
        return dy

def write_shaded_image(arrin, l=None, h=None, out="Out", cmap = None, show=True):
    if torch.is_tensor(arrin):
        arr = arrin.detach().cpu().numpy().astype(np.float64)
    else:
        arr = arrin.astype(np.float64)
    if (l is None) or (h is None):
        l = np.min(arr)
        h = np.max(arr)
    Mn = arr < 0
    Mp = arr > 0
    arr[arr == 0.0] = 0.5
    if l != 0:
        arr[Mn] -= l
        arr[Mn] *= 0.25/float(-l)
    if h != 0:
        arr[Mp] *= 0.25/float(h)
        arr[Mp] += 0.75
    if(cmap is None):
        cmap = cm.rainbow
    img = Image.fromarray(np.uint8(cmap(arr)*255))
    img = img.convert("RGB")
    if show:
        img.show()
    img.save(FIGURE_DUMP_PATH_ + out +".png")
    
def read_voxel(FileName):
    with open(FileName, newline='') as f:
        data = list(csv.reader(f, delimiter=' '))
    data = np.array(data, dtype='int')
    return data

def convert_to_dense_voxel_grid(sparse_voxel_grid, one_axis_length, scale=1):
    voxel_grid = np.zeros((one_axis_length*scale, one_axis_length*scale, one_axis_length*scale))
    nr_voxels = sparse_voxel_grid.shape[0]
    for i in range(nr_voxels):
        position = sparse_voxel_grid[i,:]
        position *= scale
        position_high = position + scale
        voxel_grid[position[0]:position_high[0], position[1]:position_high[1], position[2]:position_high[2]] = 1.
    return voxel_grid

def Voxel_Format_zxy(voxelGrid):
    Zdim = voxelGrid.shape[2]
    NewGrid = np.zeros([Zdim, 1, voxelGrid.shape[0], voxelGrid.shape[1]])
    for i in range(Zdim):
        NewGrid[i, 0] = voxelGrid[:,:,i]
    assert NewGrid.shape == (Zdim, 1, voxelGrid.shape[0], voxelGrid.shape[1])
    return NewGrid

def mid_squeeze(Voxel):
    ret = Voxel[:,0,:,:]
    #print("squeezing......")
    return ret

def voxel_to_pcd(voxel, tresh = 0.0, render = False):
    points = np.argwhere(voxel>tresh)
    #rospy.loginfo("Total selected points: %s", points.shape)
    if render:
        #v = pptk.viewer(points)
        #v.set(point_size=0.1)
        raise NotImplementedError("Module pptk is not compatible with current python version. Need to update alternate renderer")
    #v.wait()
    return points

def crop(voxel):
    points = voxel_to_pcd(voxel)
    x_min = min(points[:,0])
    x_max = max(points[:,0])
    y_min = min(points[:,1])
    y_max = max(points[:,1])
    z_min = min(points[:,2])
    z_max = max(points[:,2])
    cropped = voxel[x_min:x_max, y_min:y_max, z_min:z_max]
    #rospy.loginfo("Bounds: %f %f %f", x_max-x_min, y_max-y_min, z_max-z_min)
    return cropped

def get_bounds(voxel):
    points = voxel_to_pcd(voxel)
    mins = []
    maxs = []
    mins.append(min(points[:,0]))
    maxs.append(max(points[:,0]))
    mins.append(min(points[:,1]))
    maxs.append(max(points[:,1]))
    mins.append(min(points[:,2]))
    maxs.append(max(points[:,2]))
    return mins, maxs

def update_Env(Env, Obj, x, y, z=0):
    x_end = x + Obj.shape[2]
    y_end = y + Obj.shape[3]
    z_end = z + Obj.shape[0]
    Env[z:z_end, x:x_end, y:y_end] = Obj[:,0,:,:]
    return Env

def flatten_vox(voxel, fn="Conv", show=True, axis = 0, treshold=True):
    #rospy.loginfo("Flattening image")
    #voxel = np.einsum('zxy->xyz', voxel)
    if treshold:
        if torch.is_tensor(voxel):
            voxel = torch.where(voxel > 0.999, 1.0, 0.0)
        else:
    #print(len(voxel[voxel>0.999]))
            voxel[voxel < 0.999] = 0.0
            voxel[voxel > 0.999] = 1.0
    #rospy.loginfo("Filters Done")
    ret = voxel.prod(axis)
    #return ret
    if(torch.is_tensor(ret)):
        flat_arr = ret.cpu().detach().numpy().astype(np.uint8)
    else:
        flat_arr = ret.astype(np.uint8)
    #print(flat_arr)
    img = Image.fromarray(flat_arr*255)
    img = img.convert("L")
    if show:
        img.show()
    img.save(FIGURE_DUMP_PATH_ + fn +".png")
    #print("saved at: "+ FIGURE_DUMP_PATH_ + fn +".png" )
    return ret

def Generate_Gripper_Kernel():
    Pkg_Path = "/home/mohanraj/ll4ma_prime_WS/src/gpu_sd_map"
    Base_BV_Path = Pkg_Path + "/gripper_augmentation/base.binvox" #Use rospack find to generate the full path
    Finger_BV_Path = Pkg_Path + "/gripper_augmentation/finger.binvox"
    Base = GripperVox(Base_BV_Path)
    Finger1 = GripperVox(Finger_BV_Path)
    Finger2 = GripperVox(Finger1)
    Base.print_metadata()
    Finger2.print_metadata()
    Gripper = ReflexAug(Base, Finger1)
    Grid = Gripper.Set_Pose_Preshape(rpy=[0.0, 0.0, 0.0])
    #print(Grid.shape)
    return Grid

def close_grid(vox):
    with torch.no_grad():
        tresh = 1.0/9.0
        cuda1 = torch.device('cuda')
        w = torch.ones((1,1,3,3,3), device=cuda1, dtype=torch.half)
        x = torch.tensor(vox, device=cuda1, dtype=torch.half)
        x = x.unsqueeze_(0)
        x = x.unsqueeze_(0)
        dilator3d = torch.nn.Conv3d(1, 1, 3, padding=1, bias=False)
        dilator3d.weight = nn.Parameter(w)
        bnds = np.array(get_bounds(vox))
        iter_lim = min(bnds[:][1]-bnds[:][0]) // 3
        for i in range(iter_lim):
            x = dilator3d(x.half())
            x = torch.where(x < tresh, 0.0, 1.0)
        x = 1 - x
        for i in range(iter_lim - 3 ):
            x = dilator3d(x.half())
            x = torch.where(x < tresh, 0.0, 1.0)
    return 1 - x

def gen_3d_sd_map(vox):
    with torch.no_grad():
        Tstart = time.time()
        cuda1 = torch.device('cuda')
        w = torch.ones((1,1,3,3,3), device=cuda1, dtype=torch.half)
        x = torch.tensor(vox, device=cuda1, dtype=torch.half)
        #x[0][:][:] = 1
        #x[:, 0 ,:] = 1
        #x[:, : ,0] = 1
        #x[-1][:][:] = 1
        #x[:, -1 ,:] = 1
        #x[:, :, -1] = 1
        #flatten_vox(1-x, "st_bf")
        x = x.unsqueeze_(0)
        x = x.unsqueeze_(0)
        #rospy.loginfo("Vox Grid size: {}".format(x.element_size() * x.nelement()))
        k = 0
        j = -1
        #x = 1 - x
        x = x.half()
        sdm = torch.zeros(vox.shape, device=cuda1)
        mid_idx = torch.tensor(torch.tensor(vox.shape)/2, device=cuda1).long()
        curr_pts = x[0][0].nonzero(as_tuple = False)
        sdm[tuple(curr_pts.T)] = 0.0
        tresh = 1.0/9.0
        dilator3d = torch.nn.Conv3d(1, 1, 3, padding=1, bias=False)
        dilator3d.weight = nn.Parameter(w)
        TP1 = time.time() - Tstart
        #rospy.loginfo(f"Initialized in {TP1} sec(s)")
        DTime = 0
        PTime = 0
        while not torch.all(x == 1.0):
            #print("Current k - {}".format(k))
            Dst = time.time()
            x_new = dilator3d(x.half())
            Den = time.time()
            x_new = torch.where(x_new < tresh, 0.0, 1.0)
            x_diff = x_new - x
            x_diff = torch.where(x_diff < tresh, 0.0, 1.0)
            Fen = time.time()
            #new_pts = x_diff[0][0].nonzero(as_tuple = False)
            x_diff = x_diff.bool()
            sdm[x_diff[0][0]] = k
            Hen = time.time()
            #sdm[tuple(new_pts.T)] = k
            #if k < 10 or k % 10 == 0:
            #flatten_vox(1-x[0][0], "bf"+str(k))
            x = x_new
            k += 1
            #Fen = time.time()
            DTime += Den - Dst
            PTime += Hen - Fen
        #rospy.loginfo(f"Full Dilate at {k} iters")
        x = torch.tensor(vox, device=cuda1, dtype=torch.half)
        x = x.unsqueeze_(0)
        x = x.unsqueeze_(0)
        x = 1 - x
        x = x.half()
        while not torch.all(x == 1.0):
            Dst = time.time()
            x_new = dilator3d(x.half())
            Den = time.time()
            x_new = torch.where(x_new < tresh, 0.0, 1.0)
            x_diff = x_new - x
            x_diff = torch.where(x_diff < tresh, 0.0, 1.0)
            Fen = time.time()
            #new_pts = x_diff[0][0].nonzero(as_tuple = False)
            x_diff = x_diff.bool()
            sdm[x_diff[0][0]] = j
            Hen = time.time()
            #sdm[tuple(new_pts.T)] = j
            x = x_new
            j -= 1
            #Fen = time.time()
            DTime += Den - Dst
            PTime += Hen - Fen
        face_mids = [[0,mid_idx[1],mid_idx[2]], \
                     [-1,mid_idx[1],mid_idx[2]], \
                     [mid_idx[0],0,mid_idx[2]], \
                     [mid_idx[0],-1,mid_idx[2]], \
                     [mid_idx[0],mid_idx[1],0],\
                     [mid_idx[0],mid_idx[1],-1]]
        trunc_val = torch.tensor(float("Inf"), device=cuda1)
        #for idx in face_mids:
        trunc_val = torch.min(torch.min(sdm[0]),trunc_val)
        trunc_val = torch.min(torch.min(sdm[-1]),trunc_val)
        trunc_val = torch.min(torch.min(sdm[:,0]),trunc_val)
        trunc_val = torch.min(torch.min(sdm[:,-1]),trunc_val)
        trunc_val = torch.min(torch.min(sdm[:,:,0]),trunc_val)
        trunc_val = torch.min(torch.min(sdm[:,:,-1]),trunc_val)
        sdm = torch.minimum(sdm,trunc_val)
        #write_shaded_image(sdm[mid_idx[0]])
        #rospy.loginfo(f"Full Dilate at {j} iters")
        TTot = time.time() - Tstart
        #rospy.loginfo(f"Generated 3D-SDf Map in {TTot} sec(s)")
        #rospy.loginfo(f"Total {DTime}s for Dilate, Avg {DTime/(k-j)}s per iter")
        #rospy.loginfo(f"Total {PTime}s for Processing, Avg {PTime/(k-j)}s per iter")
        #rospy.loginfo(f"Shape of sdf {sdm.shape}")
        dx_sdm = gen_dsdf_x(sdm)
        #rospy.loginfo(f"Shape of dsdf_x {dx_sdm.shape}")
        #write_shaded_image(dx_sdm[mid_idx[0]])
        dy_sdm = gen_dsdf_y(sdm)
        #rospy.loginfo(f"Shape of dsdf_y {dy_sdm.shape}")
        dz_sdm = gen_dsdf_z(sdm)
        #rospy.loginfo(f"Shape of dsdf_z {dz_sdm.shape}")
    #sdm_np = sdm.cpu().detach().numpy().astype(int)
    #d_sdm = torch.stack([dx_sdm,dy_sdm,dz_sdm]).cpu().detach().numpy()
    sdfg = torch.stack([sdm,dx_sdm,dy_sdm,dz_sdm], axis = -1).cpu().detach().numpy()
    del sdm, dx_sdm, dy_sdm, dz_sdm
    del x
    del x_new
    del x_diff
    torch.cuda.empty_cache()
    return sdfg
    #return sdm_np, d_sdm

def min_from_sdm3d(sdm3d, axis = 0):
    return np.min(sdm3d, axis) #Tuple of min_vals and min_idx, only min_val needed for now

def gen_dsdf_x(sdm, pre=""):
    Tstart = time.time()
    conv_dx = torch.nn.Conv2d(1, 1, (3,1), padding=(1,0), padding_mode='reflect', bias=False)
    cuda1 = torch.device('cuda')
    w = torch.zeros((1,1,3,1), device=cuda1, dtype=torch.float)
    x = sdm.clone()
    x = x.unsqueeze_(1)
    #x = x.unsqueeze_(0)
    w[0][0][0] = -1.0
    w[0][0][2] = 1.0
    with torch.no_grad():
        conv_dx.weight = nn.Parameter(w)
        dx = conv_dx(x)
        dx /= 2.0
    #write_shaded_image(dx[0][0], None, None, pre+"Dx_Out", cm.seismic)
    return mid_squeeze(dx)

def gen_dsdf_y(sdm, pre=""):
    Tstart = time.time()
    conv_dy = torch.nn.Conv2d(1, 1, (1,3), padding=(0,1), padding_mode='reflect', bias=False)
    cuda1 = torch.device('cuda')
    w = torch.zeros((1,1,1,3), device=cuda1, dtype=torch.float)
    y = sdm.clone()
    y = y.unsqueeze_(1)
    #y = y.unsqueeze_(0)
    w[0][0][0][0] = -1.0
    w[0][0][0][2] = 1.0
    with torch.no_grad():
        conv_dy.weight = nn.Parameter(w)
        dy = conv_dy(y)
        dy /= 2.0
    #write_shaded_image(dy[0][0], None, None, pre+"Dy_Out", cm.seismic)
    return mid_squeeze(dy)

def gen_dsdf_z(sdm):
    z = sdm.clone()
    z = z.permute(1,2,0)
    z = z.unsqueeze_(1)
    conv_dz = torch.nn.Conv2d(1, 1, (1,3), padding=(0,1), padding_mode='reflect', bias=False)
    cuda1 = torch.device('cuda')
    w = torch.zeros((1,1,1,3), device=cuda1, dtype=torch.float)
    w[0][0][0][0] = -1.0
    w[0][0][0][2] = 1.0
    with torch.no_grad():
        conv_dz.weight = nn.Parameter(w)
        dz = conv_dz(z)
        dz /= 2.0
    return mid_squeeze(dz).permute(2,0,1)

def gen_3d_radial_kernel(radius=10):
    cuda1 = torch.device('cuda')
    x = torch.arange(radius * 2 + 1, device=cuda1, dtype=torch.half)
    y = torch.arange(radius * 2 + 1, device=cuda1, dtype=torch.half)
    z = torch.arange(radius * 2 + 1, device=cuda1, dtype=torch.half)
    m = torch.meshgrid(x,y,z)
    r = torch.sqrt((m[0] - radius)**2 + (m[1] - radius)**2 + (m[2] - radius)**2)
    return torch.tensor(r, device=cuda1, dtype=torch.int)

def gen_one_shot_3d_tsdm(vox, trunc_dist=5):
    cuda1 = torch.device('cuda')
    torch.cuda.empty_cache()
    with torch.no_grad():
        print("start")
        print(torch.cuda.memory_allocated(cuda1))
        w = gen_3d_radial_kernel(trunc_dist).detach() + 1
        print("w created")
        print(torch.cuda.memory_allocated(cuda1))
        x = torch.tensor(torch.where(vox==0., float("Inf"), 1.0), dtype=torch.int32).detach()
        print("x created")
        print(torch.cuda.memory_allocated(cuda1))
        x = F.pad(x, (trunc_dist, trunc_dist) * 3)
        print("x padded")
        print(torch.cuda.memory_allocated(cuda1))
        x = x.unfold(0,w.shape[0],1).unfold(1,w.shape[1],1).unfold(2,w.shape[2],1).detach()
        print("x unfolded")
        print(x.element_size() * x.nelement())
        #print(x.view(-1).shape)
        garb = x.reshape(x.nelement())
        sdm = x * w
        sdm = sdm.reshape(vox.shape + (-1,)).min(-1)
        x = torch.where(vox==1., float("Inf"), 1.0)
        x = F.pad(x, (trunc_dist, trunc_dist) * 3)
        x = x.unfold(0,trunc_dist,1).unfold(1,trunc_dist,1).unfold(2,trunc_dist,1)
        sdm -= (x * w).reshape(vox.shape + (-1,)).min(-1)

        #UF = torch.nn.Unfold(w.shape, padding=trunc_dist)
        #x = x.unsqueeze(0)
        #x = x.unsqueeze(0)
        #x = UF(x)
        #sdm = (x.T * w.reshape(-1,1))[0,:].min(1)
        #x = torch.where(vox==1., float("Inf"), 1.0)
        #x = x.unsqueeze(0)
        #x = x.unsqueeze(0)
        #x = UF(x)
        #sdm -= (x.T * w.reshape(-1,1))[0,:].min(1)
        #sdm = sdm.reshape(vox.shape)
        dx_sdm = gen_dsdf_x(sdm)
        #rospy.loginfo(f"Shape of dsdf_x {dx_sdm.shape}")
        write_shaded_image(dx_sdm[mid_idx[0]])
        dy_sdm = gen_dsdf_y(sdm)
        #rospy.loginfo(f"Shape of dsdf_y {dy_sdm.shape}")
        dz_sdm = gen_dsdf_z(sdm)
        #rospy.loginfo(f"Shape of dsdf_z {dz_sdm.shape}")
        sdm_np = sdm.cpu().detach().numpy().astype(int)
        d_sdm = torch.stack([dx_sdm,dy_sdm,dz_sdm]).cpu().detach().numpy()
    return sdm_np, d_sdm 

if __name__ == '__main__':
    rospy.init_node('gpu_sdm_node')
    totT = 0
    fstart = time.time()
    Obj_Scale = 3

    # Voxel 1
    sparse_vox = read_voxel('vox3.dat')
    full_vox1 = convert_to_dense_voxel_grid(sparse_vox, 64, Obj_Scale)
    full_vox1 = crop(full_vox1)
    full_vox1 = Voxel_Format_zxy(full_vox1)
    envox = np.zeros([400,1000,800]) #Created in format zxy
    #envox = update_Env(envox, full_vox1, 100, 100)
    #envox = update_Env(envox, full_vox1, 200, 50)
    #envox = update_Env(envox, full_vox1, 800, 650)

    #Voxel 2
    sparse_vox = read_voxel('vox2.dat')
    full_vox = convert_to_dense_voxel_grid(sparse_vox, 64, Obj_Scale)
    full_vox = crop(full_vox)
    #voxel_to_pcd(full_vox, render = True)
    full_vox = Voxel_Format_zxy(full_vox)
    #envox = np.zeros([64,100,100]) #Created in format zxy
    envox = update_Env(envox, full_vox, 50, 50)
    envox = update_Env(envox, full_vox, 50, 700)
    envox = update_Env(envox, full_vox, 500, 50)
    envox = update_Env(envox, full_vox, 500, 500)
    envox = update_Env(envox, full_vox, 70, 350)
    envox = update_Env(envox, full_vox, 160, 350)

    sparse_vox = read_voxel('vox1.dat')
    full_vox3 = convert_to_dense_voxel_grid(sparse_vox, 64, Obj_Scale)
    full_vox3 = crop(full_vox3)
    full_vox3 = Voxel_Format_zxy(full_vox3)
    envox = update_Env(envox, full_vox3, 900, 750)
    #envox = update_Env(envox, full_vox3, 950, 50)
    envox = update_Env(envox, full_vox3, 700, 400)
    envox = update_Env(envox, full_vox3, 250, 600)
    envox = update_Env(envox, full_vox3, 300, 660)
    envox = update_Env(envox, full_vox3, 250, 660)
    envox = update_Env(envox, full_vox3, 300, 600)

    emins, emaxs = get_bounds(envox)
    
    GripperVox = Generate_Gripper_Kernel()
    Gmins, Gmaxs = get_bounds(GripperVox)
    
    GripperVox = Voxel_Format_zxy(GripperVox)
    ub = min(emaxs[0],Gmaxs[2]-Gmins[2])
    flatten_vox(mid_squeeze(1-GripperVox[Gmins[2]:Gmins[2]+ub]), "GFlat")
    testvox = GripperVox[Gmins[2]:Gmins[2]+ub]
    #visualize_points(voxel_to_pcd(mid_squeeze(testvox)))
    #exit(0)
    
    voxel_to_pcd(envox, render = False)
    start = time.time()
    #envoxT = torch.tensor(envox)
    flatten_vox(1 - envox, "Flat")
    
    convr = ConvNet(full_vox.shape)
    T_tocuda = time.time()
    GPU0 = torch.device("cuda:0")
    convr.to(GPU0)
    sdm3d, j, k = convr.gen_3d_sd_map(envox[0:full_vox.shape[0]])
    convr.gen_sd_map(torch.tensor(flatten_vox(envox[0:full_vox.shape[0]]), device=convr.cuda1))
    for i in range(sdm3d.shape[0]):
        if (i % 5) == 0:
            write_shaded_image(sdm3d[i], j, k, out='3ds'+str(i))
    exit(1)
    
    #rospy.loginfo("Init took: %f", time.time() - T_tocuda)
    envoxT = convr.forward(envox[0:full_vox.shape[0]], full_vox)
    #torch.cuda.synchronize()
    conv_map = flatten_vox(envoxT)
    print("Getting Jac ", envoxT.shape, convr.conv1.weight.shape)
    jac1 = torch.autograd.grad(envoxT, convr.conv1.weight, torch.ones_like(envoxT))
    print("Jac obtained: ", len(jac1), jac1[0].shape)
    #exit(1)
    sdm = convr.gen_sd_map(conv_map)
    dx = convr.gen_dsdf_x(sdm)
    dy = convr.gen_dsdf_y(sdm)
    dmag = torch.sqrt(torch.square(dx) + torch.square(dy))
    write_shaded_image(dmag[0][0], None, None, "MAG", cm.seismic)
    #rospy.loginfo("Test point out")

    #Gripper Augtest
    ub = min(emaxs[0],Gmaxs[2]-Gmins[2])
    convaug = ConvNet((ub, GripperVox.shape[1], GripperVox.shape[2], GripperVox.shape[3]))
    convaug.to(GPU0)
    envoxA = convaug.forward(envox[0:ub], GripperVox[Gmins[2]:Gmins[2]+ub], False)
    #rospy.loginfo("Sync Test")
    #torch.cuda.synchronize()
    #visualize_points(voxel_to_pcd((1-envoxA).cpu().detach().numpy()))
    conv_map = flatten_vox(envoxA)
    #rospy.loginfo("Flatten Out")
    sdm = convr.gen_sd_map(conv_map)
    dx = convr.gen_dsdf_x(sdm)
    dy = convr.gen_dsdf_y(sdm)
    exit(1)
    envoxNumpy = (envoxT).cpu().detach().numpy()
    #rospy.loginfo("Operation Time: %f", time.time() - start)
    voxel_to_pcd(envoxNumpy, 0.99999, render=True)
    totT = time.time() - fstart
    #rospy.loginfo("Time taken: %f", totT)
