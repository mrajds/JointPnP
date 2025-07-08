#!/usr/bin/env python

import rospy
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import matplotlib
import pptk
import time

from gripper_augment import GripperVox, ReflexAug, visualize_points
from PIL import Image
from matplotlib import pyplot, cm
import pylab
from mpl_toolkits.mplot3d import Axes3D

class ConvNet(nn.Module):
    def __init__(self, Obj_shape):
        super(ConvNet, self).__init__()
        # In production version set these variables based on the object voxel size
        Z_env, _, O_w, O_h = Obj_shape
        ####################
        self.conv1 = torch.nn.Conv2d(Z_env, Z_env, (O_w, O_h), groups=Z_env, bias=False) #groups=Z_env eliminates summation of conv2d
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
        x2[0] = x2[0] * inv_sums[:, None, None]
        t_sum = time.time() - t_start
        rospy.loginfo("Time for Forward Pass: %f", t_sum)
        return x2[0]

    def gen_sd_map(self, img):
        #Tstart = time.time()
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
        rospy.loginfo(f"Full Dilate at {k} iters")
        write_shaded_image(sdm, j, k)
        #TTot = time.time() - Tstart
        #rospy.loginfo(f"Generated SDf Map in {TTot} sec(s)")
        return sdm

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

def write_shaded_image(arrin, l, h, out="Out", cmap = None):
    if torch.is_tensor(arrin):
        arr = arrin.detach().cpu().numpy().astype(np.float)
    else:
        arr = arrin.astype(np.float)
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
    img.save(out+".png")
    
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
    print("squeezing......")
    return ret

def voxel_to_pcd(voxel, tresh = 0.0, render = False):
    points = np.argwhere(voxel>tresh)
    rospy.loginfo("Total selected points: %s", points.shape)
    if render:
        v = pptk.viewer(points)
        v.set(point_size=0.1)
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
    rospy.loginfo("Bounds: %f %f %f", x_max-x_min, y_max-y_min, z_max-z_min)
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

def flatten_vox(voxel, fn="Conv"):
    rospy.loginfo("Flattening image")
    if torch.is_tensor(voxel):
        voxel = torch.where(voxel > 0.999, 1.0, 0.0)
    else:
    #print(len(voxel[voxel>0.999]))
        voxel[voxel < 0.999] = 0.0
        voxel[voxel > 0.999] = 1.0
    rospy.loginfo("Filters Done")
    ret = voxel.prod(0)
    return ret
    if(torch.is_tensor(ret)):
        flat_arr = ret.cpu().detach().numpy().astype(np.uint8)
    else:
        flat_arr = ret.astype(np.uint8)
    print(flat_arr.shape)
    img = Image.fromarray(flat_arr*255)
    img = img.convert("L")
    img.save(fn+".png")
    return ret

def Generate_Gripper_Kernel():
    Base_BV_Path = "../gripper_augmentation/base.binvox" #Use rospack find to generate the full path
    Finger_BV_Path = "../gripper_augmentation/finger.binvox"
    Base = GripperVox(Base_BV_Path)
    Finger1 = GripperVox(Finger_BV_Path)
    Finger2 = GripperVox(Finger1)
    Base.print_metadata()
    Finger2.print_metadata()
    Gripper = ReflexAug(Base, Finger1)
    Grid = Gripper.configure(rpy=[0.0, 0.0, 0.0])
    print(Grid.shape)
    return Grid

if __name__ == '__main__':
    rospy.init_node('Conv_SDM_node')
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
    rospy.loginfo("Init took: %f", time.time() - T_tocuda)
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
    rospy.loginfo("Test point out")

    #Gripper Augtest
    ub = min(emaxs[0],Gmaxs[2]-Gmins[2])
    convaug = ConvNet((ub, GripperVox.shape[1], GripperVox.shape[2], GripperVox.shape[3]))
    convaug.to(GPU0)
    envoxA = convaug.forward(envox[0:ub], GripperVox[Gmins[2]:Gmins[2]+ub], False)
    rospy.loginfo("Sync Test")
    #torch.cuda.synchronize()
    #visualize_points(voxel_to_pcd((1-envoxA).cpu().detach().numpy()))
    conv_map = flatten_vox(envoxA)
    rospy.loginfo("Flatten Out")
    sdm = convr.gen_sd_map(conv_map)
    dx = convr.gen_dsdf_x(sdm)
    dy = convr.gen_dsdf_y(sdm)
    exit(1)
    envoxNumpy = (envoxT).cpu().detach().numpy()
    rospy.loginfo("Operation Time: %f", time.time() - start)
    voxel_to_pcd(envoxNumpy, 0.99999, render=True)
    totT = time.time() - fstart
    rospy.loginfo("Time taken: %f", totT)
