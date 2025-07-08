#!/usr/bin/env python
'''
Library to manager the planning environment for pick and place based on partial view(w/ reconstructed)
voxelizations of the objects in the environment.
'''

import numpy as np
import copy
import csv
import trimesh
from PIL import Image
from gpu_sd_map.transforms_lib import vector2mat, transform_points, vector2matinv, invert_trans
from scipy.interpolate import RegularGridInterpolator
#from sklearn.preprocessing import normalize

#ROS Stuff:
import rospy
import std_msgs.msg
from gpu_sd_map.ros_transforms_lib import point2numpy, pose2array, mat_to_vec, convert_array_to_pose
from geometry_msgs.msg import Point32, Pose
from sensor_msgs.msg import PointCloud, ChannelFloat32
from gpu_sd_map.srv import CreateEnv, AddObj2Env
from gpu_sd_map.GenSDM import gen_3d_sd_map, flatten_vox, get_bounds, close_grid, gen_one_shot_3d_tsdm

FIGURE_DUMP_PATH_ = "/home/mohanraj/ll4ma_prime_WS/src/gpu_sd_map/Figures/"

class VoxGrid:
    def __init__(self, sparse_grid, true_size, pose, grid_size = None, res = 1):
        '''
        Base class to hold the voxel grid.

        Params:
        sparse_grid - Sparse indices of occupied voxels (N x 3 int).
        true_size - Dimensions of the entity in meters (1 x 3 float).
        pose - 6 dof pose of the entity (1 x 6 float).
        grid_size - 3D grid dimension (1 x 3 float). Calculate if None.

        Output:
        None - This is a constructor.

        Status: In Progress

        Testing:
        '''
        assert sparse_grid.shape[1] == 3, "sparse_grid is not N x 3"
        if grid_size is None:
            grid_size = sparse_grid.max(0) - sparse_grid.min(0)
        assert grid_size.shape[0] == 3, "grid_size is not 3D."
        assert true_size.shape[0] == 3, "true_size is not 3D."
        assert isinstance(pose, Pose), "pose is not of type geometry_msgs.msg/Pose."

        self.sparse_grid = sparse_grid
        self.color = np.tile(np.random.rand(1,3),(self.sparse_grid.shape[0],1))
        assert self.color.shape == self.sparse_grid.shape, "color and grid shapes mismathch {}!={}".\
            format(self.color.shape,self.sparse_grid.shape)
        self.true_size = np.array(true_size)
        self.pose = pose
        self.grid_size = np.array(grid_size)
        self.frame_id = "object_pose"
        self.res = res
        self.dim_idx = [1, 0, 2]

    def scale_to_res(self, res):
        '''
        Scales the sparse grid to target resolution.

        Params:
        res - Target resolution, num voxels per meter (int)

        Output:
        sparse_scaled - Sparse indices of occupied voxels (N x 3 int).

        Status: In Progress

        Testing:
        '''
        grid_size = self.sparse_grid.max(0) - self.sparse_grid.min(0)
        scale = self.true_size[self.dim_idx]/grid_size
        points_scaled_ = self.sparse_grid * scale
        sparse_scaled = np.array([], dtype=np.int64).reshape(0,3)
        sparse_scaled = np.vstack([sparse_scaled, points_scaled_])
        #for p_ in points_scaled_:
            #p_x_ = np.arange(p_[0]-scale[0]/2, p_[0]+scale[0]/2, 1.0/res)
            #p_y_ = np.arange(p_[1]-scale[1]/2, p_[1]+scale[1]/2, 1.0/res)
            #p_z_ = np.arange(p_[2]-scale[2]/2, p_[2]+scale[2]/2, 1.0/res)
            #p_x_, p_y_, p_z_ = np.meshgrid(p_x_, p_y_, p_z_)
            #p_all_ = np.vstack([p_x_.ravel(), p_y_.ravel(), p_z_.ravel()]).T
            #sparse_scaled = np.vstack([sparse_scaled, p_all_])
        return sparse_scaled

    def compute_pretrans(self, trans = True, rot = True):
        p_ = pose2array(self.pose)
        if not trans:
            p_[:3] = [0.0] * 3
        if not rot:
            p_[3:] = [0.0] * 3
        t_ = vector2mat(p_[:3], p_[3:])
        print("computing tee")
        return t_
            

    def publish_as_point_cloud(self, topic = 'env_points', res=1, init_node = False, points = None, frame=None):
        '''
        Publish the sparse grid as a point_cloud over ROS topic.

        Params:
        None

        Output:
        None - Publish ros topic over
        '''

        if init_node:
            rospy.init_node('env_grid_viz')
        pcl_pub_ = rospy.Publisher(topic, PointCloud, queue_size=10)
        point_cloud_ = PointCloud()
        point_cloud_.header = std_msgs.msg.Header()
        point_cloud_.header.stamp = rospy.Time.now()
        point_cloud_.header.frame_id = self.frame_id
        if frame is not None:
            point_cloud_.header.frame_id = frame
        
        point_cloud_.channels.append(ChannelFloat32())
        point_cloud_.channels[0].name = "r"
        point_cloud_.channels.append(ChannelFloat32())
        point_cloud_.channels[1].name = "g"
        point_cloud_.channels.append(ChannelFloat32())
        point_cloud_.channels[2].name = "b"

        red = np.random.uniform(0,1)
        grn = np.random.uniform(0,1)
        blu = np.random.uniform(0,1)

        if points is None:
            points_ = self.sparse_grid/res
        else:
            points_ = points
        for i in range(points_.shape[0]):
            point_cloud_.points.append(Point32(points_[i][0], points_[i][1], points_[i][2]))
            point_cloud_.channels[0].values.append(self.color[0][0])
            point_cloud_.channels[1].values.append(self.color[0][1])
            point_cloud_.channels[2].values.append(self.color[0][2])

        rate = rospy.Rate(20)
        for i in range(10):
            pcl_pub_.publish(point_cloud_)
            rate.sleep()
        return
        

    def get_dense_grid(self):
        '''
        Generates a dense voxel grid from the sparse grid.

        Params:
        None

        Output:
        dense_grid - numpy array 0s, 1s of shape grid_size (int)

        Status: In Progress

        Testing:

        TODO:
        - Make it of int / binary type
        '''
        dense_grid = np.zeros(self.grid_size)
        for point_ in self.sparse_grid:
            if np.all(point_ > 0):
                dense_grid[tuple((point_*self.res)).astype(np.int64)] = 1
        return dense_grid

    def get_dense_zxy_grid(self, points):
        '''
        Generate a dense grid in zxy format, so it is torch compatiable

        Params:
        points - array of occupied points (N x 3 numpy.ndarray)

        Output:
        dense_grid - numpy array 0s, 1s of shape (grid_size[2,0,1]) (int)

        Status: In Progress

        Testing:
        '''

        dense_grid = np.zeros([self.grid_size[2], self.grid_size[0], self.grid_size[1]])
        lb = (points.min(0) * self.res).astype(np.int64)
        for point_ in points:
            if np.all(point_ > 0):
                idx = (point_ * self.res).astype(np.int64)
                if idx[2] < self.grid_size[2] and idx[0] < self.grid_size[0] and idx[1] < self.grid_size[1]:
                    #print(idx)
                    #dense_grid[lb[2]:idx[2], idx[0], idx[1]] = 1
                    dense_grid[idx[2], idx[0], idx[1]] = 1
        return dense_grid

    def flatten(self, fname=None):
        '''
        Writes a 2D image of points projected on the xy - plane.
        
        Params:
        fname - Name of the image file (str) (Default: None, saving image is skipped if None)
        
        Output:
        flat_arr - flattened 2d image of voxel grid (w*res x h*res).
        <write_image> - if fname is not None.

        Status: Implemented

        Testing:

        TODO:
        - Add projections for the other 2 cardinal planes.
        '''
        self.recenter()
        self.recorner()
        sparse_scaled = self.scale_to_res(self.res)
        obj_trans_ = self.compute_pretrans(trans=False) #@ vector2mat(obj_.pose[:3], obj_.pose[3:])
        sparse_scaled = transform_points(sparse_scaled, obj_trans_)
        offset_ = -sparse_scaled.min(axis=0)
        offset_trans_ = vector2mat(translation = offset_)
        sparse_scaled = transform_points(sparse_scaled, offset_trans_)
        gmin = (sparse_scaled.min(axis=0)*self.res).astype(np.int)
        gmax = (sparse_scaled.max(axis=0)*self.res).astype(np.int)
        print(gmin)
        flat_arr = np.zeros(gmax[:2]-gmin[:2] + 1) # +1 to include the max index too.
        for point_ in sparse_scaled*self.res:
            flat_arr[tuple((point_[:2]).astype(np.int64))] = 1
            #flat_arr[tuple((point_[:2]+1).astype(np.int64))] = 1

        if fname is not None:
            img = Image.fromarray(flat_arr*255)
            img = img.convert("L")
            img.save(FIGURE_DUMP_PATH_ + str(fname) +".png")
        return flat_arr

class ObjGrid(VoxGrid):
    '''
    TBD what goes here.
    '''
    def __init__(self, sparse_grid, true_size, pose, grid_size, res = 1):
        super().__init__(sparse_grid, true_size, pose, grid_size, res)
        #self.grid_size = grid_size * 2
        self.sdm3d = None
        self.obj_pts = self._raw_grid_idx_to_pts()
        self.binned_pts = self.bin_points()
        print(f"binning would make obj_pts shape {self.obj_pts.shape} to shape {self.binned_pts.shape}")
        self.color = np.tile(np.random.rand(1,3),(self.obj_pts.shape[0],1))
        self.dim_idx = [0,1,2]
        self.Trans = np.eye(4)
        self.Buffer = 0.3
        self.trunc_max = None
        #if res != 1:
            #print("Warning!!! Setting parameter res does nothing in this implementation")

    def bin_points(self, binsize = 0.03):
        binnedpts = []
        binnedpts2 = []
        for i in range(3):
            bins = np.arange(self.obj_pts.min(axis = 0)[i], \
                             self.obj_pts.max(axis = 0)[i] + binsize*2, \
                             binsize)
            binnedpts.append(bins[np.digitize(self.obj_pts[:,i], bins, right=False)])
            binnedpts2.append(bins[np.digitize(self.obj_pts[:,i], bins, right=True)])
        binnedpts = np.vstack(binnedpts).T
        binnedpts2 = np.vstack(binnedpts2).T
        binned = np.unique(np.vstack([binnedpts, binnedpts2]), axis = 0) - binsize/2
        self.publish_as_point_cloud(topic='bin_test', points=binned, frame='object_pose')
        return binned
            
    def set_pose(self, pose, preserve=False, sref=True):
        if isinstance(pose, Pose):
            pose_arr = np.asarray(pose2array(pose))
        else:
            pose = np.asarray(pose)
            if pose.size == 6:
                pose_arr = pose
            elif pose.shape == (4,4):
                self.Trans = pose
                return
        if preserve:
            Trans = vector2mat(pose_arr[:3], pose_arr[3:])
            self.Trans = self.compute_pretrans()
            self.Trans[:3,:3] = Trans[:3, :3] @ self.Trans[:3,:3]
            self.Trans[:3, 3] = Trans[:3, 3]
        else:
            self.Trans = vector2mat(pose_arr[:3], pose_arr[3:])
        if sref:
            self.Trans[2,3] = self.true_size[2]/2
        #print(self.Trans)
        return

    def get_pose(self):
        vec = mat_to_vec(self.Trans)
        return convert_array_to_pose(vec, 'place_table_corner')
        
        
    def offset_grid(self, offset):
        '''
        Only translate points in the grid by an offset

        Params:
        offset - 3d translation (3 x 1 float)
        '''
        toff_ = vector2mat(offset)
        return transform_points(self.sparse_grid, toff_)

    def offset_points(self, offset):
        '''
        Only translate points in the grid by an offset

        Params:
        offset - 3d translation (3 x 1 float)
        '''
        toff_ = vector2mat(offset)
        return transform_points(self.obj_pts, toff_)
    
    def recenter(self):
        '''
        Points on the voxel grid are positive by default hence origin is at min corner,
        this functions sets the origin to the centroid of the grid.
        The the grid has negative points it is already centered hence ignore.
        '''
        #print("Current Min")
        #print(self.sparse_grid.min(0))
        raise DeprecationWarning("recenter and affects grid state!!")
        if np.all(self.sparse_grid.min(0)>=0):
            offset_ = -self.grid_size/2
            #print("***********************************")
            #print(offset_)
            self.offset_grid(offset_)
        return

    def test_cornered(self):
        if np.any(self.sparse_grid.min(0)<0):
            raise RuntimeError("grid state changed")

    def recorner(self):
        '''
        Inverse of recenter, move origin to the min corner of the grid.
        Ignore is all grid is positive since it is already offset.
        '''
        raise DeprecationWarning("recenter and affects grid state!!")
        if np.any(self.sparse_grid.min(0)<0):
            offset_ = self.grid_size/2
            self.offset_grid(offset_)
        return

    def gen_sd_map(self):
        '''
        Apply brushfire on the sparse voxel grid to generate the sdm map.
        '''
        #self.grid_size *= 2
        self.expand_grid()
        offset_ = np.array(self.true_size + self.Buffer) * 0.5
        #offset_[0] = 0.0
        offset_pts = self.offset_points(offset_)
        print(offset_pts)
        #print(self.sparse_grid)
        dense_grid = self.get_dense_zxy_grid(offset_pts)
        #flatten_vox(1-dense_grid, fn = "OpenGrid_Z")
        #flatten_vox(1-dense_grid, fn = "OpenGrid_X", axis=1)
        #flatten_vox(1-dense_grid, fn = "OpenGrid_Y", axis=2)
        dense_grid = close_grid(dense_grid)[0][0]
        try:
            mins, maxs = get_bounds(dense_grid)
        except:
            mins = [0]*3
            maxs = dense_grid.shape
        #flatten_vox(1-dense_grid, fn = "ClosedGrid_Z")
        #flatten_vox(1-dense_grid, fn = "ClosedGrid_X", axis=1)
        #flatten_vox(1-dense_grid, fn = "ClosedGrid_Y", axis=2)
        #print(dense_grid.shape)
        #convr = ConvNet(dense_grid.shape)
        self.sdm3d_fg = gen_3d_sd_map(dense_grid)#gen_one_shot_3d_tsdm(dense_grid)
        self.sdm3d = self.sdm3d_fg[:,:,:,0]
        lv_pts = self.get_level_set()
        self.frame_id = "object_pose"
        self.publish_as_point_cloud("lvl_set", points=lv_pts)
        #lv_pts = self.get_level_set(0,5)
        #self.publish_as_point_cloud("lvl_spl", points=lv_pts)
        z = np.arange(dense_grid.shape[0])
        x = np.arange(dense_grid.shape[1])
        y = np.arange(dense_grid.shape[2])
        self.sdm3dI = RegularGridInterpolator((z, x, y), self.sdm3d_fg, bounds_error=False, fill_value=np.max(self.sdm3d))
        return self.sdm3d

    def expand_grid(self):
        #self.recenter()
        #self.recorner()
        #obj_pts_ = self.raw_grid_idx_to_pts()
        #print("Size Compare: ")
        #print(obj_pts_.max(0) - obj_pts_.min(0))
        #print(self.true_size)
        self.grid_size = (self.true_size[self.dim_idx] + self.Buffer) * self.res
        self.grid_size = self.grid_size.astype(int)
        #print(self.grid_size)
        #obj_trans_ = obj_.compute_pretrans() #@ vector2mat(obj_.pose[:3], obj_.pose[3:])
        #obj_pts_ = transform_points(obj_pts_, obj_trans_)
        #self.sparse_grid = obj_pts_

    def _raw_grid_idx_to_pts(self):
        offset_ = -self.grid_size/2
        sparse_centered = self.offset_grid(offset_)
        grid_size = sparse_centered.max(0) - sparse_centered.min(0)
        step = grid_size / 32
        scale = self.true_size[self.dim_idx]/grid_size
        #print(f'***obj_grid_size: {self.true_size[self.dim_idx]}')
        #print(f'scale: {scale}')
        pre_trans = self.compute_pretrans(trans = False)
        #pre_trans = pre_trans @ pre_trans
        sparse_centered = transform_points(sparse_centered * scale, pre_trans)
        ub = sparse_centered.max(0)
        lb = sparse_centered.min(0)
        grid_size = sparse_centered.max(0) - sparse_centered.min(0)
        print(f'\n\n\ncentroid sanity: {sparse_centered.max(0) + sparse_centered.min(0)}')
        step = grid_size / 32
        self.publish_as_point_cloud(topic='fill_test', points=sparse_centered, frame=None)
        new_sparse = np.array([]).reshape(0,3)
        for sp in sparse_centered:
            zs = np.arange(lb[2],sp[2],step[2]).reshape(-1,1)
            ns = np.tile(sp[:2],(zs.shape[0],1))
            ns = np.hstack([ns,zs])
            new_sparse = np.vstack([new_sparse, ns])
        sparse_centered = np.vstack([sparse_centered, new_sparse])
        points = transform_points(sparse_centered, invert_trans(pre_trans))
        self.publish_as_point_cloud(topic='fillnt_test', points=points, frame=None)
        return points

    def get_level_set(self, low=0, high=None):
        #if self.sdm3d is not None:
        if high is None:
            high = low
        idx = np.logical_and(self.sdm3d >= low, self.sdm3d <= high).nonzero()
        idx = np.vstack(idx)
        idx = idx[[1,2,0]].T
        pts = idx / self.res
        offset = -np.array(self.true_size + self.Buffer) * 0.5
        toff_ = vector2mat(offset)
        pts = transform_points(pts, toff_)
        return pts

    def max_tsdf(self):
        if not hasattr(self,'trunc_max') or self.trunc_max is None:
            self.trunc_max = np.max(self.sdm3d)
        return self.trunc_max
    def query_points(self, qpoints):
        oTt = invert_trans(self.Trans)
        offset = np.array(self.true_size + self.Buffer) * 0.5
        toff = vector2mat(offset)
        o_qpoints = transform_points(qpoints, toff @ oTt)
        idx = o_qpoints * self.res#.astype(np.int)
        #print('idx:',idx[437])
        #idx = np.maximum(idx,0)
        #idx = np.minimum(idx[:,[2,0,1]], np.asarray(self.sdm3d.shape)-1) # idx is now in zxy
        sdfg = self.sdm3dI(idx[:,[2,0,1]])#self.sdm3d[idx[:,0], idx[:,1], idx[:,2]]
        sd = sdfg[:,0]
        dsd = sdfg[:,1:]
        #print(sd[437])
        grad_check = not True
        if grad_check:
            delt = [0, 0, 0]
            diff = 1e-10
            dsd_fd = np.zeros((idx.shape[0],3))
            for i in range(3):
                delt = [0, 0, 0]
                delt[i] = diff
                idxf = idx + delt
                idxb = idx - delt
                sd_f = self.sdm3dI(idxf[:,[2,0,1]])[:,0]
                sd_b = self.sdm3dI(idxb[:,[2,0,1]])[:,0]
                dsd_fd[:, i] = (sd_f - sd_b) / (2*diff)
            coses = np.einsum('ij, ij -> i', dsd, dsd_fd)
            coses = coses/(np.linalg.norm(dsd,axis=1) * np.linalg.norm(dsd_fd,axis=1))
            print(coses)
        #idxI = idx.astype(np.int)
        #idxI = np.maximum(idxI,0)
        #idxI = np.minimum(idxI[:,[2,0,1]], np.asarray(self.sdm3d.shape)-1) # idx is now in zxy
        #dsd = self.d_sdm[:, idxI[:,0], idxI[:,1], idxI[:,2]].T #d_sdf in xyz but sdf in zxy
        #dsd = normalize(dsd, axis=1)
        dsd = dsd @ oTt[:3,:3]
        #dsd = normalize(dsd, axis=1)
        #print(dsd)
        return sd, dsd

    def get_mesh(self, fn='grasp_obj.stl', show=False):
        lv_pts = self.get_level_set(-0.001, 0)
        print(lv_pts.shape)
        print(self.sdm3d.min(), self.sdm3d.max())
        mesh = trimesh.voxel.ops.points_to_marching_cubes(lv_pts, pitch=0.01)
        mesh.export(fn)
        if show:
            mesh.show()


    def get_bounds_at_config(self):
        pts = self.obj_pts
        if self.sdm3d is not None:
            pts = self.get_level_set(-0.001, 0)
        pts = transform_points(pts, self.Trans)
        min_bnds = pts.min(axis=0)[:3]
        max_bnds = pts.max(axis=0)[:3]
        return min_bnds, max_bnds
        
def test():
    grid_size = np.array([32,32,32])
    def get_arange(gsize):
        low = np.random.randint(gsize//2)
        high = np.random.randint(gsize//2)
        return np.arange(low, gsize//2 + high)
    x = get_arange(grid_size[0])
    y = get_arange(grid_size[1])
    z = get_arange(grid_size[2])
    sparse_grid = np.stack(np.meshgrid(x,y,z), axis=-1).reshape(-1,3)

    def circle_filter(radii):
        rows = []
        center = np.random.randint(grid_size)
        for i in range(sparse_grid.shape[0]):
            if np.linalg.norm(sparse_grid[i] - grid_size/2) <= radii:
                rows.append(i)
        return rows

    sparse_grid = sparse_grid[circle_filter(np.random.randint(grid_size[0]))]
    
    obj_size = np.random.rand(3) * 0.15
    ogrid = ObjGrid(sparse_grid, obj_size, Pose(), grid_size, 1000)
    ogrid.gen_sd_map()
    qpts = np.random.rand(10,3) * 2 * obj_size - 1 * obj_size
    ogrid.query_points(qpts)
    ogrid.get_mesh()
    
    
    

class EnvVox(VoxGrid):
    def __init__(self, x_dim = 0.0, y_dim = 0.0, z_dim=0.5, res=1000, world_trans=Pose()):
        '''
        Holds the feed forward memory of environment occupancy.
        Environment here being a table surface.

        Params:
        x_dim - dimension in x direction w.r.t world frame (float).
        y_dim - dimension in y direction w.r.t world frame (float).
        z_dim - dimension in z direction w.r.t world frame (float).
        res - points per unit dimension (int).
        world_trans - Transform between the world frame and env reference (1 x 6 float).

        Output:
        None - This is a constructor.
        '''
        sparse_grid = np.array([], dtype=np.int64).reshape(0,3)
        true_size = np.array([x_dim, y_dim, z_dim])
        grid_size = np.array(true_size*res).astype(np.int64)
        super().__init__(sparse_grid, true_size, world_trans, grid_size, res)
        self.color = np.array([], dtype=np.float32).reshape(0,3)
        self.frame_id = "place_table_corner"
        self.ObjList = {}

    def test_initialized(self):
        '''
        Tests if the EnvVox instance is valid and initialized properly.

        Params:
        self

        Output:
        self

        Status: Working

        Testing: Verified

        TODO:
        - Add other checks for resolution here
        '''
        if np.any(self.true_size <= 0):
            return False
        return True

    def add_object(self, obj_id, sparse_obj_grid, obj_size, place_conf):
        '''
        Creates an ObjGrid instance for the given voxel grid and adds it to the environment.

        Params:
        obj_id - Unique identifier for object (int or str).
        sparse_obj_grid - Sparse voxel grid of object (N x 3 int).
        obj_size - Dimensions of the object (1 x 3 float).
        place_conf - 6 dof placement configuration (1 x 6 float).
        
        Status: Implemented

        Testing:
        '''
        assert self.test_initialized(), "Impractical dimensions set, Configure EnvVox instance first"
        obj_ = ObjGrid(sparse_obj_grid, obj_size, place_conf)
        self.ObjList[obj_id] = obj_

    def expand_grid(self):
        '''
        Processes the ObjVox in the ObjList and adds them to the environment grid.
        
        Params:
        None

        Output:
        None - Updates the sparse grid.

        Status: Implemented

        Testing:

        TODO:
        - Apply numpy.unique to sparse grid and check if eliminates duplicate points
        '''
        for id_, obj_ in self.ObjList.items():
            obj_.recenter()
            #obj_pts_ = obj_.sparse_grid
            obj_pts_ = obj_.scale_to_res(self.res)
            obj_trans_ = obj_.compute_pretrans() #@ vector2mat(obj_.pose[:3], obj_.pose[3:])
            obj_pts_ = transform_points(obj_pts_, obj_trans_)
            self.sparse_grid = np.vstack([self.sparse_grid, obj_pts_])
            self.color = np.vstack([self.color, obj_.color])
            #obj_.recorner()
        return

    def add_object_handle(self, req):
        '''
        Service to add object to the EnvVox object from external packages (C++, python 2, ...)
        
        Params:
        req - gpu_sd_map/AddObj2Env.srv
        
        OutPut:
        resp - gpu_sd_map/AddObj2Env.srv
        
        Status: Implemented
        
        Testing:
        '''

        rospy.loginfo("Add object service invoked")
        #try:
        obj_size = np.array(req.obj_size)
        #place_conf = np.array(req.place_conf)
        self.add_object(req.obj_id, point2numpy(req.sparse_obj_grid), obj_size, req.place_conf)
        self.expand_grid()
        rospy.loginfo(self.sparse_grid.min(axis=0))
        rospy.loginfo(self.sparse_grid)
        self.publish_as_point_cloud(topic=req.env_name+"_env")
        return True
        ##except Exception as e:
        ##    rospy.loginfo("Exception {} occured".format(e))
        #    return False

    def config_environment_handle(self, req):
        '''
        Ros Service to create Envox object from external packages (C++, python 2, ...)
        
        Params:
        req - gpu_sd_map/CreateEnv.srv
        
        Output:
        resp - gpu_sd_map/CreateEnv.srv
        
        Status: Working
        
        Testing: Verified
        '''
        if not self.test_initialized():
            rospy.loginfo("Initializing new voxelized environment for SD mapping")
        else:
            rospy.logwarn("Environment exists, Overwriting")
            
        dims = req.dimension
        res = req.resolution
        self.__init__(dims[0], dims[1], dims[2], res, req.world_trans)

        return self.test_initialized()

    def gen_sd_map(self):
        '''
        Apply brushfire on the sparse voxel grid to generate the sdm map.
        '''

        dense_grid = self.get_dense_zxy_grid()
        try:
            mins, maxs = get_bounds(dense_grid)
        except:
            mins = [0]*3
            maxs = dense_grid.shape
        flatten_vox(1-dense_grid[mins[0]:maxs[0]])
        #print(dense_grid.shape)
        #convr = ConvNet(dense_grid.shape)
        sdm3d,j,k = gen_3d_sd_map(dense_grid[mins[0]:maxs[0]])
        return sdm3d, j, k
    
def read_voxel(FileName):
    with open(FileName, newline='') as f:
        data = list(csv.reader(f, delimiter=' '))
    data = np.array(data, dtype='int')
    return data

if __name__ == '__main__':
    rospy.init_node('gpu_sdm_env_manager_server')
    envs = {}

    test()
    
    def create_environment_handle(req):
        '''
        Local function to maintain different SD map environments, if this thread dies, the envs are lost.

        Params:
        req - gpu_sd_map/CreateEnv.srv

        Output:
        bool - operation successful or not

        Status: Implemented
        '''
        if req.env_name in envs.keys():
            rospy.logwarn("Environment already exists, Overwriting... Use or define a configure service")
        else:
            rospy.loginfo("Initializing new voxelized environment for SD mapping")
            envs[req.env_name] = EnvVox()
        return envs[req.env_name].config_environment_handle(req)

    def add_object_to_env_handle(req):
        if req.env_name in envs.keys():
            return envs[req.env_name].add_object_handle(req)
        else:
            rospy.logerr("Environment doesnt exist, create it first")
            return False
    
    rospy.Service('gpu_sdm_create_env', CreateEnv, create_environment_handle)
    rospy.Service('gpu_sdm_add_obj', AddObj2Env, add_object_to_env_handle)
    rospy.spin()
