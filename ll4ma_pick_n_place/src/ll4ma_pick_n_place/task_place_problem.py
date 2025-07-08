#!/usr/bin/env python3

from ll4ma_opt.problems.problem import Problem, Constraint
from gpu_sd_map.gripper_augment import GripperVox, ReflexAug, visualize_points
from gpu_sd_map.environment_manager import ObjGrid
from gpu_sd_map.ros_transforms_lib import convert_array_to_pose, TransformPose, \
    pose_to_array, pose_to_6dof, TransformPoseStamped, get_tf_mat, mat_to_vec, vector_transform, \
    test_3d_rotation_equivalence, dof6_to_dof7_array, get_space_jacobian, get_adjoint_transform, \
    vector_inverse, skew_symmetric_mat, batch_vector_skew_mat, euler_from_mat, \
    publish_as_point_cloud, euler_diff
from gpu_sd_map.transforms_lib import vector2mat, vector2matinv, get_rotation_jacobian, \
    get_rotation_matrix, vector_similarity, invert_trans, jacobian_similarity, \
    angular_velocity_to_rpy_dot, decompose_transform, verify_rpy_dot, \
    transform_points, TransformPoseMat, yaw_mat, euler_to_angular_velocity
import ll4ma_pick_n_place.placement_heuristics as heurs
from ll4ma_pick_n_place.opt_problem import ExecLogger, chomp_smooth, chomp_smooth_grads, \
    compare_grads, gen_3d_grid_pts, gen_query_grid_pts
import numpy as np
import copy

class Collision_Constraint(Constraint, ExecLogger):
    pblm = None
    epsilon = 0
    EGrid = None
    Q_CA = {}

    nErrorCalls = 0
    tErrorTime = 0

    nGradCalls = 0
    tGradTime = 0

    nCache_Calls = 0
    tCache_Time = 0

    def static_init(pblm, epsilon = 0.025):
        Collision_Constraint.pblm = pblm
        Collision_Constraint.epsilon = epsilon
        for key in pblm.Env_List.keys():
            Collision_Constraint.Q_CA[key] = (np.array([]), np.array([]), np.array([]))
    
    def __init__(self, EKey, idx):
        self.EKey = EKey
        self.idx = idx
        self._cname = "Obj_Coll_" + str(EKey)
        #ExecLogger.__init__(self, [self._cname, self._cname+G_GRAD_LOG_SUFFIX, self._cname+P_GRAD_LOG_SUFFIX, \
        #                           self._cname + 'sdf_grad'])
        self.sbuff = 5
        self.mag = 1
        if idx == 0:
            self.always_active=True

    @classmethod
    def performance_stats(cls):
        cls.cache_performance_stats()
        return [(cls.nErrorCalls, cls.tErrorTime), \
                (cls.nGradCalls, cls.tGradTime)]

    @classmethod
    def reset_performance_stats(cls):
        cls.nErrorCalls = 0
        cls.tErrorTime = 0
        cls.nGradCalls = 0
        cls.tGradTime = 0
        nCache_Callss = 0
        tCache_Time = 0
        
    @classmethod
    def cache_performance_stats(cls):
        print(f"Cache calls: {cls.nCache_Calls}, taking {cls.tCache_Time}s")

    def cache_(self, q):
        type(self).nCache_Calls += 1
        
        q = self.pblm._process_q(q)
        
        tTo = vector2mat(q[:3], [0.0, 0.0, q[3]])

        obj_pts = self.pblm.Obj_Grid.binned_pts

        query_pts = transform_points(obj_pts, tTo)
        
        sd, dsd = self.pblm.Env_List[self.EKey].query_points(query_pts)

        dl = self.pblm.Env_List[self.EKey].max_tsdf() - self.sbuff
        
        csd = -chomp_smooth(sd-dl, self.sbuff) + dl
        csd_g = -chomp_smooth_grads(sd-dl, self.sbuff)

        cdsd = np.einsum('i,ij->ij', csd_g, dsd)

        JR = get_rotation_jacobian(0, 0, q[3])

        cdsd_o = np.einsum('ij, ij -> i', cdsd, (JR[2] @ obj_pts.T).T)

        grads = np.hstack([cdsd, cdsd_o.reshape(-1,1)])

        type(self).Q_CA[self.EKey] = (q, csd*1e-3, -grads, dsd)
                         
        #type(self).tCache_Time += time.perf_counter() - time0

    def error(self, q):
        type(self).nErrorCalls += 1
        res = self.error_core(q)
        #type(self).tErrorTime += time.perf_counter() - time0
        return res

    def error_core(self, q):
        q = self.pblm._process_q(q)
        if not np.array_equal(self.Q_CA[self.EKey][0], q):
            self.cache_(q)
        #if idx is None:
        idx = np.argmin(self.Q_CA[self.EKey][1])
        #print(np.argmin(self.Q_CA[self.EKey][1]))
        #idx = self.idx
        return (self.epsilon - self.Q_CA[self.EKey][1][idx]) * self.mag

    def error_gradient(self, q):
        type(self).nGradCalls += 1
        #time0 = time.perf_counter()
        grads = self.error_gradient_core(q)
        #self.log_val(self._cname+G_GRAD_LOG_SUFFIX, q.tobytes(), np.linalg.norm(grads[0, self.pblm.G_IDX]))
        #self.log_val(self._cname+P_GRAD_LOG_SUFFIX, q.tobytes(), np.linalg.norm(grads[0, self.pblm.P_IDX]))
        idx = np.argmin(self.Q_CA[self.EKey][1])
        cart_grad = self.Q_CA[self.EKey][3][idx]
        #self.log_val(self._cname+'sdf_grad', q.tobytes(), np.linalg.norm(cart_grad))
        #type(self).tGradTime += time.perf_counter() - time0
        return grads

    def error_gradient_core(self, q):
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)
        if not np.array_equal(self.Q_CA[self.EKey][0], q):
            self.cache_(q)
        idx = np.argmin(self.Q_CA[self.EKey][1])
        #idx = self.idx
        q_grads[0, :] = self.Q_CA[self.EKey][2][idx]

        GradCheck = not True
        if GradCheck:
            fd_grads = self.error_gradient_fd(q)
            #np.set_printoptions(precision=2, floatmode='maxprec_equal')
            #print('fd vs a_grads error: \n', fd_grads, q_grads, fd_grads - q_grads)
            compare_grads(fd_grads, q_grads.T * self.mag)
            #print('Similarity: ', np.arccos(vecsim)*180/np.pi, vecsim )
        return q_grads.T * self.mag

    def error_gradient_fd(self, q, delt=1e-3):
        #delt = 1e-5
        print("FD grads called")
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        q = q.reshape(-1) # Change input to row vector
        g = np.matrix(q * 0.0)
        for i in range(self.pblm.dof):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            cf = self.error_core(q_new)
            
            q_new = copy.deepcopy(q)
            q_new[i] -= delt
            cb = self.error_core(q_new)

            g[0, i] = (cf - cb) / (2*delt)
        #g = g / np.linalg.norm(g)
        return g


class Bounds_IEq(Constraint, ExecLogger):
    def __init__(self, idx, bound, pblm, upper_bound=False):
        self.idx = idx
        self.bound = bound
        self.upper_bound = upper_bound
        self._cname = str(self.idx)
        self.pblm = pblm
        self.always_active=True
        if self.upper_bound:
            self._cname += '_upper_bound'
        else:
            self._cname += '_lower_bound'

    def error(self, q):
        #q = np.asarray(q).reshape(-1)
        q = self.pblm._process_q(q)
        err = self.bound - q[self.idx]
        if self.upper_bound:
            return -err.item()
        return err.item()
    
    def error_gradient(self, q):
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)
        #print(q_grads.shape)
        if self.upper_bound:
            q_grads[0, self.idx] = 1
        else:
            q_grads[0, self.idx] = -1
        return q_grads.T

class Task_Place_Problem(Problem):
    def __init__(self, obj_grid, env_list, place_heuristic=None):
        super().__init__()
        if place_heuristic is None:
            place_heuristic = heurs.corner_heuristic()
        self.place_heuristic = place_heuristic
        self.place_frame = 'place_table_corner'
        if hasattr(self.place_heuristic, 'frame_id'):
            self.place_frame = self.place_heuristic.frame_id
        self.Env_List = env_list
        self.dof = 4

        self.table_corner = [0.07442, 0.186142]
        self.table_size = [0.5, 0.6]
        self.upright = [0.0, 0.0]

        assert isinstance(obj_grid, ObjGrid), "Param obj_grid is not of ObjGrid type"
        obj_grid.test_cornered()
        print("Unpacking obj_grid")
        self.Obj_Grid = obj_grid
        self.object_sparse_grid = obj_grid.sparse_grid
        self.object_voxel_dims = obj_grid.grid_size
        self.object_size = obj_grid.true_size
        self.x0 = np.zeros((self.dof, 1))
        self.xp = np.zeros(6)
        self.initial_solution = self.x0
        pad = max(self.object_size[:2])/2

        for i_ in range(2):
            self.min_bounds[i_] = pad
            self.max_bounds[i_] = self.table_size[i_]
        self.min_bounds[2] = self.object_size[2]/2
        self.max_bounds[2] = self.object_size[2]/2 + 0.05
        self.min_bounds[3] = -np.pi
        self.max_bounds[3] = np.pi
        self.add_bounds_as_ieqs()

        try:
            self.obj_world_mat = get_tf_mat("object_pose", "world")
            self.ptable_world_mat = get_tf_mat(self.place_frame, "world")
        except:
            print("Warning states not found")

    def size(self):
        return 4

    def _process_q(self, q):
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        return q.reshape(-1) # Change input to row vector

    def add_bounds_as_ieqs(self):
        for i in range(self.size()):
            self.inequality_constraints.append(Bounds_IEq(i, self.min_bounds[i], self))
            self.inequality_constraints.append(Bounds_IEq(i, self.max_bounds[i], self, True))

    def add_collision_constraints(self):
        Collision_Constraint.static_init(self)
        obj_pts = self.Obj_Grid.binned_pts
        num_pts = obj_pts.shape[0]
        print(f"Adding {num_pts} object collision checks per object")
        for key in self.Env_List.keys():
            for i in range(0,num_pts, 1):
                self.inequality_constraints.append(Collision_Constraint(key, i))
                break
            
    def full_pose(self, q):
        self.xp *= 0
        self.xp[:3] = q[:3]
        self.xp[5] = q[3]
        return self.xp
        
    def cost(self, q):
        q = self._process_q(q)
        xp = self.full_pose(q)
        return self.place_heuristic.cost(np.asarray(xp))

    def cost_gradient(self, q):
        q = self._process_q(q)
        q_grads = np.matrix(q*0.0)
        xp = self.full_pose(q)
        q_grads[0, :] = self.place_heuristic.grad(np.asarray(xp))[0,[0,1,2,5]]
        return q_grads.T

    def publish_env_cloud(self):
        publish_env_cloud(self.Env_List)

    def query_full_env(self, grid_pts):
        min_sd = None
        for key in self.Env_List.keys():
            #keys = [*Env_List]
            sd, dsd = self.Env_List[key].query_points(grid_pts)
            if min_sd is None:
                min_sd = sd
            else:
                min_sd = np.minimum(sd, min_sd)
        return min_sd

    def get_2d_sd_grid(self, min_z = 0.0, max_z=0.5):
        MAX_HT = max_z
        MIN_HT = min_z
        g = gen_query_grid_pts(self.table_size[0], self.table_size[1])
        g[:,2] = MIN_HT
        min_sd = self.query_full_env(g)
        for z in np.arange(MIN_HT, MAX_HT, 0.01):
            g[:,2] = z
            sd = self.query_full_env(g)
            if sd is not None:
                min_sd = np.minimum(sd, min_sd)
        I = np.zeros(np.round(np.array(self.table_size)*1e2).astype(int))
        VI = np.stack([I,I,I,I], axis=-1)
        I += 100
        ids = (np.around(g,2)*1e3).astype(int)
        if min_sd is not None:
            #colors = map_rgb(min_sd)
            #VI[ids[:,0], ids[:,1], :] = colors
            I[ids[:,0], ids[:,1]] = min_sd
        #pyplot.imshow(VI)
        #pyplot.show()
        I[I < 0.5] = 0
        I[I > 0.5] = 1
        return I

    def get_3d_sd_grid(self):
        MAX_HT = 0.5
        grid_size = np.round(np.array(self.table_size)*1e2).astype(int).tolist()
        grid_size = grid_size + [50]
        I = np.zeros(grid_size) + 1000
        #print(I.size*I.itemsize)
        g = gen_3d_grid_pts(self.table_size[0], self.table_size[1],0.0, MAX_HT)
        #print(np.max(g[:,2]))
        ids = (np.around(g,2)*1e2).astype(int)
        ids[:,2] = (np.around(g[:,2],2)*1e2).astype(int)
        #for i, z in enumerate(np.arange(0.0, MAX_HT, 0.01)):
        #    g[:,2] = z
        sd = self.query_full_env(g)
        if sd is not None:
            I[ids[:,0], ids[:,1], ids[:,2]] = sd
        return I

    def flatten_3d_grid(self, G, min_z=0.0, max_z=0.5, bound=False):
        lo = (np.around(min_z-0.1, 2)*100).astype(int)
        lo = max(0, lo)
        hi = (np.around(max_z, 2)*100).astype(int)
        #print(np.min(G), np.max(G))
        I = np.min(G[:,:,lo:hi], axis=2)
        #print(np.min(I), np.max(I))
        I[I < 10] = 0
        I[I > 10] = 1
        if bound:
            obj_size = np.array(self.object_size)
            obj_size = np.around(obj_size, 2) * 100
            boundary_size = np.max(obj_size[:2]/2).astype(int) + 1
            I[:boundary_size, :] = 0
            I[-boundary_size:, :] = 0
            I[:, :boundary_size] = 0
            I[:, -boundary_size:] = 0
        return I

    def get_2d_obj_grid(self):
        obj_size = np.array(self.object_size)
        obj_size = np.around(obj_size, 2)
        kernel_size = np.max(obj_size[:2]) + 0.06
        #print(obj_size)
        g = gen_query_grid_pts(kernel_size, kernel_size)
        print(kernel_size)
        print(np.max(g))
        q = g - np.array([kernel_size, kernel_size, 0])/2
        min_sd, dsd = self.Obj_Grid.query_points(q)
        for z in np.arange(-obj_size[2]/2, obj_size[2], 0.01):
            q[:,2] = z
            sd, dsd = self.Obj_Grid.query_points(q)
            min_sd = np.minimum(sd, min_sd)
        print(kernel_size)
        print(np.max(g))
        #print(np.min(min_sd), np.max(min_sd))
        I = np.zeros(np.array([kernel_size*1e2, kernel_size*1e2]).astype(int))
        VI = np.stack([I,I,I,I], axis=-1)
        I += 100
        ids = (np.around(g,2)*1e2).astype(int)
        if min_sd is not None:
            #colors = map_rgb(min_sd)
            #VI[ids[:,0], ids[:,1], :] = colors
            I[ids[:,0], ids[:,1]] = min_sd
        #pyplot.imshow(VI)
        #pyplot.show()
        I[I > 0] = 0
        I[I < 0] = 1
        return I

    def get_2d_gripper_kernel(self):
        #gripper = Reflex_Collision.generate_gripper()
        Pkg_Path = "/home/mohanraj/robot_WS/src/gpu_sd_map"
        Base_BV_Path = Pkg_Path + "/gripper_augmentation/base_hd.binvox" 
        Finger_BV_Path = Pkg_Path + "/gripper_augmentation/finger_sweep_hd.binvox"
        Base = GripperVox(Base_BV_Path)
        Finger1 = GripperVox(Finger_BV_Path)
        Finger2 = GripperVox(Finger1)

        gripper = ReflexAug(Base, Finger1)
        gripper_pts = gripper.Set_Pose_Preshape(preshape=0, centered=False)
        #print(gripper_pts)
        gripper_pts = np.vstack([gripper_pts, \
                                 [-0.160, 0.0, 0.0], \
                                 [-0.125, 0.0, 0.0], \
                                 [-0.245, 0.0, 0.0]])
        #print(gripper_pts)

        #gripper2 = Reflex_Collision.generate_gripper()
        #binned = gripper2.Set_Pose_Preshape(preshape=0)
        
        #publish_as_point_cloud(binned, topic='reflex_binned', frame_id = 'reflex_palm_link')
        oTg = self.joint_to_grasp(self.x0, asmat=True)
        tTo = self.get_place_proto()
        tTo[0,3] = 0.0
        tTo[1,3] = 0.0
        tTg = oTg
        query_pts = transform_points(gripper_pts, tTg)
        min_z, max_z = (np.min(query_pts[:,2]), np.max(query_pts[:,2]))
        kern_ref_pts = np.abs(query_pts[:,:2])
        kern_ref_pts = np.vstack([kern_ref_pts, kern_ref_pts[-2] + 0.08, \
                                  kern_ref_pts[-1] + 0.12])
        kernel_size = np.around(np.max(kern_ref_pts) + 0.02, 2) * 2
        query_pts += kernel_size/2
        fill_idx = np.around(query_pts * 1e2, 2).astype(int)
        flrist_fill = fill_idx[-3]
        flange_fill = fill_idx[-2]
        wrist_fill = fill_idx[-1]
        I = np.zeros(np.array([kernel_size*1e2, kernel_size*1e2]).astype(int))
        I[fill_idx[:,0], fill_idx[:,1]] = 1
        def circle_mask(radii, center, shape):
            return (np.arange(0, shape)[np.newaxis, :] - center[1])**2 + \
                (np.arange(0, shape)[:,np. newaxis] - center[0])**2 < radii**2
        I[circle_mask(0.10*1e2, flrist_fill, I.shape[0])] = 1
        I[circle_mask(0.08*1e2, flange_fill, I.shape[0])] = 1
        I[circle_mask(0.12*1e2, wrist_fill, I.shape[0])] = 1
        return I, min_z, max_z
        
def publish_env_cloud(env_dict, frame_id='place_table_corner'):
    all_pts = np.array([]).reshape(0,3)
    colors = np.array([]).reshape(0,3)
    for key in env_dict.keys():
        VoxGrid = env_dict[key]
        pts = VoxGrid.obj_pts
        pts = transform_points(pts, VoxGrid.Trans)
        all_pts = np.vstack([all_pts, pts])
        colors = np.vstack([colors,VoxGrid.color])
    publish_as_point_cloud(all_pts, colors=colors, frame_id=frame_id)
