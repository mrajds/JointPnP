#!/usr/bin/env python3

from ll4ma_opt.problems.problem import Problem, Constraint
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
from ll4ma_pick_n_place.opt_problem import ExecLogger, chomp_smooth, chomp_smooth_grads, \
    compare_grads, Grasp_Collision, Reflex_Collision
from ll4ma_pick_n_place.planning_client import get_IK
from ll4ma_pick_n_place.kdl_utils import ManipulatorKDL
from ll4ma_pick_n_place.grasp_utils_client import GraspUtils
import numpy as np
import copy

ROBOT_DESCRIPTION = 'robot_description'
NP_DTYPE_ = np.float32

class Reflex_Collision(Constraint, ExecLogger):
    pblm = None
    gripper = None
    epsilon = 10
    EGrid = None
    Q_CA = {}

    nErrorCalls = 0
    tErrorTime = 0

    nGradCalls = 0
    tGradTime = 0

    nCache_Calls = 0
    tCache_Time = 0

    def static_init(pblm, epsilon = 0.025):
        Reflex_Collision.pblm = pblm
        Reflex_Collision.epsilon = epsilon
        Reflex_Collision.gripper = Reflex_Collision.generate_gripper()
        for key in pblm.Env_List.keys():
            Reflex_Collision.Q_CA[key] = (np.array([]), np.array([]), np.array([]))

        Reflex_Collision.flange_radii = 0.06 + 0.03
        Reflex_Collision.flrist_radii = 0.08 + 0.03
        Reflex_Collision.wrist_radii = 0.10 + 0.03
        
    
    def __init__(self, EKey, idx):
        self.EKey = EKey
        self.idx = idx
        self._cname = "Robot_Coll_" + str(EKey)
        ExecLogger.__init__(self, [self._cname])
        self.Wt_G = int(self.pblm.Wt_G > 0)
        self.Wt_P = int(self.pblm.Wt_P > 0)
        self.sbuff = 5
        if idx == 0:
            self.always_active=True

    def generate_gripper():
        #Use rospack find to generate the full path
        Pkg_Path = "/home/mohanraj/robot_WS/src/gpu_sd_map"
        Base_BV_Path = Pkg_Path + "/gripper_augmentation/base.binvox" 
        Finger_BV_Path = Pkg_Path + "/gripper_augmentation/finger_sweep.binvox"
        
        Base = GripperVox(Base_BV_Path)
        Finger1 = GripperVox(Finger_BV_Path)
        Finger2 = GripperVox(Finger1)

        #Base.print_metadata()
        #Finger2.print_metadata()

        Gripper = ReflexAug(Base, Finger1)
        return Gripper

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
        nCache_Calls = 0
        tCache_Time = 0
        
    @classmethod
    def cache_performance_stats(cls):
        print(f"Cache calls: {cls.nCache_Calls}, taking {cls.tCache_Time}s")

    @classmethod
    def get_gripper_points(cls, preshape = 0, ignore_fingers=False):
        cls.flange_pt = [-0.125, 0.0, 0.0]
        cls.flrist_pt = [-0.16, 0.0, 0.0] # Rougly middle but closer to flange (smaller)
        cls.wrist_pt = [-0.245, 0.0, 0.0]
        return cls.gripper.Set_Pose_Preshape(preshape = preshape, ignore_fingers=ignore_fingers)

        #return np.vstack([gripper_pts, flange_pt, wrist_pt])

    @classmethod
    def validate_grasp_pose(cls, tTg, Env_List):
        Reflex_Collision.gripper = Reflex_Collision.generate_gripper()
        Reflex_Collision.flange_radii = 0.06 + 0.02
        Reflex_Collision.wrist_radii = 0.10 + 0.02
        
        gripper_pts = cls.get_gripper_points()
        gripper_pts = np.vstack([gripper_pts, cls.flange_pt, cls.wrist_pt])
        query_pts = transform_points(gripper_pts, tTg)
        min_sd = 100
        for key in Env_List:
            sd, dsd = Env_List[key].query_points(query_pts)
            min_sd = min(min(sd), min_sd)
            if min_sd < 0:
                return False
        return True
        
    def cache_(self, q):
        type(self).nCache_Calls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        
        x_p = self.pblm.joint_to_place(q)
        tTg = vector2mat(x_p[:3], x_p[3:])

        x_g = self.pblm.joint_to_grasp(q)
        gTo = vector2matinv(x_g[:3], x_g[3:])

        tTo = tTg @ gTo

        gripper_pts = type(self).get_gripper_points(preshape = q[self.pblm.PRESHAPE])
        gripper_pts = np.vstack([gripper_pts, \
                                 type(self).flrist_pt, \
                                 type(self).flange_pt, \
                                 type(self).wrist_pt])

        query_pts = transform_points(gripper_pts, tTg)
        sd, dsd = self.pblm.Env_List[self.EKey].query_points(query_pts)
        sd[-3] -= type(self).flrist_radii
        sd[-2] -= type(self).flange_radii
        sd[-1] -= type(self).wrist_radii

        dl = self.pblm.Env_List[self.EKey].max_tsdf() - self.sbuff
        
        csd = -chomp_smooth(sd-dl, self.sbuff) + dl
        csd_g = -chomp_smooth_grads(sd-dl, self.sbuff)
        
        #sd -=5
        #sd *= 1e-3
        q_grads_p = self.pblm.gripper_position_grads(q, gripper_pts)[0]
        q_grads_pre = self.gripper.Get_Derivate_Preshape([0.0, 0.0, 0.0], q[self.pblm.PRESHAPE])
        q_grads_pre = np.vstack([q_grads_pre, np.zeros(3), np.zeros(3), np.zeros(3)])
        #print(q_grads_p.shape, q_grads_pre.shape)
        q_grads = np.concatenate((q_grads_p, np.expand_dims(q_grads_pre,2)), axis=2)

        cdsd = np.einsum('i,ij->ij', csd_g, dsd)
        grads = np.einsum('ij,ijk->ik',cdsd, q_grads)
        #grads = np.einsum('ij,ijk->ik',dsd, q_grads)

        type(self).Q_CA[self.EKey] = (q, csd*1e-3, -grads, dsd)
        #type(self).Q_CA[self.EKey] = (q, sd, grads)
        #type(self).Q_CA[self.EKey][1].SD = sd
        #type(self).Q_CA[self.EKey][2].Grad = grads

        type(self).tCache_Time += time.perf_counter() - time0


    def error(self, q):
        type(self).nErrorCalls += 1
        time0 = time.perf_counter()
        res = self.error_core(q)
        self.log_val(self._cname, q.tobytes(), res)
        type(self).tErrorTime += time.perf_counter() - time0
        return res
        
    def error_core(self, q):
        q = self.pblm._process_q(q)
        if not np.array_equal(self.Q_CA[self.EKey][0], q):
            self.cache_(q)
        idx = np.argmin(self.Q_CA[self.EKey][1])
        return self.epsilon - self.Q_CA[self.EKey][1][idx]

    def error_gradient(self, q):
        type(self).nGradCalls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)
        if not np.array_equal(self.Q_CA[self.EKey][0], q):
            self.cache_(q)
        idx = np.argmin(self.Q_CA[self.EKey][1])
        q_grads[0, self.pblm.P_IDX] = self.Q_CA[self.EKey][2][idx][:7] * self.Wt_P
        q_grads[0, self.pblm.PRESHAPE] = self.Q_CA[self.EKey][2][idx][7] * self.Wt_G
        GradCheck = not True
        if GradCheck:
            fd_grads = self.error_gradient_fd(q)
            fd_grads[0, self.pblm.G_IDX] *= self.Wt_G
            fd_grads[0, self.pblm.P_IDX] *= self.Wt_P
            #np.set_printoptions(precision=2, floatmode='maxprec_equal')
            #print('fd vs a_grads error: \n', fd_grads, q_grads, fd_grads - q_grads)
            compare_grads(fd_grads, q_grads.T)
        type(self).tGradTime += time.perf_counter() - time0
        return q_grads.T

    def error_gradient_fd(self, q, delt=1e-3):
        #delt = 1e-5
        print("FD grads called")
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        q = q.reshape(-1) # Change input to row vector
        g = np.matrix(q * 0.0)
        for i in range(self.pblm.dof*2 + 1):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            cf = self.error(q_new)
            
            q_new = copy.deepcopy(q)
            q_new[i] -= delt
            cb = self.error(q_new)

            g[0, i] = (cf - cb) / (2*delt)
        #g = g / np.linalg.norm(g)
        return g


class Pick_For_Problem(Problem):
    def __init__(self, obj_grid, env_list, x_p, grasp_client):
        super().__init__()
        self.Wt_G = 1
        self.Wt_P = 0
        self.tTo = vector2mat(x_p[:3], x_p[3:])
        self.place_frame = 'place_table_corner'
        self.Env_List = env_list

        self.grasp_kdl = ManipulatorKDL(robot_description=ROBOT_DESCRIPTION)
        self.arm_palm_KDL = ManipulatorKDL(robot_description=ROBOT_DESCRIPTION)
        
        self.dof = self.arm_palm_KDL.get_dof()
        self.G_IDX = slice(0, self.dof)
        self.PRESHAPE = slice(self.dof, self.dof+1)
        assert isinstance(obj_grid, ObjGrid), "Param obj_grid is not of ObjGrid type"
        obj_grid.test_cornered()
        print("Unpacking obj_grid")
        self.Obj_Grid = obj_grid
        self.object_sparse_grid = obj_grid.sparse_grid
        self.object_voxel_dims = obj_grid.grid_size
        self.object_size = obj_grid.true_size
        self.x0 = np.zeros((self.dof+1, 1))
        self.initial_solution = self.x0

        joint_lows, joint_highs = self.arm_palm_KDL.get_joint_limits()

        for i_ in range(self.dof):
            self.min_bounds[i_] = joint_lows[i_ % self.dof]
            self.max_bounds[i_] = joint_highs[i_ % self.dof]

        self.min_bounds[self.dof] = 0.0
        self.max_bounds[self.dof] = 1.57

        try:
            self.obj_world_mat = get_tf_mat("object_pose", "world")
            self.ptable_world_mat = get_tf_mat(self.place_frame, "world")
        except:
            print("Warning states not found")

        self.GraspClient = grasp_client
        #input("check smi")
        #self.GraspClient.init_grasp_nets(self.object_sparse_grid, \
                                         #self.object_voxel_dims, \
                                         #self.object_size)
        self.Wt_Post = [1., 0.5]
        self.initialize_grasp(idx=1)


    def size(self):
        return 8

    def add_bounds_as_ieqs(self):
        for i in range(self.size()):
            self.inequality_constraints.append(Bounds_IEq(i, self.min_bounds[i], self))
            self.inequality_constraints.append(Bounds_IEq(i, self.max_bounds[i], self, True))

    def _process_q(self, q):
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        return q.reshape(-1) # Change input to row vector

    def joint_to_grasp(self, q, asmat=False):
        """
        Return fk of grasp joints in the object frame
        """
        q = self._process_q(q)
        x = self.grasp_kdl.fk(q[self.G_IDX])
        if asmat:
            return TransformPoseMat(x, self.obj_world_mat)
        else:
            return mat_to_vec(TransformPoseMat(x, self.obj_world_mat))


    def grasp_probabilities(self, q):
        q = self._process_q(q)

        x_g = self.joint_to_grasp(q)

        grasp_conf = np.asarray(list(x_g) + list(q[self.PRESHAPE]), dtype=NP_DTYPE_)
        grasp_probs = self.GraspClient.get_grasp_probs(grasp_conf, self.Wt_Post)
        return grasp_probs
    
    def grasp_grads(self, q, idx=0):

        q = self._process_q(q)
        
        x_g_ = self.joint_to_grasp(q)

        grasp_conf = np.asarray(list(x_g_) + list(q[self.PRESHAPE]), dtype=NP_DTYPE_)
        grasp_grads_ = self.GraspClient.get_grasp_grads(grasp_conf, self.Wt_Post)
        
        if idx == 2:
            grasp_grads_ = self.GraspClient.get_grasp_prior_grad(grasp_conf)

        elif idx == 1:
            grasp_grads_ = self.GraspClient.get_grasp_lkh_grad(grasp_conf)
            
        oRw, oPw = decompose_transform(self.obj_world_mat)
        
        jac_g = self.grasp_kdl.jacobian(q[self.G_IDX])
        omega_g = jac_g[3:]
        linvel_g = jac_g[:3]

        o_omega_g = oRw @ batch_vector_skew_mat(omega_g) @ oRw.T
        o_rpy_jac = []
        for o_omega in o_omega_g[:]:
            o_omega_v = np.array([o_omega[2][1], o_omega[0][2], o_omega[1][0]])
            rpy_dot = angular_velocity_to_rpy_dot(o_omega_v, x_g_[3:])
            o_rpy_jac.append(rpy_dot)
        o_rpy_jac = np.vstack(o_rpy_jac).T
        o_jac_g = np.vstack([oRw @ linvel_g, o_rpy_jac])
        grasp_grads_qg = grasp_grads_[:6] @ o_jac_g

        return grasp_grads_qg, grasp_grads_[6]


    def cost(self, q, split=False):
        q = self._process_q(q)

        grasp_probs = self.grasp_probabilities(q)
        grasp_cost = grasp_probs[0]
        
        return grasp_cost

    def cost_gradient(self, q):
        q = self._process_q(q)

        x_g_ = self.joint_to_grasp(q)
        
        q_grads = np.matrix(q*0.0)
        
        grasp_conf = np.asarray(list(x_g_) + list(q[self.PRESHAPE]), dtype=NP_DTYPE_)
        grasp_grads_ = self.GraspClient.get_grasp_grads(grasp_conf, self.Wt_Post)

        oRw, oPw = decompose_transform(self.obj_world_mat)
        
        jac_g = self.grasp_kdl.jacobian(q[self.G_IDX])
        omega_g = jac_g[3:]
        linvel_g = jac_g[:3]

        o_omega_g = oRw @ batch_vector_skew_mat(omega_g) @ oRw.T
        o_rpy_jac = []
        for o_omega in o_omega_g[:]:
            o_omega_v = np.array([o_omega[2][1], o_omega[0][2], o_omega[1][0]])
            rpy_dot = angular_velocity_to_rpy_dot(o_omega_v, x_g_[3:])
            o_rpy_jac.append(rpy_dot)
        o_rpy_jac = np.vstack(o_rpy_jac).T
        o_jac_g = np.vstack([oRw @ linvel_g, o_rpy_jac])
        
        grasp_grads_qg = grasp_grads_[:6] @ o_jac_g
        
        q_grads[0, self.G_IDX] += grasp_grads_qg
        q_grads[0, self.PRESHAPE] += grasp_grads_[6]

        GradCheck = False
        if GradCheck:
            fd_grads = self.cost_gradient_fd(q)
            compare_grads(fd_grads, q_grads)
        return q_grads.T

    def cost_gradient_fd(self, q, delt=1e-5):
        print("warning cost gradient fd called")
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        q = q.reshape(-1) # Change input to row vector
        g = np.matrix(q * 0.0)
        for i in range(self.dof + 1):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            cf = self.cost(q_new)
            
            q_new = copy.deepcopy(q)
            q_new[i] -= delt
            cb = self.cost(q_new)

            g[0, i] = (cf - cb) / (2*delt)
        #g = g / np.linalg.norm(g)
        return g.T

    def add_collision_constraints(self):
        Reflex_Collision.static_init(self)
        Grasp_Collision.static_init(self)
        #obj_pts = self.Obj_Grid.binned_pts#_raw_grid_idx_to_pts()
        #gripper_pts = Reflex_Collision.get_gripper_points()
        #publish_as_point_cloud(gripper_pts, topic='reflex_binned', frame_id = 'reflex_palm_link')
        #num_pts = obj_pts.shape[0]
        #print(f"Adding {num_pts} object collision checks per object")
        #num_gpts = gripper_pts.shape[0]
        #print(f"Adding {num_gpts} robot collision checks per object")
        #for j in range(num_gpts):
        self.inequality_constraints.append(Grasp_Collision(0))
        #break
        return
        if self.Wt_P:
            for key in self.Env_List.keys():
                for i in range(0,num_pts, 1):
                    self.inequality_constraints.append(Collision_Constraint(key, i))
                    #if key in ['ball', 'pringle']:
                    #    self.inequality_constraints[-1].always_active = False
                    break
                for j in range(num_gpts):
                    self.inequality_constraints.append(Reflex_Collision(key, j))
                    #if key in ['ball', 'pringle']:
                    #    self.inequality_constraints[-1].always_active = False
                    break

    def initialize_grasp(self, idx=None, max_trials = 100):
        self.x0 = self.x0.reshape(-1)
        for iter_ in range(max_trials):
            if idx is None:
                grasp_prior_ = self.GraspClient.sample_prior()
            else:
                grasp_prior_ = self.GraspClient.sample_explicit(idx)
            self.raw_prior = grasp_prior_[:6]

            prior_cost_ = self.GraspClient.get_grasp_prior(grasp_prior_)
            
            grasp_prior_pose_ = convert_array_to_pose(grasp_prior_[:6], "object_pose")
            world_grasp_pose_ = TransformPose(grasp_prior_pose_.pose, "world", "object_pose")
            g_trans_, g_quat_ = pose_to_array(world_grasp_pose_)
            self.xt = pose_to_6dof(world_grasp_pose_)
            qg_0_ = get_IK(world_grasp_pose_, 'left')
            if qg_0_ is None:
                print("Unreachable grasp, retrying...")
            else:
                qg_0_f_ = np.arctan2(np.sin(qg_0_), np.cos(qg_0_))
                qg_0_ = qg_0_f_
                self.x0[self.G_IDX] = qg_0_[:]
                self.x0[self.PRESHAPE] = grasp_prior_[6]
                xg_0_ = self.joint_to_grasp(self.x0)
                _, lkh, prior_cost_fk_ = self.GraspClient.get_grasp_probs(grasp_prior_, self.Wt_Post)
                print("Cost at Prior: {}\n After IK: {}\n Error: {}".format(prior_cost_, prior_cost_fk_, prior_cost_ - prior_cost_fk_))
                if (prior_cost_fk_ < -3):
                    self.x0 = self.x0.reshape(-1,1)
                    return
                else:
                    print('Found feasible but bad grasp, retrying...')
        if qg_0_ is None:
            print("Unable to generate reachable grasp prior, proceeding with zero joint init")
        self.x0 = self.x0.reshape(-1)


    def gripper_position_grads(self, q, gripper_pts):
        '''
        Verified works
        '''
        q = self._process_q(q)

        tRw, tPw = decompose_transform(self.ptable_world_mat)

        oRw, oPw = decompose_transform(self.obj_world_mat)
        
        wx_g = self.arm_palm_KDL.fk(q[self.G_IDX])
        gPw_inv = np.array(wx_g[:3]).reshape(3,1) 
        gx_g = vector_inverse(wx_g)
        gTw = vector2mat(gx_g[:3], gx_g[3:])
        gRw, gPw = decompose_transform(gTw)

        jac_g = self.grasp_kdl.jacobian(q[self.G_IDX])
        omega_g = jac_g[3:]
        linvel_g = jac_g[:3]

        oJRg_ = oRw @ batch_vector_skew_mat(omega_g) @ gRw.T
        q_grads_g = (oJRg_ @ gripper_pts.T).T + oRw @ linvel_g

        return None, q_grads_g
