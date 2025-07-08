#!/usr/bin/env python3
"""
Pick N Place optimization problem definition for ll4ma_opt suite of solvers.
Extends ll4ma_opt/problem class.
"""

from ll4ma_pick_n_place.grasp_utils_client import GraspUtils
from ll4ma_opt.problems.problem import Problem, Constraint
from gpu_sd_map.gripper_augment import GripperVox, ReflexAug, visualize_points
from gpu_sd_map.environment_manager import EnvVox, ObjGrid
from gpu_sd_map.ros_transforms_lib import convert_array_to_pose, TransformPose, \
    pose_to_array, pose_to_6dof, TransformPoseStamped, get_tf_mat, mat_to_vec, vector_transform, \
    test_3d_rotation_equivalence, dof6_to_dof7_array, get_space_jacobian, get_adjoint_transform, \
    vector_inverse, skew_symmetric_mat, batch_vector_skew_mat, euler_from_mat, publish_as_point_cloud
from gpu_sd_map.transforms_lib import vector2mat, vector2matinv, get_rotation_jacobian, \
    get_rotation_matrix, vector_similarity, invert_trans, jacobian_similarity, \
    angular_velocity_to_rpy_dot, decompose_transform, verify_rpy_dot, transform_points
from ll4ma_pick_n_place.kdl_utils import ManipulatorKDL, pose_difference
from collections import defaultdict
import numpy as np
import tf
import tf2_ros
import tf2_geometry_msgs
import copy
import rospy
import time


def compare_grads(a, b):
    ac = a.reshape(-1,1)
    bc = b.reshape(-1,1)
    np.set_printoptions(precision=2, floatmode='maxprec_equal')
    vecsim = vector_similarity(ac, bc)
    if vecsim != 0:
        print('fd vs a_grads error: \n', np.hstack([ac, bc, ac - bc]))
        print('Similarity: ', np.arccos(vecsim)*180/np.pi, vecsim )

class ExecLogger():
    LogDict = {}
    def __init__(self, keys):
        for k in keys:
            if k not in type(self).LogDict.keys():
                type(self).LogDict[k] = defaultdict(lambda: np.nan)

    def log_val(self, key, inp, val):
        if key in type(self).LogDict.keys() and inp in type(self).LogDict[key].keys():
            type(self).LogDict[key][inp] = np.max([val, type(self).LogDict[key][inp]])
        else:
            type(self).LogDict[key][inp] = val

    def extract_iters(self, key, iters):
        out = []
        for i in iters:
            if isinstance(i, np.ndarray):
                i = i.tobytes()
            out.append(type(self).LogDict[key][i])
        return out
        

class FK_Constraint(Constraint, ExecLogger):
    pblm = None
    epsilon = 10
    EGrid = None
    Q_CA = {}

    nErrorCalls = 0
    tErrorTime = 0

    nGradCalls = 0
    tGradTime = 0

    nCache_Calls = 0
    tCache_Time = 0

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

    def __init__(self, pblm, st_id, idx):
        self.pblm = pblm
        self.cart_grad = np.eye(6)
        self.St_ID = st_id
        self.idx = idx
        axes = {0:'x', 1:'y', 2:'z', 3:'roll', 4:'pit', 5:'yaw'}
        self._cname = "Grasp_Eq_" + axes[idx]
        ExecLogger.__init__(self, [self._cname])

    def error(self, q):
        type(self).nErrorCalls += 1
        time0 = time.perf_counter()
        res = self.error_core(q)
        self.log_val(self._cname, q.tobytes(), res)
        type(self).tErrorTime += time.perf_counter() - time0
        return res
    
    def error_core(self, q):
        self.pblm._process_q(q)
        q_g = q[self.pblm.G_IDX]
        q_p = q[self.pblm.P_IDX]
        x_g = q[self.pblm.Gx_IDX]
        x_p = q[self.pblm.Px_IDX]
        g_fk = self.pblm.grasp_kdl.fk(q_g)
        oTg_fk = self.pblm.obj_world_mat @ vector2mat(g_fk[:3], g_fk[3:])
        oTg = vector2mat(x_g[:3].T, x_g[3:]) #Could be this or FK of grasp joints but this is the true grasp pose hence choosing this
        #tTo = self.pblm.ptable_world_mat @ \
        #       vector2mat(self.pblm.grasp_kdl.FK(q_p)) @ invert_trans(oTg)
        fk_g = mat_to_vec(oTg_fk)
        #fk_p = mat_to_vec(tTo)
        return (fk_g - x_g.T) @ (fk_g - x_g.T).T#, fk_p - x_p

    def error_gradient(self, q):
        type(self).nGradCalls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)

        q_g = q[self.pblm.G_IDX]
        q_p = q[self.pblm.P_IDX]
        x_g = q[self.pblm.Gx_IDX]
        x_p = q[self.pblm.Px_IDX]
        
        qg_grads = self.pblm.grasp_grads(q)
        xg_grads = self.cart_grad

        g_fk = self.pblm.grasp_kdl.fk(q_g)
        oTg_fk = self.pblm.obj_world_mat @ vector2mat(g_fk[:3], g_fk[3:])
        fk_g = mat_to_vec(oTg_fk)
        
        q_grads[0, self.pblm.G_IDX] = 2 * (fk_g - x_g.T) @ qg_grads
        q_grads[0, self.pblm.Gx_IDX] = -2*(fk_g - x_g.T) #-xg_grads[self.idx]

        grad_check =  False
        if grad_check:
            fd_grads = self.error_gradient_fd(q)
            compare_grads(fd_grads, q_grads)
        type(self).tGradTime += time.perf_counter() - time0
        return q_grads.T

    def error_gradient_fd(self, q, delt=1e-5):
        #delt = 1e-5
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        q = q.reshape(-1) # Change input to row vector
        g = np.matrix(q * 0.0)
        for i in range(self.pblm.size()):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            cf = self.error(q_new)
            
            q_new = copy.deepcopy(q)
            q_new[i] -= delt
            cb = self.error(q_new)

            g[0, i] = (cf - cb) / (2*delt)
        #g = g / np.linalg.norm(g)
        return g

        
    
class Collision_Constraint(Constraint, ExecLogger):
    pblm = None
    epsilon = 10
    EGrid = None
    Q_CA = {}

    nErrorCalls = 0
    tErrorTime = 0

    nGradCalls = 0
    tGradTime = 0

    nCache_Calls = 0
    tCache_Time = 0

    def static_init(pblm, epsilon = 10):
        Collision_Constraint.pblm = pblm
        Collision_Constraint.epsilon = epsilon
        for key in pblm.Env_List.keys():
            Collision_Constraint.Q_CA[key] = (np.array([]), np.array([]), np.array([]))
    
    def __init__(self, EKey, idx):
        self.EKey = EKey
        self.idx = idx
        self._cname = "Obj_Coll_" + str(EKey)
        ExecLogger.__init__(self, [self._cname])

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

    def cache_(self, q):
        type(self).nCache_Calls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        
        #x_p = self.pblm.joint_to_place(q)
        #tTg = vector2mat(x_p[:3], x_p[3:])

        #x_g = self.pblm.joint_to_grasp(q)
        #gTo = vector2matinv(x_g[:3], x_g[3:])

        x_p = q[self.pblm.Px_IDX]
        
        tTo = vector2mat(x_p[:3], x_p[3:])#tTg @ gTo

        obj_pts = self.pblm.Obj_Grid.binned_pts

        query_pts = transform_points(obj_pts, tTo)
        sd, dsd = self.pblm.Env_List[self.EKey].query_points(query_pts)

        #dsd = dsd @ tTo[:3, :3]

        JR = get_rotation_jacobian(x_p[0], x_p[1], x_p[2])
        JR = (JR[0] @ obj_pts.T, JR[1] @ obj_pts.T, JR[2] @ obj_pts.T)
        dsd_o = np.vstack([np.einsum('ij, ij -> i', dsd, JR[0].T), \
                           np.einsum('ij, ij -> i', dsd, JR[1].T), \
                           np.einsum('ij, ij -> i', dsd, JR[2].T)])
        #print(dsd_o.shape)

        #sd -= 50
        
        #dsd_fd = np.copy(dsd.T)
        #delt = [0, 0, 0]
        #for i in range(3):
        #    delt = [0, 0, 0]
        #    delt[i] = 1e-2
        #    qp_f = transform_points(obj_pts + delt, tTo)
        #    sdf, _ = self.pblm.Env_List[self.EKey].query_points(qp_f)
        #    qp_b = transform_points(obj_pts - delt, tTo)
        #    sdb, _ = self.pblm.Env_List[self.EKey].query_points(qp_b)
        #    dsd_fd[i] = (sdf - sdb)/2*1e-2

        #print(dsd.shape)
        #print(dsd[self.idx], dsd_fd.T[self.idx], dsd[self.idx] - dsd_fd.T[self.idx])
        #vecsim = vector_similarity(dsd[self.idx], dsd_fd.T[self.idx])
        #print('DSD Similarity: ', np.arccos(vecsim)*180/np.pi, vecsim )

        #q_grads = np.concatenate(self.pblm.place_position_grads(q, obj_pts), axis=2)
        #grads = np.einsum('ij,ijk->ik',dsd_fd.T, q_grads)
        grads = np.hstack([dsd, dsd_o.T])
        #print(grads.shape)

        type(self).Q_CA[self.EKey] = (q, sd, grads)
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
        if not self.Q_CA[self.EKey][0].tobytes() == q.tobytes():
            self.cache_(q)
        idx = np.argmin(self.Q_CA[self.EKey][1])
        return self.epsilon - self.Q_CA[self.EKey][1][idx]

    def error_gradient(self, q):
        type(self).nGradCalls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)
        #if not np.array_equal(self.Q_CA[self.EKey][0], q):
        if not self.Q_CA[self.EKey][0].tobytes() == q.tobytes():
            self.cache_(q)
        idx = np.argmin(self.Q_CA[self.EKey][1])
        q_grads[0, self.pblm.Px_IDX] = self.Q_CA[self.EKey][2][idx]
        #q_grads[0, self.pblm.G_IDX] = self.Q_CA[self.EKey][2][self.idx][7:]

        GradCheck = False
        if GradCheck:
            fd_grads = self.error_gradient_fd(q)
            compare_grads(fd_grads, q_grads)
            #print('Similarity: ', np.arccos(vecsim)*180/np.pi, vecsim )
        
        type(self).tGradTime += time.perf_counter() - time0
        return -q_grads.T

    def error_gradient_fd(self, q, delt=1e-3):
        #delt = 1e-5
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        q = q.reshape(-1) # Change input to row vector
        g = np.matrix(q * 0.0)
        for i in range(self.pblm.size()):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            cf = self.error(q_new)
            
            q_new = copy.deepcopy(q)
            q_new[i] -= delt
            cb = self.error(q_new)

            g[0, i] = (cf - cb) / (2*delt)
        #g = g / np.linalg.norm(g)
        return g

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

    def static_init(pblm, epsilon = 10):
        Reflex_Collision.pblm = pblm
        Reflex_Collision.epsilon = epsilon
        Reflex_Collision.gripper = Reflex_Collision.generate_gripper()
        for key in pblm.Env_List.keys():
            Reflex_Collision.Q_CA[key] = (np.array([]), np.array([]), np.array([]))
    
    def __init__(self, EKey, idx):
        self.EKey = EKey
        self.idx = idx
        self._cname = "Robot_Coll_" + str(EKey)
        ExecLogger.__init__(self, [self._cname])

    def generate_gripper():
        #Use rospack find to generate the full path
        Pkg_Path = "/home/mohanraj/ll4ma_prime_WS/src/gpu_sd_map"
        Base_BV_Path = Pkg_Path + "/gripper_augmentation/base.binvox" 
        Finger_BV_Path = Pkg_Path + "/gripper_augmentation/finger.binvox"
        
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
    def get_gripper_points(cls, preshape = 0):
        return cls.gripper.Set_Pose_Preshape(preshape = preshape)
        
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

        query_pts = transform_points(gripper_pts, tTg)
        sd, dsd = self.pblm.Env_List[self.EKey].query_points(query_pts)
        sd -=5
        q_grads_p = self.pblm.gripper_position_grads(q, gripper_pts)
        q_grads_pre = self.gripper.Get_Derivate_Preshape([0.0, 0.0, 0.0], q[self.pblm.PRESHAPE])
        #print(q_grads_p.shape, q_grads_pre.shape)
        q_grads = np.concatenate((q_grads_p, np.expand_dims(q_grads_pre,2)), axis=2)
        grads = np.einsum('ij,ijk->ik',dsd, q_grads)

        type(self).Q_CA[self.EKey] = (q, sd, grads)
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
        return self.epsilon - self.Q_CA[self.EKey][1][self.idx]

    def error_gradient(self, q):
        type(self).nGradCalls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)
        if not np.array_equal(self.Q_CA[self.EKey][0], q):
            self.cache_(q)
        q_grads[0, self.pblm.P_IDX] = self.Q_CA[self.EKey][2][self.idx][:7]
        q_grads[0, self.pblm.PRESHAPE] = self.Q_CA[self.EKey][2][self.idx][7]
        GradCheck = True
        if GradCheck:
            fd_grads = self.error_gradient_fd(q)
        type(self).tGradTime += time.perf_counter() - time0
        return fd_grads.T

    def error_gradient_fd(self, q, delt=1e-3):
        #delt = 1e-5
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

class Grasp_Constraint(Constraint, ExecLogger):

    nErrorCalls = 0
    tErrorTime = 0

    nGradCalls = 0
    tGradTime = 0
    
    def __init__(self, pblm, grasp_bound=0.15):
        self.pblm = pblm
        self.grasp_bound = grasp_bound
        self._cname = "Grasp_Bound_"
        ExecLogger.__init__(self, [self._cname])

    @classmethod
    def performance_stats(cls):
        return [(cls.nErrorCalls, cls.tErrorTime), \
                (cls.nGradCalls, cls.tGradTime)]

    @classmethod
    def reset_performance_stats(cls):
        cls.nErrorCalls = 0
        cls.tErrorTime = 0
        cls.nGradCalls = 0
        cls.tGradTime = 0


    def error(self, q):
        type(self).nErrorCalls += 1
        time0 = time.perf_counter()
        res = self.error_core(q)
        self.log_val(self._cname, q.tobytes(), res)
        type(self).tErrorTime += time.perf_counter() - time0
        return res
        
    def error_core(self, q):
        x_g = np.array(self.pblm.joint_to_grasp(q)[:3])
        return x_g @ x_g.T - (0.05)**2
        #grasp_cost = -self.pblm.grasp_probabilities(q)[0]
        #return grasp_cost - self.grasp_bound

    def error_gradient(self, q):
        type(self).nGradCalls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)

        x_g = np.array(self.pblm.joint_to_grasp(q)[:3])

        oRw, oPw = decompose_transform(self.pblm.obj_world_mat)
        
        jac_g = self.pblm.grasp_kdl.jacobian(q[self.pblm.G_IDX])
        omega_g = jac_g[3:]
        linvel_g = jac_g[:3]
        
        #qg_grads, qpre_grads = self.pblm.grasp_grads(q)
        q_grads[0, self.pblm.G_IDX] = 2*x_g @ oRw @ linvel_g #qg_grads
        #q_grads[0, self.pblm.PRESHAPE] = qpre_grads
        
        type(self).tGradTime += time.perf_counter() - time0

        #fd_grads = self.error_gradient_fd(q)

        #np.set_printoptions(precision=2, floatmode='maxprec_equal')
        #print('fd vs a_grads error: \n', fd_grads, q_grads, fd_grads - q_grads)
        #vecsim = vector_similarity(fd_grads, q_grads.T)
        #print('Similarity: ', np.arccos(vecsim)*180/np.pi, vecsim )
        
        return q_grads.T

    def error_gradient_fd(self, q, delt=1e-3):
        #delt = 1e-5
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

class Placement_Constraint(Constraint, ExecLogger):

    nErrorCalls = 0
    tErrorTime = 0

    nGradCalls = 0
    tGradTime = 0
    
    def __init__(self, idx, pblm, upper_bound=False):
        self.idx = idx
        self.pblm = pblm
        self.upper_bound = upper_bound
        axes = {0:'x', 1:'y', 2:'z', 3:'yaw'}
        self._cname = "Place_Bound_" + axes[idx]
        if upper_bound:
            self._cname += "_u"
        else:
            self._cname += "_l"
        #self.place_kdl = ManipulatorKDL()
        #self.grasp_kdl = ManipulatorKDL()
        #self.tJw = get_space_jacobian(pblm.ptable_world_mat)
        #self.oJw = get_space_jacobian(pblm.obj_world_mat)
        ExecLogger.__init__(self, [self._cname])
        self.wTo = invert_trans(pblm.obj_world_mat)
        self.upright = self.wTo[2, 2] #euler_from_mat(pblm.obj_world_mat)
        self.x_p_grad = np.zeros((1,6))
        self.x_p_grad[0, self.idx] = 1.0
        self.POS_CONST_L = [0.0, 0.0, self.pblm.object_size[2]/2] #-self.pblm.obj_world_mat[3,2] - 0.59]
        self.POS_CONST_H = [self.pblm.table_size[0], self.pblm.table_size[1], self.pblm.object_size[2]/2 + 0.1]
                            #-self.pblm.obj_world_mat[3,2] - 0.59 + 0.02]
        self.cache = (np.array([]), np.nan, np.nan)

    @classmethod
    def performance_stats(cls):
        return [(cls.nErrorCalls, cls.tErrorTime), \
                (cls.nGradCalls, cls.tGradTime)]

    @classmethod
    def reset_performance_stats(cls):
        cls.nErrorCalls = 0
        cls.tErrorTime = 0
        cls.nGradCalls = 0
        cls.tGradTime = 0
    

    def error(self, q):
        type(self).nErrorCalls += 1
        time0 = time.perf_counter()
        res = self.error_core(q)
        self.log_val(self._cname, q.tobytes(), res)
        type(self).tErrorTime += time.perf_counter() - time0
        return res
    
    def error_core(self, q):
        q = self.pblm._process_q(q)
        
        #x_p_ = self.pblm.joint_to_place(q)
        #tTg_ = vector2mat(x_p_[:3], x_p_[3:])

        #x_g_ = self.pblm.joint_to_grasp(q)
        #gTo_ = vector2matinv(x_g_[:3], x_g_[3:])

        #tTo_ = tTg_ @ gTo_
        if np.array_equal(q, self.cache[0]):
            tx_p_ = self.cache[1]
            tTo_ = self.cache[2]
        else:
            tx_p_ = q[self.pblm.Px_IDX]#self.pblm.object_pose_from_joints(q)#mat_to_vec(tTo_)
            tTo_ = vector2mat(tx_p_[:3], tx_p_[3:])
            self.cache = (q, tx_p_, tTo_)

        if self.idx < 3:
            if self.upper_bound:
                return tx_p_[self.idx] - self.POS_CONST_H[self.idx] #x_p[id]<0
            else:
                return self.POS_CONST_L[self.idx] - tx_p_[self.idx] #x_p[id]<0
        else:
            return (tTo_[2, 2] - self.upright) ** 2 - 1e-6

    def p_jac_fd(self, q, delt = 1e-5):
        x_c = self.pblm.joint_to_place(q)
        x_c = np.array(x_c)
        j = []
        for i in range(self.pblm.dof):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            x_f = self.pblm.joint_to_place(q_new)
            x_f = np.array(x_f)
            v = (x_f - x_c) / delt
            v = v.reshape(6,1)
            j.append(v)
        return np.hstack(j)

    def g_jac_fd(self, q, delt = 1e-5):
        x_c = self.pblm.joint_fk_se3(q)
        x_c = np.array(vector_inverse(x_c))
        j = []
        for i in range(self.pblm.dof):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            x_f = self.pblm.joint_fk_se3(q_new)
            x_f = np.array(vector_inverse(x_f))
            v = (x_f - x_c) / delt
            v = v.reshape(6,1)
            j.append(v)
        return np.hstack(j)

    def error_gradient(self, q):
        type(self).nGradCalls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)

        if self.idx < 3:
            q_grads_p, q_grads_g = self.pblm.place_position_grads(q)
            q_grads[0, self.pblm.P_IDX] = q_grads_p[self.idx]
            q_grads[0, self.pblm.G_IDX] = q_grads_g[self.idx]

            type(self).tGradTime += time.perf_counter() - time0
            if self.upper_bound:
                return q_grads.T
            else:
                return -q_grads.T
        else:
            tx_p_ = q[self.pblm.Px_IDX]#self.pblm.object_pose_from_joints(q)
            tTo = vector2mat(tx_p_[:3], tx_p_[3:])
            cart_grad = 2 * (tTo[2, 2] - self.upright)# * (-1/(1-tTo[2, 2]**2)**0.5)
            #2 * (tTo[2, 2] - self.wTo[2, 2]) #(tx_p_[self.idx] - self.upright[self.idx-3])
            #q_grads_p, q_grads_g = self.pblm.place_orientation_grads(q)
            q_grads[0, self.pblm.Px_IDX][0,3] = cart_grad * -np.sin(tx_p_[3]) * np.cos(tx_p_[4])
            q_grads[0, self.pblm.Px_IDX][0,4] = cart_grad * -np.sin(tx_p_[4]) * np.cos(tx_p_[3])#* q_grads_p
            #q_grads[0, self.pblm.G_IDX] = cart_grad * q_grads_g
            #fd_grads = self.error_gradient_fd(q)
        grad_check = False
        if grad_check:
            fd_grads = self.error_gradient_fd(q)
            compare_grads(fd_grads, q_grads)
        type(self).tGradTime += time.perf_counter() - time0
        return q_grads.T

    def error_gradient_fd(self, q, delt=1e-5):
        #delt = 1e-5
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        q = q.reshape(-1) # Change input to row vector
        g = np.matrix(q * 0.0)
        for i in range(self.pblm.size()):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            cf = self.error(q_new)
            
            q_new = copy.deepcopy(q)
            q_new[i] -= delt
            cb = self.error(q_new)

            g[0, i] = (cf - cb) / (2*delt)
        #g = g / np.linalg.norm(g)
        return g

class PnP_Problem_CS(Problem, ExecLogger):

    nCostCalls = 0
    tCostTime = 0

    nGradCalls = 0
    tGradTime = 0
    
    def __init__(self, obj_grid):
        '''
        PnP_Problem in joint configuration space.
        '''
        super().__init__()
        ExecLogger.__init__(self, ["post", "prior", "lkh", "place", "reg_p", "reg_g"])
        self.Env_List = {}
        self.place_kdl = ManipulatorKDL()
        self.grasp_kdl = ManipulatorKDL()
        self.arm_palm_KDL = ManipulatorKDL()
        self.dof = self.arm_palm_KDL.get_dof()

        self.P_IDX = slice(0, self.dof) # 7 - dof for arm
        self.G_IDX = slice(self.dof, self.dof * 2) # 7 - dof for arm
        self.PRESHAPE = slice(self.dof * 2, self.dof*2 + 1) # 1 - dof for preshape
        self.Px_IDX = slice(2 * self.dof + 1, 2 * self.dof + 7) # 6 - dof for place pose
        self.Gx_IDX = slice(2 * self.dof + 7, 2 * self.dof + 13) # 6 - dof for grasp_pose

        self.Wt_P = 10.0
        self.Wt_G = 1.0
        self.Wt_R = 1.0

        self.Wt_Post = [1., 5.]

        self.table_corner = [0.0, 0.0]
        self.table_size = [1.2, 0.6]
        self.upright = [0.0, 0.0]
        
        self.GraspClient = GraspUtils()
        assert isinstance(obj_grid, ObjGrid), "Param obj_grid is not of ObjGrid type"
        obj_grid.test_cornered()
        print("Unpacking obj_grid")
        self.Obj_Grid = obj_grid
        self.object_sparse_grid = obj_grid.sparse_grid
        self.object_voxel_dims = obj_grid.grid_size
        self.object_size = obj_grid.true_size

        print("setting bounds")
        self.set_bounds()

        self.x0 = np.zeros((self.size(), 1))
        self.initial_solution = self.x0
        print("Waiting for tf states")
        self.tf_b = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_b)
        self.obj_world_mat = get_tf_mat("object_pose", "world")
        self.ptable_world_mat = get_tf_mat("place_table_corner", "world")
        print("Setting initialization")
        self.initialize_grasp()
        print("Acquired grasp initialization")
        self.initialize_place()
        print("Values initialized")

        xt = self.arm_palm_KDL.fk(self.x0[self.G_IDX])
        xt = convert_array_to_pose(xt, "world")
        xt = TransformPoseStamped(xt, "object_pose", self.tf_b, self.tf_listener)
        self.xt = pose_to_6dof(xt)
        #self.inequality_constraints.append(Grasp_Constraint(self))
        #self.inequality_constraints.append(Placement_Constraint(0, self))
        #self.inequality_constraints.append(Placement_Constraint(0, self, True))
        #self.inequality_constraints.append(Placement_Constraint(1, self))
        #self.inequality_constraints.append(Placement_Constraint(1, self, True))
        #self.inequality_constraints.append(Placement_Constraint(2, self))
        #self.inequality_constraints.append(Placement_Constraint(2, self, True))
        self.inequality_constraints.append(Placement_Constraint(3, self))
        self.equality_constraints.append(FK_Constraint(self, 0, 0))
        #self.equality_constraints.append(FK_Constraint(self, 0, 1))
        #self.equality_constraints.append(FK_Constraint(self, 0, 2))
        #self.equality_constraints.append(FK_Constraint(self, 0, 3))
        #self.equality_constraints.append(FK_Constraint(self, 0, 4))
        #self.equality_constraints.append(FK_Constraint(self, 0, 5))
        self.reset_performance_stats()
        #self.inequality_constraints.append(Placement_Constraint(4, self))
    
    def size(self):
        return 27

    def _process_q(self, q):
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        return q.reshape(-1) # Change input to row vector

    @classmethod
    def reset_performance_stats(cls):
        cls.nCostCalls = 0
        cls.tCostTime = 0
        cls.nGradCalls = 0
        cls.tGradTime = 0
        Grasp_Constraint.reset_performance_stats()
        Placement_Constraint.reset_performance_stats()
        Collision_Constraint.reset_performance_stats()
        Reflex_Collision.reset_performance_stats()
        FK_Constraint.reset_performance_stats()

    @classmethod
    def performance_stats(cls):
        stats = [(cls.nCostCalls, cls.tCostTime), \
                 (cls.nGradCalls, cls.tGradTime)]
        stats = stats + Grasp_Constraint.performance_stats()
        stats = stats + Placement_Constraint.performance_stats()
        stats = stats + Collision_Constraint.performance_stats()
        stats = stats + Reflex_Collision.performance_stats()
        return stats

    def set_bounds(self):
        '''
        TODO: verify if pose bounds are actually -pi to pi
        '''
        joint_lows, joint_highs = self.arm_palm_KDL.get_joint_limits()
        place_pose_bounds = [(self.table_corner[0], self.table_size[0]), \
                             (self.table_corner[1], self.table_size[1]), \
                             (self.object_size[2]/2, self.object_size[2]/2 + 0.05), \
                             (-np.pi, np.pi), \
                             (-np.pi, np.pi), \
                             (-np.pi, np.pi)]
        grasp_pose_bounds = [(-self.object_size[0]/2 - 0.3, self.object_size[0]/2 + 0.3), \
                             (-self.object_size[1]/2 - 0.3, self.object_size[1]/2 + 0.3), \
                             (-self.object_size[2]/2, self.object_size[2] + 0.3), \
                             (-np.pi, np.pi), \
                             (-np.pi, np.pi), \
                             (-np.pi, np.pi)]
        for i_ in range(self.dof * 2):
            self.min_bounds[i_] = joint_lows[i_ % self.dof]
            self.max_bounds[i_] = joint_highs[i_ % self.dof]

        self.min_bounds[self.dof * 2] = 0.0
        self.max_bounds[self.dof * 2] = 1.57

        for i_ in range(self.dof*2+1, self.dof*2+7):
            self.min_bounds[i_], self.max_bounds[i_] = place_pose_bounds[i_ - (self.dof*2+1)]
        for i_ in range(self.dof*2+7, self.dof*2+13):
            self.min_bounds[i_], self.max_bounds[i_] = grasp_pose_bounds[i_ - (self.dof*2+7)]

    def joint_fk_se3(self, q, ref_frame="world"):
        q = self._process_q(q)
        assert len(q) == self.dof, "Joint array expected of length {}, got length {} instead".format(self.dof, len(q)) 
        x = self.arm_palm_KDL.fk(q)
        x_pose = convert_array_to_pose(x, "world")
        x_pose = TransformPoseStamped(x_pose, ref_frame, self.tf_b, self.tf_listener)
        return pose_to_6dof(x_pose)

    def joint_to_grasp(self, q):
        """
        Return fk of grasp joints in the object frame
        """
        self._process_q(q)
        return self.joint_fk_se3(q[self.G_IDX], "object_pose")
    
    def joint_to_place(self, q):
        """
        Return fk of grasp joints in the placement reference frame
        """
        self._process_q(q)
        return self.joint_fk_se3(q[self.P_IDX], "place_table_corner")

    def grasp_probabilities(self, q):
        q = self._process_q(q)

        x_g = q[self.Gx_IDX] #self.joint_to_grasp(q)
        grasp_probs = self.GraspClient.query_grasp_probabilities(x_g, \
                                                       q[self.PRESHAPE], \
                                                       self.object_sparse_grid, \
                                                       self.object_voxel_dims, \
                                                       self.object_size)
        return grasp_probs

    def grasp_grads(self, q):

        q = self._process_q(q)
        
        x_g_ = self.joint_to_grasp(q)
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

        return o_jac_g
    
    def corner_heuristic(self, x_p):
        cost = (x_p[0] - self.table_corner[0])**2 + (x_p[1] - self.table_corner[1])**2 #+ \
        #100 * (self.upright[0] - x_p[3])**2 + 100 * (self.upright[1] - x_p[4])**2
        return cost.item()

    def corner_heuristic_grads(self, x_p):
        cart_grad_ = np.zeros((1,6))
        #grads_p, grads_g = self.place_position_grads(q)
        cart_grad_[0,0] = 2 * (x_p[0] - self.table_corner[0]) #* x_p_[1]**2
        cart_grad_[0,1] = 2 * (x_p[1] - self.table_corner[1])#* x_p_[0]**2
        #cart_grad_[0,2] = -2 * 100 * (self.object_size[1] - x_p[2])
        #cart_grad_[0,3] = -2 * 100 * (self.upright[0] - x_p[3])
        #cart_grad_[0,4] = -2 * 100 * (self.upright[1] - x_p[4])
        return cart_grad_ #@ grads_p, cart_grad_ @ grads_g

    def joints_reg(self, q):
        q_p = q[self.P_IDX]
        q_g = q[self.G_IDX]
        return (q_p.T @ q_p).item(), (q_g.T @ q_g).item()
    def joints_reg_grads(self, q):
        return 2*q[self.P_IDX], 2*q[self.G_IDX]
    def cost_IK(self, q, grad=False):
        xc = self.arm_palm_KDL.fk(q[self.G_IDX])
        xc = convert_array_to_pose(xc, "world")
        xc = TransformPoseStamped(xc, "object_pose", self.tf_b, self.tf_listener)
        xc = pose_to_6dof(xc)
        xc4 = dof6_to_dof7_array(xc)
        xt = [0.0, 0.0, 0.2] + [0.0, -1.57, 0.0]
        xt4 = dof6_to_dof7_array(xt)
        diff = pose_difference(xt4, xc4)
        diff = np.array(diff) 
        #print(diff)
        if grad:
            return diff
        return diff @ diff.T
    
    def cost(self, q, split=False):
        type(self).nCostCalls += 1
        time0 = time.perf_counter()
        
        self._process_q(q)

        tx_p_ = q[self.Px_IDX] #mat_to_vec(tTo_)
        place_cost = self.corner_heuristic(tx_p_)
        self.log_val("place", q.tobytes(), place_cost)
        grasp_probs = self.grasp_probabilities(q)
        self.log_val("lkh", q.tobytes(), -grasp_probs[1])
        self.log_val("prior", q.tobytes(), -grasp_probs[2])
        reg_cost_ = self.joints_reg(q)
        #print(reg_cost_)
        self.log_val("reg_p", q.tobytes(), reg_cost_[0])
        self.log_val("reg_g", q.tobytes(), reg_cost_[1])
        grasp_cost = -(self.Wt_Post[0] * grasp_probs[1] + self.Wt_Post[1] * grasp_probs[2])
        self.log_val("post", q.tobytes(), grasp_cost)
        type(self).tCostTime += time.perf_counter() - time0
        if split:
            return grasp_cost , place_cost
        return place_cost * self.Wt_P + grasp_cost * self.Wt_G + \
            reg_cost_[0] * self.Wt_R + reg_cost_[1] * self.Wt_R #Grasp cost is already -log

    def cost_gradient_fd(self, q, delt=1e-5):
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        q = q.reshape(-1) # Change input to row vector
        g = np.matrix(q * 0.0)
        for i in range(self.size()):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            cf = self.cost(q_new)
            
            q_new = copy.deepcopy(q)
            q_new[i] -= delt
            cb = self.cost(q_new)

            g[0, i] = (cf - cb) / (2*delt)
        #g = g / np.linalg.norm(g)
        return g.T

    
    def cost_gradient(self, q):
        type(self).nGradCalls += 1
        time0 = time.perf_counter()
        
        q = self._process_q(q)

        x_g_ = q[self.Gx_IDX]#self.joint_to_grasp(q)
        tx_p_ = q[self.Px_IDX]#mat_to_vec(tTo_)
        
        q_grads = np.matrix(q*0.0)
        q_grads[0, self.Px_IDX] = self.corner_heuristic_grads(tx_p_) * self.Wt_P

        grasp_grads_ = self.GraspClient.query_grasp_grad(x_g_, \
                                                        q[self.PRESHAPE], \
                                                        self.object_sparse_grid, \
                                                        self.object_voxel_dims, \
                                                        self.object_size)
        grasp_grads_ = -(self.Wt_Post[0] * np.array(grasp_grads_[1]) + self.Wt_Post[1] * np.array(grasp_grads_[2]))
        q_grads[0, self.Gx_IDX] = grasp_grads_[:6] * self.Wt_G
        q_grads[0, self.PRESHAPE] = grasp_grads_[6] * self.Wt_G

        qp_reg, qg_reg = self.joints_reg_grads(q)

        q_grads[0, self.G_IDX] += qg_reg * self.Wt_R
        q_grads[0, self.P_IDX] += qp_reg * self.Wt_R

        GradCheck = False
        if GradCheck:
            fd_grads = self.cost_gradient_fd(q)
            np.set_printoptions(precision=2, floatmode='maxprec_equal')
            print('fd vs a_grads error: \n', np.hstack([fd_grads ,q_grads.T, (fd_grads - q_grads.T)]))
            vecsim = vector_similarity(fd_grads, q_grads.T)
            print('Similarity: ', np.arccos(vecsim)*180/np.pi, vecsim )
        type(self).tGradTime += time.perf_counter() - time0
        return q_grads.T
        #ik_grads_ = - 2 * self.cost_IK(q, grad=True)

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
        
        q_grads[0, self.G_IDX] += grasp_grads_qg * self.Wt_G
        q_grads[0, self.PRESHAPE] += grasp_grads_[6] * self.Wt_G
        
        #q_grads = q_grads / np.linalg.norm(q_grads)
        type(self).tGradTime += time.perf_counter() - time0
        return q_grads.T

    def o_jac_fd(self, q, delt = 1e-5):
        x_c = self.joint_to_grasp(q)
        x_c = np.array(x_c)
        j = []
        for i in range(self.dof):
            q_new = copy.deepcopy(q)
            q_new[i + self.dof] += delt
            x_f = self.joint_to_grasp(q_new)
            x_f = np.array(x_f)
            v = (x_f - x_c) / delt
            v = v.reshape(6,1)
            j.append(v)
        return np.hstack(j)

    def jactest(self):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        print(self.arm_palm_KDL.jacobian([0.0]*7))
        print(get_space_jacobian(self.obj_world_mat) @ self.arm_palm_KDL.jacobian([0.0]*7))

    def initialize_place(self, max_trials = 10):
        self.x0 = self.x0.reshape(-1)
        place_prior_ = [0.0] * 4 + [1.57] + [0.0]#mat_to_vec(self.obj_world_mat)#[0.0] * 4 + [1.57] + [0.0]
        place_prior_[0] = np.random.rand()*0
        place_prior_[1] = np.random.rand()*0
        place_prior_[2] = self.object_size[2]/2
        self.x0[self.Px_IDX] = place_prior_
        for iter_ in range(max_trials):
            place_prior_pose_ = convert_array_to_pose(place_prior_, "place_table_corner")
            world_place_pose_ = TransformPoseStamped(place_prior_pose_,"world", self.tf_b, self.tf_listener)
            p_trans_, p_quat_ = pose_to_array(world_place_pose_)
            qp_0_ = self.arm_palm_KDL.ik(p_trans_, p_quat_)
            if qp_0_ is None:
                place_prior_[0] = np.random.rand()
                place_prior_[1] = np.random.rand()
                self.x0[self.Px_IDX] = place_prior_
            else:
                qp_0_ = np.arctan2(np.sin(qp_0_), np.cos(qp_0_))
                self.x0[self.P_IDX] = qp_0_[:]
                self.x0 = self.x0.reshape(-1,1)
                return
        if qp_0_ is None:
            print("Unable to generate reachable grasp prior, proceeding with zero joint init")
        self.x0 = self.x0.reshape(-1,1)

    def initialize_grasp(self, max_trials = 10):
        self.x0 = self.x0.reshape(-1)
        for iter_ in range(max_trials):
            grasp_prior_ = self.GraspClient.query_grasp_prior(self.object_sparse_grid, \
                                                              self.object_voxel_dims, \
                                                              self.object_size)
            self.raw_prior = grasp_prior_[:6]
            prior_cost_ = -self.GraspClient.query_grasp_probabilities(grasp_prior_[:6], \
                                                        [grasp_prior_[6]], \
                                                        self.object_sparse_grid, \
                                                        self.object_voxel_dims, \
                                                        self.object_size)[0]
            self.x0[self.Gx_IDX] = grasp_prior_[:6]
            self.x0[self.PRESHAPE] = grasp_prior_[6]
            
            grasp_prior_pose_ = convert_array_to_pose(grasp_prior_[:6], "object_pose")
            world_grasp_pose_ = TransformPose(grasp_prior_pose_.pose, "world", "object_pose")
            g_trans_, g_quat_ = pose_to_array(world_grasp_pose_)
            self.xt = pose_to_6dof(world_grasp_pose_)
            qg_0_ = self.arm_palm_KDL.ik(g_trans_, g_quat_)
            if qg_0_ is None:
                print("\n\n\n\n\n\n\nUnreachable grasp, retrying")
            else:
                qg_0_f_ = np.arctan2(np.sin(qg_0_), np.cos(qg_0_))
                #assert np.all(np.sin(qg_0_)==np.sin(qg_0_f_)), "{}\n\n{}".format(qg_0_, qg_0_f_)
                #assert np.all(np.cos(qg_0_)==np.cos(qg_0_f_))
                #print("Found IK Solution: ", qg_0_)
                qg_0_ = qg_0_f_
                self.x0[self.G_IDX] = qg_0_[:]
                #self.x0[self.PRESHAPE] = grasp_prior_[6]
                xg_0_ = self.joint_to_grasp(self.x0)
                prior_cost_fk_ = -self.GraspClient.query_grasp_probabilities(xg_0_, \
                                                        [grasp_prior_[6]], \
                                                        self.object_sparse_grid, \
                                                        self.object_voxel_dims, \
                                                        self.object_size)[0]
                #print("Received Prior: {}\n After IK FK and Trans: {}".format(grasp_prior_[3:6], xg_0_[3:6]))
                #print("Equivalence: ", test_3d_rotation_equivalence(grasp_prior_[3:6], xg_0_[3:6]))
                #print("Cost at Prior: {}\n After IK: {}\n Error: {}".format(prior_cost_, prior_cost_fk_, prior_cost_ - prior_cost_fk_))
                self.x0 = self.x0.reshape(-1,1)
                return
        if qg_0_ is None:
            print("Unable to generate reachable grasp prior, proceeding with zero joint init")
        self.x0 = self.x0.reshape(-1,1)

    def gripper_position_grads(self, q, gripper_pts):
        '''
        Verified works
        '''
        q = self._process_q(q)

        tRw, tPw = decompose_transform(self.ptable_world_mat)

        wRo, wPo = decompose_transform(invert_trans(self.obj_world_mat))
        
        x_p = self.arm_palm_KDL.fk(q[self.P_IDX])
        wTg = vector2mat(x_p[:3], x_p[3:])
        wRg, wPg = decompose_transform(wTg)

        jac_p = self.place_kdl.jacobian(q[self.P_IDX])
        omega_p = jac_p[3:]
        linvel_p = jac_p[:3]

        wx_g = self.arm_palm_KDL.fk(q[self.G_IDX])
        gPw_inv = np.array(wx_g[:3]).reshape(3,1) 
        gx_g = vector_inverse(wx_g)
        gTw = vector2mat(gx_g[:3], gx_g[3:])
        gRw, gPw = decompose_transform(gTw)

        jac_g = self.grasp_kdl.jacobian(q[self.G_IDX])
        omega_g = jac_g[3:]
        linvel_g = jac_g[:3]
        
        tJRg_ = tRw @ batch_vector_skew_mat(omega_p) @ wRg
        q_grads_p = (tJRg_ @ gripper_pts.T).T + tRw @ linvel_p

        return q_grads_p

    def place_position_grads(self, q, obj_pts = None):
        '''
        Verified works
        '''
        q = self._process_q(q)

        tRw, tPw = decompose_transform(self.ptable_world_mat)

        wRo, wPo = decompose_transform(invert_trans(self.obj_world_mat))

        if obj_pts is not None:
            wPo = wRo @ obj_pts.T + wPo
        
        x_p = self.arm_palm_KDL.fk(q[self.P_IDX])
        wTg = vector2mat(x_p[:3], x_p[3:])
        wRg, wPg = decompose_transform(wTg)

        jac_p = self.place_kdl.jacobian(q[self.P_IDX])
        omega_p = jac_p[3:]
        linvel_p = jac_p[:3]

        wx_g = self.arm_palm_KDL.fk(q[self.G_IDX])
        gPw_inv = np.array(wx_g[:3]).reshape(3,1) 
        gx_g = vector_inverse(wx_g)
        gTw = vector2mat(gx_g[:3], gx_g[3:])
        gRw, gPw = decompose_transform(gTw)

        jac_g = self.grasp_kdl.jacobian(q[self.G_IDX])
        omega_g = jac_g[3:]
        linvel_g = jac_g[:3]
        
        tJRg_ = tRw @ batch_vector_skew_mat(omega_p) @ wRg
        q_grads_p = np.squeeze(tJRg_ @ (gRw @ wPo + gPw)).T + tRw @ linvel_p

        gJRw_ = -gRw @ batch_vector_skew_mat(omega_g)
        gJPw_ = -gRw @ linvel_g - np.squeeze(gJRw_ @ gPw_inv).T
        q_grads_g = tRw @ (np.squeeze(wRg @ gJRw_ @ wPo).T + wRg @ gJPw_)

        return q_grads_p, q_grads_g

    def place_orientation_grads(self, q):
        q = self._process_q(q)

        txo = self.object_pose_from_joints(q)

        tRw, tPw = decompose_transform(self.ptable_world_mat)

        wRo, wPo = decompose_transform(invert_trans(self.obj_world_mat))
        x_p = self.arm_palm_KDL.fk(q[self.P_IDX])
        wTg = vector2mat(x_p[:3], x_p[3:])
        wRg, wPg = decompose_transform(wTg)

        jac_p = self.place_kdl.jacobian(q[self.P_IDX])
        omega_p = jac_p[3:]
        linvel_p = jac_p[:3]

        wx_g = self.arm_palm_KDL.fk(q[self.G_IDX])
        gPw_inv = np.array(wx_g[:3]).reshape(3,1) 
        gx_g = vector_inverse(wx_g)
        gTw = vector2mat(gx_g[:3], gx_g[3:])
        gRw, gPw = decompose_transform(gTw)

        jac_g = self.grasp_kdl.jacobian(q[self.G_IDX])
        omega_g = jac_g[3:]
        linvel_g = jac_g[:3]

        gJRw = -gRw @ batch_vector_skew_mat(omega_g)
        tRg = tRw @ wRg
        tJRo_g = tRg @ gJRw @ wRo

        tJRo_p = tRw @ batch_vector_skew_mat(omega_p) @ wRg @ gRw @ wRo

        return tJRo_p[:,2,2].T, tJRo_g[:,2,2].T
        t_omega_g = tRg @ gJRw @ gRw.T @ tRg.T

        t_omega_p = tRw @ batch_vector_skew_mat(omega_p) @ tRw.T

        g_rpy_jac = []
        p_rpy_jac = []
        for i in range(t_omega_g.shape[0]):
            t_omega = t_omega_g[i]
            t_omega_v = np.array([t_omega[2][1], t_omega[0][2], t_omega[1][0]])
            g_rpy_dot = angular_velocity_to_rpy_dot(t_omega_v, txo[3:])
            g_rpy_jac.append(g_rpy_dot)

            t_omega = t_omega_p[i]
            t_omega_v = np.array([t_omega[2][1], t_omega[0][2], t_omega[1][0]])
            p_rpy_dot = angular_velocity_to_rpy_dot(t_omega_v, txo[3:])
            p_rpy_jac.append(p_rpy_dot)
        g_rpy_jac = np.vstack(g_rpy_jac).T
        p_rpy_jac = np.vstack(p_rpy_jac).T

        return p_rpy_jac, g_rpy_jac

    def object_pose_from_joints(self, q):
        q = self._process_q(q)

        x_g = q[self.Px_IDX]
        
        return x_g

    def add_collision_constraints(self):
        Collision_Constraint.static_init(self)
        Reflex_Collision.static_init(self)
        obj_pts = self.Obj_Grid.binned_pts#_raw_grid_idx_to_pts()
        gripper_pts = Reflex_Collision.get_gripper_points()
        num_pts = obj_pts.shape[0]
        print(f"Adding {num_pts} object collision checks per object")
        num_gpts = gripper_pts.shape[0]
        print(f"Adding {num_gpts} robot collision checks per object")
        for key in self.Env_List.keys():
            for i in range(num_pts):
                self.inequality_constraints.append(Collision_Constraint(key, i))
                break
            #for j in range(num_gpts):
               #self.inequality_constraints.append(Reflex_Collision(key, i))

    def evaluate_constraints(self, q):
        q = self._process_q(q)
        num_e = len(self.Env_List)
        obj_pts = self.Obj_Grid.binned_pts#_raw_grid_idx_to_pts()
        gripper_pts = Reflex_Collision.get_gripper_points()
        num_pts = obj_pts.shape[0]
        num_gpts = gripper_pts.shape[0]
        fitness = defaultdict(lambda: -np.inf)#np.zeros((8 + 2 * num_e, 1)) - np.inf
        for i,c in enumerate(self.inequality_constraints):
            fit = c.error(q)
            fitness[c._cname] = np.max([fit, fitness[c._cname]])
            continue
            if i < 8:
                fitness[c._cname] = np.max([fit, fitness[c._cname]])
            else:
                oid = (i - 8) // (num_pts + num_gpts)
                #print(oid)
                if ((i - 8) % (num_pts + num_gpts)) < num_pts:
                    fitness[2 * oid + 8] = np.max([fit, fitness[2 * oid + 8]])
                else:
                    fitness[2 * oid + 9] = np.max([fit, fitness[2 * oid + 9]])
                    #off_id = 8 + num_e * num_pts
                    #fitness[((i - off_id) // num_gpts) + off_id] += fit
        ret = []
        labels = self.get_constraint_labels()
        for l in labels:
            ret.append(fitness[l])
        return ret

    def get_constraint_labels(self):
        labels = []
        for c in self.inequality_constraints:
            labels.append(c._cname)
        return list(set(labels))

    def print_fitness(self, fitness):
        num_e = len(self.Env_List)
        print(f"""Constraint Errors: \n
        Grasp:             {fitness[0]}\n
        Place position:    {fitness[1:7]}\n
        Place Orientation: {fitness[7]}\n
        Object Collisions: {fitness[8:8+num_e]}\n
        Robot Collisions:  {fitness[8+num_e:8+2*num_e]}""")

    def publish_env_cloud(self):
        all_pts = np.array([]).reshape(0,3)
        colors = np.array([]).reshape(0,3)
        for key in self.Env_List.keys():
            VoxGrid = self.Env_List[key]
            pts = VoxGrid.obj_pts
            pts = transform_points(pts, VoxGrid.Trans)
            all_pts = np.vstack([all_pts, pts])
            colors = np.vstack([colors,VoxGrid.color])
        publish_as_point_cloud(all_pts, colors=colors, frame_id="place_table_corner")

    def get_env_poses(self):
        obj_poses = {}
        for key in self.Env_List.keys():
            obj_poses[key] = convert_array_to_pose(mat_to_vec(self.Env_List[key].Trans),\
                                                   "place_table_corner").pose
        return obj_poses

    def new_id(self):
        return str(len(self.Env_List))

    def obj_env_collision_test(self):
        tTo = self.Obj_Grid.Trans
        obj_pts = self.Obj_Grid.obj_pts
        query_pts = transform_points(obj_pts, tTo)
        for key in self.Env_List.keys():
            print(query_pts.max(0))
            img_grid = np.zeros((query_pts.max(0)*1000 + 10).astype(np.int)) + 255
            sdm, dsd = self.Env_List[key].query_points(query_pts)
            idx = (query_pts*1000).astype(np.int)
            for i in range(idx.shape[0]):
                print(idx[i] , sdm[i])
                if np.all(idx[i] > 0):
                    img_grid[idx[i][0], idx[i][1], idx[i][2]] = sdm[i] - 5
            return img_grid
       
class PnP_Problem(Problem):
    def __init__(self, obj_grid, place_heuristic = None, place_grads = None):
        '''
        Hyper parameters of the optimization problem goes here
        TODO: Might need to add arbitrary weights, and bounds here
        '''
        super().__init__()
        self.PLACE_SLICE = slice(0,3) # x,y,yaw
        self.GRASP_SLICE = slice(3,9) # x,y,z,roll,pitch,yaw
        self.PRESHAPE_SLICE = slice(9,10) # yaw
        self.G_IDX = self.GRASP_SLICE
        self.P_IDX = self.PLACE_SLICE
        self.PRESHAPE = self.PRESHAPE_SLICE
        self.GraspClient = GraspUtils()
        assert isinstance(obj_grid, ObjGrid), "Param obj_grid is not of ObjGrid type"
        obj_grid.test_cornered()
        self.object_sparse_grid = obj_grid.sparse_grid
        self.object_voxel_dims = obj_grid.grid_size
        self.object_size = obj_grid.true_size
        if place_heuristic is None:
            self.place_heuristic = lambda x: x[0] * x[1] #Place close origin
        if place_grads is None:
            self.place_grads = lambda x: np.array([1., 1., 0.])
        self.x0 = np.zeros((10,1))
        self.initialize_grasp()
        self.boundConstraints = { \
                                  0 : (0.0, 0.5), \
                                  1 : (0.0, 0.5), \
                                  2 : (0.0, 3.14), \
                                  3 : (-0.3, 0.3), \
                                  4 : (-0.3, 0.3), \
                                  5 : (-0.3, 0.3), \
                                  6 : (-3.14, 3.14), \
                                  7 : (-3.14, 3.14), \
                                  8 : (-3.14, 3.14), \
                                  9 : (0.0, 1.57) }

    def initialize_grasp(self):
        self.x0 = self.x0.reshape(-1)
        grasp_prior_ = self.GraspClient.query_grasp_prior(self.object_sparse_grid, \
                                                         self.object_voxel_dims, \
                                                         self.object_size)
        self.x0[self.GRASP_SLICE] = grasp_prior_[:6]
        self.x0[self.PRESHAPE_SLICE] = grasp_prior_[6]
        self.x0 = self.x0.reshape(-1,1)
        prior_cost_ = self.GraspClient.query_grasp_cost(grasp_prior_[:6], \
                                                        [grasp_prior_[6]], \
                                                        self.object_sparse_grid, \
                                                        self.object_voxel_dims, \
                                                        self.object_size)
        print("Cost at Prior({}): {}".format(grasp_prior_, prior_cost_))
        

    def cost(self, x):
        '''
        Overriding base class function.
        Cost = H(FK(q_p):Zo,Ze) -log(p(FK(q_g):Zo))
        param:
        x - Decision variable (6(place)+6(grasp)+1(preshape) x 1 float)
        '''
        if isinstance(x, np.matrix):
            x = np.squeeze(np.asarray(x))
        x = x.reshape(-1) # Change input to row vector
        place_cost_ = self.place_heuristic(x[self.PLACE_SLICE])
        grasp_cost_ = self.GraspClient.query_grasp_cost(x[self.GRASP_SLICE], \
                                                        x[self.PRESHAPE_SLICE], \
                                                        self.object_sparse_grid, \
                                                        self.object_voxel_dims, \
                                                        self.object_size)
        return place_cost_ + grasp_cost_ #Grasp cost is already -log

    def cost_gradient(self, x):
        if isinstance(x, np.matrix):
            x = np.squeeze(np.asarray(x))
        x = x.reshape(-1) # Change input to row vector
        x_grads = np.matrix(x*0.0)
        x_grads[0, self.PLACE_SLICE] = self.place_grads(x[self.PLACE_SLICE])
        grasp_grads = self.GraspClient.query_grasp_grad(x[self.GRASP_SLICE], \
                                                        x[self.PRESHAPE_SLICE], \
                                                        self.object_sparse_grid, \
                                                        self.object_voxel_dims, \
                                                        self.object_size)
        print("*********grasp _grads *************")
        print(grasp_grads)
        print("************************************")
        x_grads[0, self.GRASP_SLICE] = grasp_grads[0:6]
        x_grads[0, self.PRESHAPE_SLICE] = grasp_grads[6]
        return x_grads.T

    def cost_gradient_fd(self, q, delt=1e-5):
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        q = q.reshape(-1)
        g = np.matrix(q * 0.0)
        for i in range(q.shape[0]):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            cf = self.cost(q_new)

            q_new = copy.deepcopy(q)
            q_new[i] -= delt
            cb = self.cost(q_new)

            g[0, i] = (cf - cb) / (2*delt)
        return g.T
        
