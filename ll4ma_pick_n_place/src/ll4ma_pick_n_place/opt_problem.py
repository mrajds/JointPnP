#!/usr/bin/env python3
"""
Pick N Place optimization problem definition for ll4ma_opt suite of solvers.
Extends ll4ma_opt/problem class.
"""

from ll4ma_pick_n_place.grasp_utils_client import GraspUtils
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
from ll4ma_pick_n_place.kdl_utils import ManipulatorKDL, pose_difference
from ll4ma_pick_n_place.planning_client import get_IK
from ll4ma_pick_n_place.visualizations import map_rgb
from collections import defaultdict
import ll4ma_pick_n_place.placement_heuristics as heurs
import numpy as np
import tf
import tf2_ros
import tf2_geometry_msgs
import copy
import rospy
import time
from matplotlib import pyplot, cm
import inspect
from sklearn.preprocessing import normalize

ROBOT_DESCRIPTION = 'robot_description'

G_GRAD_LOG_SUFFIX = 'qg_grad'
P_GRAD_LOG_SUFFIX = 'qp_grad'

NP_DTYPE_ = np.float32

def gen_query_grid_pts(l=0.5, w=0.5, res=100):
    x = np.arange(.0, l, 1/res)
    x = x[x<l]
    y = np.arange(.0, w, 1/res)
    y = y[y<w]
    G = np.array(np.meshgrid(x,y)).T.reshape(-1,2)
    G = np.hstack([G, np.zeros((G.shape[0],1))])
    return G

def gen_3d_grid_pts(l=0.5, w=0.5, min_z=0.0, max_z=0.5, res=1000):
    x = np.arange(.0, l, 10/res)
    y = np.arange(.0, w, 10/res)
    z = np.arange(min_z, max_z, 10/res)
    G = np.array(np.meshgrid(x,y,z)).T.reshape(-1,3)
    #G = np.hstack([G, np.zeros((G.shape[0],1))])
    return G

def _process_q(q):
    if isinstance(q, np.matrix):
        q = np.squeeze(np.asarray(q))
    return q.reshape(-1) # Change input to row vector

def compare_grads(a, b):
    ac = a.reshape(-1,1)
    bc = b.reshape(-1,1)
    #np.set_printoptions(precision=2, floatmode='maxprec_equal')
    print('fd vs a_grads error: \n', np.hstack([ac, bc, ac - bc]))
    vecsim = vector_similarity(ac, bc)
    print('Similarity: ', np.arccos(vecsim)*180/np.pi, vecsim )
    print('norms: ', [np.linalg.norm(ac), np.linalg.norm(bc), np.linalg.norm(ac) - np.linalg.norm(bc)])

def clip_grad(vec, tresh=0.1):
    return vec
    high = np.max(np.abs(vec))
    alpha = min(high, tresh)
    scale = alpha/high
    return scale * vec

def chomp_smooth(d, eps):
    try:
        ret = np.full_like(d, np.nan)
        #ret = np.zeros(d.shape)
        ret[d<0] = -d[d<0] + 0.5 * eps
        ret[np.logical_and(d>=0, d<=eps)] = (1/(2*eps)) \
                        * (d[np.logical_and(d>=0, d<=eps)] - eps)**2
        ret[d>eps] = 0.0
    except ValueError:
        print('ValueError in chomp_smooth: ')
        print(d.shape, ret[d<0].shape, d[d<0].shape)
        print(d.shape, ret[d>=0 and d<=eps].shape, d[d>=0 and d<=eps].shape)
    return ret

def chomp_smooth_grads(d, eps):
    ret = np.full_like(d, np.nan)
    ret[d<0] = -1
    ret[np.logical_and(d>=0, d<=eps)] = (1/eps) \
                        * (d[np.logical_and(d>=0, d<=eps)] - eps)
    ret[d>eps] = 0.0
    return ret

class ExecLogger():
    LogDict = {}
    def __init__(self, keys):
        for k in keys:
            if k not in type(self).LogDict.keys():
                type(self).LogDict[k] = defaultdict(lambda: np.nan)

    def copy_log(self, inDict):
        type(self).LogDict = inDict

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

    def get_pickle(self):
        ret = {}
        for k in type(self).LogDict.keys():
            ret[k] = {}
            for i in type(self).LogDict[k].keys():
                ret[k][i] = type(self).LogDict[k][i]
        return ret
        

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

    def static_init(pblm, epsilon = 0.015):
        Collision_Constraint.pblm = pblm
        Collision_Constraint.epsilon = epsilon
        for key in pblm.Env_List.keys():
            Collision_Constraint.Q_CA[key] = (np.array([]), np.array([]), np.array([]))
    
    def __init__(self, EKey, idx):
        self.EKey = EKey
        self.idx = idx
        self._cname = "Obj_Coll_" + str(EKey)
        ExecLogger.__init__(self, [self._cname, self._cname+G_GRAD_LOG_SUFFIX, self._cname+P_GRAD_LOG_SUFFIX, \
                                   self._cname + 'sdf_grad'])
        self.Wt_G = int(self.pblm.Wt_G > 0)
        self.Wt_P = int(self.pblm.Wt_P > 0)
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
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        tTg = self.pblm.joint_to_place(q, asmat=True)
        gTo = invert_trans(self.pblm.joint_to_grasp(q, asmat=True))
        tTo = tTg @ gTo

        obj_pts = self.pblm.Obj_Grid.binned_pts

        query_pts = transform_points(obj_pts, tTo)

        time1 = time.perf_counter()
        
        sd, dsd = self.pblm.Env_List[self.EKey].query_points(query_pts)

        time2 = time.perf_counter()

        dl = self.pblm.Env_List[self.EKey].max_tsdf() - self.sbuff
        
        csd = -chomp_smooth(sd-dl, self.sbuff) + dl
        csd_g = -chomp_smooth_grads(sd-dl, self.sbuff)

        time3 = time.perf_counter()

        #delt = [0, 0, 0]
        #diff = 1e-10
        #dsd_fd = np.zeros((query_pts.shape[0],3))
        #for i in range(3):
        #    delt = [0, 0, 0]
        #    delt[i] = diff
        #    query_ptsf = query_pts + delt
        #    query_ptsb = query_pts - delt
        #    sd_f = self.pblm.Env_List[self.EKey].query_points(query_ptsf)[0]
        #    csd_f = -Collision_Constraint.chomp_smooth(sd_f-dl, self.sbuff) + dl
        #    sd_b = self.pblm.Env_List[self.EKey].query_points(query_ptsb)[0]
        #    csd_b = -Collision_Constraint.chomp_smooth(sd_b-dl, self.sbuff) + dl
        #    dsd_fd[:, i] = (sd_f - sd_b) / (2*diff)

        #cidx = np.random.randint(query_pts.shape[0])
        #compare_grads(dsd[cidx], dsd_fd[cidx])
        #print(query_pts[437])
        #sd -= 5

        #print(dsd, dsd_fd.T, dsd - dsd_fd.T)
        #vecsim = vector_similarity(dsd[self.idx], dsd_fd.T[self.idx])
        #print('DSD Similarity: ', np.arccos(vecsim)*180/np.pi, vecsim )

        q_grads = np.concatenate(self.pblm.place_position_grads(q, obj_pts), axis=2)

        time4 = time.perf_counter()
        
        #print(sd.shape, dsd.shape, q_grads.shape)
        cdsd = np.einsum('i,ij->ij', csd_g, dsd)
        grads = np.einsum('ij,ijk->ik',cdsd, q_grads)
        #grads[:,:7] = normalize(grads[:,:7], axis=1)
        #grads[:,7:] = normalize(grads[:,7:], axis=1)

        time5 = time.perf_counter()

        #print('Collision Profile:')
        #print(f'PtCmp: {time1-time0}\nQuery: {time2-time1}\nSmoth: {time3-time2}\nGrads: {time4-time3}\nPoGra: {time5-time4}')
        #print(f'Total: {time5-time0}')
        
        type(self).Q_CA[self.EKey] = (q, csd*1e-3, -grads, dsd)
        #type(self).Q_CA[self.EKey][1].SD = sd
        #type(self).Q_CA[self.EKey][2].Grad = grads

        type(self).tCache_Time += time.perf_counter() - time0

    def error(self, q):
        type(self).nErrorCalls += 1
        time0 = time.perf_counter()
        res = self.error_core(q)
        #print(self._cname, res)
        self.log_val(self._cname, q.tobytes(), res)
        type(self).tErrorTime += time.perf_counter() - time0
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
        time0 = time.perf_counter()
        grads = self.error_gradient_core(q)
        self.log_val(self._cname+G_GRAD_LOG_SUFFIX, q.tobytes(), np.linalg.norm(grads[0, self.pblm.G_IDX]))
        self.log_val(self._cname+P_GRAD_LOG_SUFFIX, q.tobytes(), np.linalg.norm(grads[0, self.pblm.P_IDX]))
        idx = np.argmin(self.Q_CA[self.EKey][1])
        cart_grad = self.Q_CA[self.EKey][3][idx]
        self.log_val(self._cname+'sdf_grad', q.tobytes(), np.linalg.norm(cart_grad))
        type(self).tGradTime += time.perf_counter() - time0
        return grads

    def error_gradient_core(self, q):
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)
        if not np.array_equal(self.Q_CA[self.EKey][0], q):
            self.cache_(q)
        idx = np.argmin(self.Q_CA[self.EKey][1])
        #idx = self.idx
        q_grads[0, self.pblm.P_IDX] = clip_grad(self.Q_CA[self.EKey][2][idx][:7] * self.Wt_P)
        q_grads[0, self.pblm.G_IDX] = clip_grad(self.Q_CA[self.EKey][2][idx][7:] * self.Wt_G)

        GradCheck = not True
        if GradCheck:
            fd_grads = self.error_gradient_fd(q)
            fd_grads[0, self.pblm.G_IDX] *= self.Wt_G
            fd_grads[0, self.pblm.P_IDX] *= self.Wt_P
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
        for i in range(self.pblm.dof*2 + 1):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            cf = self.error_core(q_new)
            
            q_new = copy.deepcopy(q)
            q_new[i] -= delt
            cb = self.error_core(q_new)

            g[0, i] = (cf - cb) / (2*delt)
        #g = g / np.linalg.norm(g)
        return g


class Table_Constraint(Constraint):
    origin = [0.0, 0.0, 0.0]
    flange_pt = [-0.125, 0.0, 0.0]
    wrist_pt = [-0.245, 0.0, 0.0]
    flange_radii = 0.06 + 0.02
    wrist_radii = 0.10 + 0.02

    def __init__(self, kdl, idx, height, cname='table'):
        self.idx = idx
        self.ck_pts = np.array([self.origin, self.flange_pt, self.wrist_pt])
        self.radii = np.array([self.flange_radii, self.flange_radii, self.wrist_radii])
        self.height = height
        self.kdl = kdl
        self._cname = cname

    def error(self, q):
        q = _process_q(q)
        x = self.kdl.fk(q[self.idx])
        T = vector2mat(x[:3], x[3:])
        query_pts = transform_points(self.ck_pts, T)
        dist = query_pts[:,2] - self.radii
        return self.height - min(dist)

    def error_gradient(self, q):
        q = _process_q(q)
        q_grads = np.matrix(q*0.0)
        x = self.kdl.fk(q[self.idx])
        T = vector2mat(x[:3], x[3:])
        query_pts = transform_points(self.ck_pts, T)
        dist = query_pts[:,2] - self.radii
        min_idx = np.argmin(dist)
        jac = self.kdl.jacobian(q[self.idx])
        linvel = jac[:3]
        omega = jac[3:]
        R_dot = batch_vector_skew_mat(omega) @ T[:3, :3]
        grads = (R_dot @ self.ck_pts[min_idx].T).T + linvel
        q_grads[0, self.idx] = -grads[2]
        grad_check=not True
        if grad_check:
            fd_grads = self.error_gradient_fd(q)
            compare_grads(fd_grads, q_grads)
        return q_grads.T

    def error_gradient_fd(self, q, delt=1e-3):
        #delt = 1e-5
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        q = q.reshape(-1) # Change input to row vector
        g = np.matrix(q * 0.0)
        for i in range(15):
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


class Grasp_Collision(Reflex_Collision):
    CACHE_ = {}

    def static_init(pblm, epsilon = 0.02):
        #super().static_init(pblm, epsilon)
        Grasp_Collision.epsilon = epsilon
        for key in pblm.GEnv_List.keys():
            Grasp_Collision.CACHE_[key] = (np.array([]), np.array([]), np.array([]))
        Grasp_Collision.CACHE_['grasp_obj'] = (np.array([]), np.array([]), np.array([]))

    def __init__(self, idx, gkey):
        self.idx = idx
        self._cname = "Grasp_Coll"
        ExecLogger.__init__(self, [self._cname])
        self.Wt_G = int(self.pblm.Wt_G > 0)
        self.Wt_P = int(self.pblm.Wt_P > 0)
        self.sbuff = 5
        self.GKey=gkey
        print(f"Setting grasp collision buffer: {self.epsilon}")
        if idx == 0:
            self.always_active=True

    def cache_(self, q):
        type(self).nCache_Calls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        
        #x_p = self.pblm.joint_to_place(q)
        #tTg = vector2mat(x_p[:3], x_p[3:])
        x_g = self.pblm.joint_to_grasp(q)
        oTg = vector2mat(x_g[:3], x_g[3:])

        gripper_pts = type(self).get_gripper_points(preshape = q[self.pblm.PRESHAPE], ignore_fingers=self.GKey=='grasp_obj')
        query_pts = transform_points(gripper_pts, oTg)
        if self.GKey=='grasp_obj':
            Obj_Grid = self.pblm.Obj_Grid
        else:
            Obj_Grid = self.pblm.GEnv_List[self.GKey]
            
        sd, dsd = Obj_Grid.query_points(query_pts)
    
        dl = Obj_Grid.max_tsdf() - self.sbuff
        
        csd = -chomp_smooth(sd-dl, self.sbuff) + dl
        csd_g = -chomp_smooth_grads(sd-dl, self.sbuff)
        
        #sd -=5
        #sd *= 1e-3
        q_grads_g = self.pblm.gripper_position_grads(q, gripper_pts)[1]
        q_grads_pre = self.gripper.Get_Derivate_Preshape([0.0, 0.0, 0.0], q[self.pblm.PRESHAPE], ignore_fingers=self.GKey=='grasp_obj')
        #print(q_grads_p.shape, q_grads_pre.shape)
        q_grads = np.concatenate((q_grads_g, np.expand_dims(q_grads_pre,2)), axis=2)
        #print(q_grads.shape)

        cdsd = np.einsum('i,ij->ij', csd_g, dsd)
        grads = np.einsum('ij,ijk->ik',cdsd, q_grads)
        #grads = np.einsum('ij,ijk->ik',dsd, q_grads)
        #print(cdsd.shape)

        type(self).CACHE_[self.GKey] = (q, csd*1e-3, -grads, dsd)
        #type(self).CACHE_ = (q, sd, grads)
        #type(self).CACHE_[1].SD = sd
        #type(self).CACHE_[2].Grad = grads
        GradCheck = False
        if GradCheck:
            idx = np.random.randint(0, query_pts.shape[0])
            check_pt = query_pts[idx]
            check_pt = check_pt.reshape(1,3)
            delt = 1e-3
            for i in range(3):
                new_pt = copy.deepcopy(check_pt[0])
                new_pt[i] += delt
                new_pt = new_pt.reshape(1,3)
                check_pt = np.vstack([check_pt, new_pt])
            #print(check_pt)
            #check_q = transform_points(check_pt, oTg)
            check_q = check_pt
            #print(check_q)
            ch_sd, ch_dsd = Obj_Grid.query_points(check_q)
            ch_csd = -chomp_smooth(ch_sd-dl, self.sbuff) + dl
            #print(ch_sd)
            #print(self.pblm.Obj_Grid.max_tsdf())
            ch_sd_fd = np.zeros((1,3))
            ch_csd_fd = np.zeros((1,3))
            for i in range(3):
                ch_sd_fd[0][i] = (ch_sd[i+1] - ch_sd[0]) / delt
                ch_csd_fd[0][i] = (ch_csd[i+1] - ch_csd[0])/delt
            #compare_grads(ch_sd_fd, ch_dsd[0])
            compare_grads(ch_csd_fd, cdsd[idx])

            #Checking the 2nd part of the gradient
            idx = np.random.randint(0, query_pts.shape[0])
            check_pt = gripper_pts[idx]
            check_pt = check_pt.reshape(1,3)
            delt = 1e-7
            q_grad_g_fd = np.zeros((3,7))
            for i in range(7):
                q_new = copy.deepcopy(q)
                q_new[7+i] += delt
                cx_g = self.pblm.joint_to_grasp(q_new)
                coTg = vector2mat(cx_g[:3], cx_g[3:])
                check_q = transform_points(check_pt, coTg)
                q_grad_g_fd[:,i] = (check_q - query_pts[idx])/delt
            #for k in range(3):
                #compare_grads(q_grads_g[idx][k], q_grad_g_fd[k])

            #checking the merged gradient
            print("Merged fd grads:")
            grad_fd = ch_csd_fd @ q_grad_g_fd
            compare_grads(grads[idx][:7], grad_fd)

        type(self).tCache_Time += time.perf_counter() - time0

    def error_core(self, q):
        q = self.pblm._process_q(q)
        if not np.array_equal(self.CACHE_[self.GKey][0], q):
            self.cache_(q)
        idx = np.argmin(self.CACHE_[self.GKey][1])
        if self.GKey=='grasp_obj':
            0.005 - self.CACHE_[self.GKey][1][idx]
        return self.epsilon - self.CACHE_[self.GKey][1][idx]

    def error_gradient(self, q):
        type(self).nGradCalls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)
        if not np.array_equal(self.CACHE_[self.GKey][0], q):
            self.cache_(q)
        idx = np.argmin(self.CACHE_[self.GKey][1])
        q_grads[0, self.pblm.G_IDX] = self.CACHE_[self.GKey][2][idx][:7] * self.Wt_G
        q_grads[0, self.pblm.PRESHAPE] = self.CACHE_[self.GKey][2][idx][7] * self.Wt_G
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
    
    
class Grasp_Constraint(Constraint, ExecLogger):

    nErrorCalls = 0
    tErrorTime = 0

    nGradCalls = 0
    tGradTime = 0
    
    def __init__(self, pblm, grasp_bound=0.5, idx=0):
        self.pblm = pblm
        self.grasp_bound = grasp_bound
        self.idx = idx
        self._cname = "Grasp_Bound_"
        ExecLogger.__init__(self, [self._cname])
        self.Wt_G = int(self.pblm.Wt_G > 0)
        self.Wt_P = int(self.pblm.Wt_P > 0)
        self.always_active=True

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
        #x_g = np.array(self.pblm.joint_to_grasp(q)[:3])
        #return x_g @ x_g.T - (0.05)**2
        grasp_cost = self.pblm.grasp_probabilities(q)[self.idx]
        return grasp_cost - self.grasp_bound

    def error_gradient(self, q):
        type(self).nGradCalls += 1
        time0 = time.perf_counter()
        
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)

        #x_g = np.array(self.pblm.joint_to_grasp(q)[:3])

        #oRw, oPw = decompose_transform(self.pblm.obj_world_mat)
        
        #jac_g = self.pblm.grasp_kdl.jacobian(q[self.pblm.G_IDX])
        #omega_g = jac_g[3:]
        #linvel_g = jac_g[:3]
        
        qg_grads, qpre_grads = self.pblm.grasp_grads(q, idx=self.idx)
        q_grads[0, self.pblm.G_IDX] = qg_grads * self.Wt_G #2*x_g @ oRw @ linvel_g #qg_grads
        q_grads[0, self.pblm.PRESHAPE] = qpre_grads * self.Wt_G
        
        type(self).tGradTime += time.perf_counter() - time0

        GradCheck= not True
        if GradCheck:
            fd_grads = self.error_gradient_fd(q)
            compare_grads(fd_grads, q_grads)

        #np.set_printoptions(precision=2, floatmode='maxprec_equal')
        #print('fd vs a_grads error: \n', fd_grads, q_grads, fd_grads - q_grads)
        #vecsim = vector_similarity(fd_grads, q_grads.T)
        #print('Similarity: ', np.arccos(vecsim)*180/np.pi, vecsim )
        
        return q_grads.T

    def error_gradient_fd(self, q, delt=1e-3):
        print("FD grads called")
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
        self.Wt = 1
        self.pblm = pblm
        self.Wt_G = int(self.pblm.Wt_G > 0)
        self.Wt_P = int(self.pblm.Wt_P > 0)
        self.upper_bound = upper_bound
        self.always_active=True
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
        ExecLogger.__init__(self, [self._cname, self._cname+G_GRAD_LOG_SUFFIX, self._cname+P_GRAD_LOG_SUFFIX])
        self.wTo = invert_trans(pblm.obj_world_mat)
        self.upright = np.arccos(max(self.wTo[2, 2], 1.0)) #euler_from_mat(pblm.obj_world_mat)
        print('constraint: ', self.upright)
        self.x_p_grad = np.zeros((1,6))
        self.x_p_grad[0, self.idx] = 1.0
        pad = max(self.pblm.object_size[:2])/2
        self.POS_CONST_L = [0.0 + pad , 0.0 + pad, self.pblm.object_size[2]/2] #-self.pblm.obj_world_mat[3,2] - 0.59]
        self.POS_CONST_H = [self.pblm.table_size[0] - pad, self.pblm.table_size[1] - pad, self.pblm.object_size[2]/2 + 0.02]

        if hasattr(self.pblm.place_heuristic, "place_bounds"):
            for i in range(len(self.pblm.place_heuristic.place_bounds[0])):
                self.POS_CONST_L[i] = self.pblm.place_heuristic.place_bounds[0][i]
                #self.POS_CONST_L[1] = self.pblm.place_heuristic.place_bounds[0][1]
                #self.POS_CONST_L[2] = self.pblm.place_heuristic.place_bounds[0][2]
                self.POS_CONST_H[i] = self.pblm.place_heuristic.place_bounds[1][i]
                #self.POS_CONST_H[1] = self.pblm.place_heuristic.place_bounds[1][1]
                #self.POS_CONST_H[2] = self.pblm.place_heuristic.place_bounds[1][2]
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
        return res * self.Wt
    
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
            tx_p_ = self.pblm.object_pose_from_joints(q)#mat_to_vec(tTo_)
            tTo_ = vector2mat(tx_p_[:3], tx_p_[3:])
            self.cache = (q, tx_p_, tTo_)

        if self.idx < 3:
            if self.upper_bound:
                return tx_p_[self.idx] - self.POS_CONST_H[self.idx] #x_p[id]<0
            else:
                return self.POS_CONST_L[self.idx] - tx_p_[self.idx] #x_p[id]<0
        else:
            #print(np.arccos(max(tTo_[2, 2], 1.0)), (np.arccos(max(tTo_[2, 2], 1.0)) - self.upright) ** 2)
            return (np.arccos(min(tTo_[2, 2], 1.0)) - self.upright) ** 2 - 3e-5

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
        grads = self.error_gradient_core(q)
        #print(grads.shape)
        #print(grads[0, self.pblm.G_IDX], grads[0, self.pblm.G_IDX])
        self.log_val(self._cname+G_GRAD_LOG_SUFFIX, q.tobytes(), np.linalg.norm(grads[self.pblm.G_IDX]))
        self.log_val(self._cname+P_GRAD_LOG_SUFFIX, q.tobytes(), np.linalg.norm(grads[self.pblm.P_IDX]))
        type(self).tGradTime += time.perf_counter() - time0
        return grads * self.Wt
        
    def error_gradient_core(self, q):
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)

        grad_check = False

        if self.idx < 3:
            q_grads_p, q_grads_g = self.pblm.place_position_grads(q)
            q_grads[0, self.pblm.P_IDX] = q_grads_p[self.idx] * self.Wt_P
            q_grads[0, self.pblm.G_IDX] = q_grads_g[self.idx] * self.Wt_G

            #type(self).tGradTime += time.perf_counter() - time0
            if self.upper_bound:
                return q_grads.T
            else:
                return -q_grads.T
        else:
            tx_p_ = self.pblm.object_pose_from_joints(q)
            tTo = vector2mat(tx_p_[:3], tx_p_[3:])
            cart_grad = 2 * (np.arccos(min(tTo[2, 2], 1.0)) - self.upright) \
                        * (-1/(1-tTo[2, 2]**2)**0.5)
            if cart_grad == np.nan:
                cart_grad = 0.0
            a_cart_grad = cart_grad * (-np.sin(tx_p_[3]) * np.cos(tx_p_[4]))
            b_cart_grad = cart_grad * (-np.sin(tx_p_[4]) * np.cos(tx_p_[3]))
            #2 * (tTo[2, 2] - self.wTo[2, 2]) #(tx_p_[self.idx] - self.upright[self.idx-3])
            q_grads_p, q_grads_g = self.pblm.place_orientation_grads(q)
            #print(q_grads_p)
            q_grads[0, self.pblm.P_IDX] = cart_grad * q_grads_p * self.Wt_P#[0] + b_cart_grad * q_grads_p[1]
            q_grads[0, self.pblm.G_IDX] = cart_grad * q_grads_g * self.Wt_G#[0] + b_cart_grad * q_grads_g[1]
            #print(np.linalg.norm(q_grads[0, self.pblm.G_IDX]),np.linalg.norm(q_grads[0, self.pblm.P_IDX]))
            #fd_grads = self.error_gradient_fd(q)
            #grad_check = not False
        if grad_check:
            fd_grads = self.error_gradient_fd(q)
            compare_grads(fd_grads, q_grads)
        return q_grads.T

    def error_gradient_fd(self, q, delt=1e-5):
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
        err = self.bound * .90 - q[self.idx]
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

class Place_Error_Bound(Constraint, ExecLogger):
    def __init__(self, bound, pblm):
        self.bound = bound
        self.pblm = pblm
        self._cname = 'place_error_bound'

    def error(self, q):
        q = self.pblm._process_q(q)
        gTo_ = invert_trans(self.pblm.joint_to_grasp(q, asmat=True))

        tTg_ = self.pblm.joint_to_place(q, asmat=True)

        tTo_ = tTg_ @ gTo_
        tx_p_ = mat_to_vec(tTo_)
        place_cost_ = self.pblm.place_cost(tx_p_)

        return place_cost_ - self.bound

    def error_gradient(self, q):
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)
        
        gTo_ = invert_trans(self.pblm.joint_to_grasp(q, asmat=True))

        tTg_ = self.pblm.joint_to_place(q, asmat=True)

        tTo_ = tTg_ @ gTo_
        tx_p_ = mat_to_vec(tTo_)

        q_grads[0, self.pblm.P_IDX], q_grads[0, self.pblm.G_IDX] = self.pblm.place_grads(tx_p_, q)

        return q_grads.T
        

class Homotopy(Constraint):
    def __init__(self, idx, pblm):
        self.pblm = pblm
        self.p_idx = idx
        self.g_idx = 7 + idx

    def error(self, q):
        q = self.pblm._process_q(q)
        return (q[self.g_idx] - q[self.p_idx])**2 - (np.pi/2) ** 2

    def error_gradient(self, q):
        q = self.pblm._process_q(q)
        q_grads = np.matrix(q*0.0)
        q_grads[0, self.p_idx] = -2 * (q[self.g_idx] - q[self.p_idx])
        q_grads[0, self.g_idx] = 2 * (q[self.g_idx] - q[self.p_idx])
        return q_grads.T
    
class PnP_Problem_CS(Problem, ExecLogger):

    nCostCalls = 0
    tCostTime = 0

    nGradCalls = 0
    tGradTime = 0
    
    def __init__(self, obj_grid, env_list, genv_list={}, G_Wt=1.0, P_Wt=15.0, place_heuristic=None, grasp_mode=None):
        '''
        PnP_Problem in joint configuration space.
        '''
        super().__init__()
        self.COST_TERMS = ["post", "prior", "lkh", "place", "reg_p", "reg_g"]
        ExecLogger.__init__(self, self.COST_TERMS)
        if place_heuristic is None:
            place_heuristic = heurs.corner_heuristic()
        self.place_heuristic = place_heuristic
        self.place_frame = 'place_table_corner'
        if hasattr(self.place_heuristic, 'frame_id'):
            self.place_frame = self.place_heuristic.frame_id
        self.Env_List = env_list
        self.GEnv_List = genv_list
        self.place_kdl = ManipulatorKDL(robot_description=ROBOT_DESCRIPTION)
        self.grasp_kdl = ManipulatorKDL(robot_description=ROBOT_DESCRIPTION)
        self.arm_palm_KDL = ManipulatorKDL(robot_description=ROBOT_DESCRIPTION)
        self.dof = self.arm_palm_KDL.get_dof()

        self.P_IDX = slice(0, self.dof) # 7 - dof for arm
        self.G_IDX = slice(self.dof, self.dof * 2) # 7 - dof for arm
        self.PRESHAPE = slice(self.dof * 2, self.dof*2 + 1) # 1 - dof for preshape

        self.Wt_P = P_Wt
        self.Wt_G = G_Wt
        print("grasp_weighting: ", self.Wt_G)
        self.Wt_R = 0.0

        self.Wt_Post = [1., 0.5]

        self.table_corner = [0.07442, 0.186142]
        self.table_size = [0.5, 0.6] # Corner
        self.table_size = [0.75, 0.92] # Packing
        self.table_size = [0.30, 0.86] # Shelfs
        self.upright = [0.0, 0.0]

        joint_lows, joint_highs = self.arm_palm_KDL.get_joint_limits()
        #self.boundConstraints = {}
        for i_ in range(self.dof * 2):
            self.min_bounds[i_] = joint_lows[i_ % self.dof]
            self.max_bounds[i_] = joint_highs[i_ % self.dof]
            #self.boundConstraints[i_] = (joint_lows[i_%self.dof], joint_highs[i_%self.dof])
        self.min_bounds[self.dof * 2] = 0.0
        self.max_bounds[self.dof * 2] = 1.57
        #self.boundConstraints[self.dof * 2] = (0.0, 1.57)
        self.GraspClient = GraspUtils()
        assert isinstance(obj_grid, ObjGrid), "Param obj_grid is not of ObjGrid type"
        obj_grid.test_cornered()
        print("Unpacking obj_grid")
        self.Obj_Grid_raw = copy.deepcopy(obj_grid)
        self.Obj_Grid = obj_grid
        self.object_sparse_grid = obj_grid.sparse_grid
        self.object_voxel_dims = obj_grid.grid_size
        self.object_size = obj_grid.true_size
        self.x0 = np.zeros((self.dof*2 + 1, 1))
        self.initial_solution = self.x0
        print("Waiting for tf states")
        self.tf_b = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_b)
        try:
            self.obj_world_mat = get_tf_mat("object_pose", "world")
            self.ptable_world_mat = get_tf_mat(self.place_frame, "world")
        except :
            print("Warning states not found")
        print("Setting initialization")
        self.GraspClient.init_grasp_nets(self.object_sparse_grid, \
                                         self.object_voxel_dims, \
                                         self.object_size)
        self.initialize_grasp(idx=grasp_mode)
        print("Acquired grasp initialization")
        self.initialize_place()
        #self.x0[self.P_IDX] = self.x0[self.G_IDX]
        print("Values initialized")
        xt = self.arm_palm_KDL.fk(self.x0[self.G_IDX])
        xt = convert_array_to_pose(xt, "world")
        xt = TransformPoseStamped(xt, "object_pose", self.tf_b, self.tf_listener)
        self.xt = pose_to_6dof(xt)
        self.add_constraints()
    def add_constraints(self):
        self.add_bounds_as_ieqs()
        #self.inequality_constraints.append(Grasp_Constraint(self,0,2))
        #self.inequality_constraints.append(Grasp_Constraint(self,0.7,1))
        #self.inequality_constraints.append(Homotopy(4, self))
        if self.Wt_G:
            self.inequality_constraints.append(Table_Constraint(self.grasp_kdl, self.G_IDX, 0.735))
        if self.Wt_P:
            self.inequality_constraints.append(Placement_Constraint(0, self))
            self.inequality_constraints.append(Placement_Constraint(0, self, True))
            self.inequality_constraints.append(Placement_Constraint(1, self))
            self.inequality_constraints.append(Placement_Constraint(1, self, True))
            self.inequality_constraints.append(Placement_Constraint(2, self))
            self.inequality_constraints.append(Placement_Constraint(2, self, True))
            self.inequality_constraints.append(Placement_Constraint(3, self))
            self.inequality_constraints.append(Table_Constraint(self.place_kdl, self.P_IDX, 0.735))
        self.reset_performance_stats()
        #self.inequality_constraints.append(Placement_Constraint(4, self))
    def cull(self):
        self.inequality_constraints = []
    def size(self):
        return 15

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

    @classmethod
    def performance_stats(cls):
        stats = [(cls.nCostCalls, cls.tCostTime), \
                 (cls.nGradCalls, cls.tGradTime)]
        stats = stats + Grasp_Constraint.performance_stats()
        stats = stats + Placement_Constraint.performance_stats()
        stats = stats + Collision_Constraint.performance_stats()
        stats = stats + Reflex_Collision.performance_stats()
        return stats

    def add_bounds_as_ieqs(self):
        for i in range(self.size()):
            self.inequality_constraints.append(Bounds_IEq(i, self.min_bounds[i], self))
            self.inequality_constraints.append(Bounds_IEq(i, self.max_bounds[i], self, True))
    def joint_fk_se3(self, q, ref_frame="world"):
        q = self._process_q(q)
        assert len(q) == self.dof, "Joint array expected of length {}, got length {} instead".format(self.dof, len(q)) 
        x = self.arm_palm_KDL.fk(q)
        x_pose = convert_array_to_pose(x, "world")
        x_pose = TransformPoseStamped(x_pose, ref_frame, self.tf_b, self.tf_listener)
        return pose_to_6dof(x_pose)

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
        #return self.joint_fk_se3(q[self.G_IDX], "object_pose")
    
    def joint_to_place(self, q, asmat=False):
        """
        Return fk of grasp joints in the placement reference frame
        """
        q = self._process_q(q)
        x = self.place_kdl.fk(q[self.P_IDX])
        if asmat:
            return TransformPoseMat(x, self.ptable_world_mat)
        else:
            return mat_to_vec(TransformPoseMat(x, self.ptable_world_mat))
        #return self.joint_fk_se3(q[self.P_IDX], "place_table_corner")

    def get_pnp_diff(self, q):
        xg = self.grasp_kdl.fk(q[self.G_IDX])
        xp = self.place_kdl.fk(q[self.P_IDX])
        #print(xg)
        #print(xp)
        #print('grasp palm pose: ', convert_array_to_pose(xg, 'world'))
        #print('place palm pose: ', convert_array_to_pose(xp, 'world'))
        diff = np.array(euler_diff(xg[3:], xp[3:]))#np.array(xg) - np.array(xp)
        #print(diff)
        #diff = np.arctan2(np.sin(diff), np.cos(diff))
        #diff[:] = np.max(np.absolute(diff))
        #print(np.absolute(diff))
        return diff, euler_to_angular_velocity(diff)

    def grasp_probabilities(self, q):
        q = self._process_q(q)

        x_g = self.joint_to_grasp(q)
        #grasp_probs = self.GraspClient.query_grasp_probabilities(x_g, \
        #                                               q[self.PRESHAPE], \
        #                                               self.object_sparse_grid, \
        #                                               self.object_voxel_dims, \
        #                                               self.object_size)

        #grasp_probs = list(grasp_probs)
        grasp_conf = np.asarray(list(x_g) + list(q[self.PRESHAPE]), dtype=NP_DTYPE_)
        grasp_probs = self.GraspClient.get_grasp_probs(grasp_conf, self.Wt_Post)
        #grasp_probs[2] = -self.GraspClient.get_grasp_prior(grasp_conf)
        return grasp_probs

    def grasp_grads(self, q, idx=0):

        q = self._process_q(q)
        
        x_g_ = self.joint_to_grasp(q)
        #grasp_grads_ = self.GraspClient.query_grasp_grad(x_g_, \
        #                                                q[self.PRESHAPE], \
        #                                                self.object_sparse_grid, \
        #                                                self.object_voxel_dims, \
        #                                                self.object_size)[idx]

        grasp_conf = np.asarray(list(x_g_) + list(q[self.PRESHAPE]), dtype=NP_DTYPE_)
        grasp_grads_ = self.GraspClient.get_grasp_grads(grasp_conf, self.Wt_Post)
        
        if idx == 2:
            grasp_grads_ = self.GraspClient.get_grasp_prior_grad(grasp_conf)

        elif idx == 1:
            grasp_grads_ = self.GraspClient.get_grasp_lkh_grad(grasp_conf)
        #print(grasp_grads_.shape)
            
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

    def place_cost(self, x_p):
        return self.place_heuristic.cost(np.asarray(x_p))
    
    def corner_heuristic(self, x_p):
        cost = (x_p[0] - self.table_corner[0])**2 + (x_p[1] - self.table_corner[1])**2 #+ \
        #100 * (self.upright[0] - x_p[3])**2 + 100 * (self.upright[1] - x_p[4])**2
        return cost

    def place_grads(self, x_p, q):
        cart_grad = self.place_heuristic.grad(np.asarray(x_p))[0]
        grads_p, grads_g = self.place_position_grads(q)
        o_grads_p, o_grads_g = self.place_orientation_grads(q, full=True)
        grads_p = np.vstack([grads_p, o_grads_p])
        grads_g = np.vstack([grads_g, o_grads_g])
        return cart_grad @ grads_p, cart_grad @ grads_g * self.Wt_G
        
    def corner_heuristic_grads(self, x_p, q):
        cart_grad_ = np.zeros((1,3))
        grads_p, grads_g = self.place_position_grads(q)
        cart_grad_[0,0] = 2 * (x_p[0] - self.table_corner[0]) #* x_p_[1]**2
        cart_grad_[0,1] = 2 * (x_p[1] - self.table_corner[1])#* x_p_[0]**2
        #cart_grad_[0,2] = -2 * 100 * (self.object_size[1] - x_p[2])
        #cart_grad_[0,3] = -2 * 100 * (self.upright[0] - x_p[3])
        #cart_grad_[0,4] = -2 * 100 * (self.upright[1] - x_p[4])
        return cart_grad_ @ grads_p, cart_grad_ @ grads_g * self.Wt_G

    def joints_reg(self, q):
        q_p = q[self.P_IDX] 
        q_g = q[self.G_IDX] - self.x0[self.G_IDX].T
        return (q_p.T @ q_p).item(), 0.0#(q_g[0].T @ q_g[0]).item()
    def joints_reg_grads(self, q):
        return 2*q[self.P_IDX], 2*(q[self.G_IDX] - self.x0[self.G_IDX].T)
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


    def residual(self, q):
        return np.array([self.cost(q)])
    
    def cost(self, q, split=False):
        type(self).nCostCalls += 1
        #print(inspect.stack()[1].function+inspect.stack()[2].function)
        time0 = time.perf_counter()
        
        q = self._process_q(q)
        #return self.cost_IK(q)
    
        gTo_ = invert_trans(self.joint_to_grasp(q, asmat=True))
        #gTo_ = vector2matinv(x_g_[:3], x_g_[3:])

        tTg_ = self.joint_to_place(q, asmat=True)
        #tTg_ = vector2mat(x_p_[:3], x_p_[3:])

        tTo_ = tTg_ @ gTo_
        tx_p_ = mat_to_vec(tTo_)
        

        #place_cost_ = self.corner_heuristic(tx_p_)
        place_cost_ = self.place_cost(tx_p_)
        self.log_val("place", q.tobytes(), place_cost_)
        grasp_probs = self.grasp_probabilities(q)
        grasp_cost = grasp_probs[0]
        self.log_val("lkh", q.tobytes(), grasp_probs[1])
        self.log_val("prior", q.tobytes(), grasp_probs[2])
        reg_cost_ = self.joints_reg(q)
        #print(reg_cost_)
        self.log_val("reg_p", q.tobytes(), reg_cost_[0])
        self.log_val("reg_g", q.tobytes(), reg_cost_[1])
        
        #grasp_cost_ = -self.GraspClient.query_grasp_probabilities(x_g_, \
        #                                                q[self.PRESHAPE], \
        #                                                self.object_sparse_grid, \
        #                                                self.object_voxel_dims, \
        #                                                self.object_size)[0]

        #grasp_cost_ = -(self.Wt_Post[0] * grasp_probs[1] + self.Wt_Post[1] * grasp_probs[2])
        self.log_val("post", q.tobytes(), grasp_cost)
        type(self).tCostTime += time.perf_counter() - time0
        if split:
            return grasp_cost , place_cost_
        return place_cost_ * self.Wt_P + grasp_cost * self.Wt_G
            #reg_cost_[1] * -grasp_probs[2] * self.Wt_R #Grasp cost is already -log

    def cost_gradient_fd(self, q, delt=1e-5):
        print("warning cost gradient fd called")
        if isinstance(q, np.matrix):
            q = np.squeeze(np.asarray(q))
        q = q.reshape(-1) # Change input to row vector
        g = np.matrix(q * 0.0)
        for i in range(self.dof*2 + 1):
            q_new = copy.deepcopy(q)
            q_new[i] += delt
            cf = self.cost(q_new)
            
            q_new = copy.deepcopy(q)
            q_new[i] -= delt
            cb = self.cost(q_new)

            g[0, i] = (cf - cb) / (2*delt)
        #g = g / np.linalg.norm(g)
        return g.T

    def jacobian(self, q):
        return self.cost_gradient(q)
    
    def cost_gradient(self, q):
        type(self).nGradCalls += 1
        time0 = time.perf_counter()
        
        q = self._process_q(q)

        x_g_ = self.joint_to_grasp(q)
        gTo_ = vector2matinv(x_g_[:3], x_g_[3:])
        
        x_p_ = self.joint_to_place(q)
        tTg_ = vector2mat(x_p_[:3], x_p_[3:]) #check this

        tTo_ = tTg_ @ gTo_
        tx_p_ = mat_to_vec(tTo_)
        
        q_grads = np.matrix(q*0.0)
        q_grads[0, self.P_IDX], q_grads[0, self.G_IDX] = self.place_grads(tx_p_, q)
        #q_grads[0, self.P_IDX], q_grads[0, self.G_IDX] = self.corner_heuristic_grads(tx_p_, q)
        q_grads *= self.Wt_P

        #grasp_grads_ = self.GraspClient.query_grasp_grad(x_g_, \
        #                                                q[self.PRESHAPE], \
        #                                                self.object_sparse_grid, \
        #                                                self.object_voxel_dims, \
        #                                                self.object_size)
        #grasp_grads_ = list(grasp_grads_)
        grasp_conf = np.asarray(list(x_g_) + list(q[self.PRESHAPE]), dtype=NP_DTYPE_)
        #grasp_grads_[2] = -self.GraspClient.get_grasp_prior_grad(grasp_conf)
        #prior_grads = np.array(grasp_grads_[2])
        #grasp_grads_ = -(self.Wt_Post[0] * np.array(grasp_grads_[1]) + self.Wt_Post[1] * np.array(grasp_grads_[2]))
        grasp_grads_ = self.GraspClient.get_grasp_grads(grasp_conf, self.Wt_Post)
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
        #print(grasp_grads_[:6].shape, o_jac_g.shape)
        grasp_grads_qg = grasp_grads_[:6] @ o_jac_g
        
        q_grads[0, self.G_IDX] += grasp_grads_qg * self.Wt_G
        q_grads[0, self.PRESHAPE] += grasp_grads_[6] * self.Wt_G

        #grasp_probs = self.grasp_probabilities(q)

        reg_cost = self.joints_reg(q)
        qp_reg, qg_reg = self.joints_reg_grads(q)

        #q_grads[0, self.G_IDX] += (qg_reg*-grasp_probs[2] + reg_cost[1]*-prior_grads[:6] @ o_jac_g) * self.Wt_R
        #q_grads[0, self.P_IDX] += qp_reg * self.Wt_R
        
        GradCheck = False
        if GradCheck:
            fd_grads = self.cost_gradient_fd(q)
            #np.set_printoptions(precision=2, floatmode='maxprec_equal')
            #print('fd vs a_grads error: \n', o_jac_g_fd, o_jac_g, o_jac_g_fd - o_jac_g)
            #vecsim = vector_similarity(fd_grads, q_grads.T)
            #print('Similarity: ', np.arccos(vecsim)*180/np.pi, vecsim )
            compare_grads(fd_grads, q_grads)
        
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

    def get_approach_head(self, q, dist=-0.1):
        cart_vel = [0.0] * 6
        cart_vel[0] = dist
        cart_pose = self.arm_palm_KDL.fk(q)
        T = vector2matinv(cart_pose[:3], cart_pose[3:])
        rot, trans = decompose_transform(T)
        s_jac = self.arm_palm_KDL.space_jacobian(q, rot, trans)
        return q + s_jac.T @ np.array(cart_vel)

    def get_lift_up_joints(self, q, dist=0.1):
        cart_vel = [0.0] * 6
        cart_vel[2] = dist
        jac = self.arm_palm_KDL.jacobian(q)
        return q + jac.T @ np.array(cart_vel)

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
        
        
    
    def get_place_proto(self):
        tTo_proto = invert_trans(self.obj_world_mat)
        tTo_proto[0, 3] = np.random.uniform(0.0, self.table_size[0])
        tTo_proto[1, 3] = np.random.uniform(0.0, self.table_size[1])
        tTo_proto[2, 3] = self.object_size[2]/2 + np.random.uniform(0.0, 0.1)
        return tTo_proto
        
    def initialize_place(self, max_trials = 10, pt_list=None):
        self.x0 = self.x0.reshape(-1)
        oTg = self.joint_to_grasp(self.x0, asmat=True)
        yaws = [0.0, np.pi/2 , np.pi, 3*np.pi/2]
        qp_0_ = None
        for iter_ in range(max_trials):
            #np.random.shuffle(yaws)
            for yaw in yaws:
                tTo_proto = self.get_place_proto()
                tTo_post = vector2mat(rpy=[0.0, 0.0, yaw])
                if pt_list is not None:
                    val_pt = pt_list[iter_]
                    
                    tTo_proto[0,3] = 0#val_pt[0]
                    tTo_proto[1,3] = 0#val_pt[1]
                    #tTo_proto[2,3] = 0#val_pt[2]
                    yaw= val_pt[5]
                    tTo_post = vector2mat(val_pt[:3], val_pt[3:])
                    
                tTg = tTo_post @ tTo_proto @ oTg
                place_prior_ = mat_to_vec(invert_trans(self.ptable_world_mat) @ tTg)
                #place_prior_ = [0.0] * 4 + [1.57] + [0.0]#mat_to_vec(self.obj_world_mat)#[0.0] * 4 + [1.57] + [0.0]
                #place_prior_[0] = np.random.uniform(0.0, self.table_size[0])
                #place_prior_[1] = np.random.uniform(0.0, self.table_size[1])
                #place_prior_[2] = self.object_size[2]/2 + np.random.uniform(0.0, 0.1)
                place_prior_pose_ = convert_array_to_pose(place_prior_, 'world').pose
                #world_place_pose_ = TransformPoseStamped(place_prior_pose_,"world", self.tf_b, self.tf_listener)
                #p_trans_, p_quat_ = pose_to_array(place_prior_pose_)
                qp_0_ = get_IK(place_prior_pose_, 'left')#self.arm_palm_KDL.ik(p_trans_, p_quat_)
                if qp_0_ is None:
                    break
                else:
                    qp_0_ = np.arctan2(np.sin(qp_0_), np.cos(qp_0_))
                    self.x0[self.P_IDX] = qp_0_[:]
                    self.x0 = self.x0.reshape(-1,1)
                    xp_fk = np.array(self.joint_to_place(self.x0))
                    print('Init Place Pose Error:\n', xp_fk - np.array(place_prior_))
                    #print(f'Best feasible: {val_pt[0]}, {val_pt[1]}, {val_pt[5]}')
                    txo = self.object_pose_from_joints(self.x0)
                    tTo_ck = vector2mat(txo[:3], txo[3:])
                    print('initp', tTo_ck[2, 2])
                return True
            
        if qp_0_ is None:
            print("Unable to generate reachable grasp prior, proceeding with zero joint init")
        self.x0 = self.x0.reshape(-1)
        return False

    def initialize_grasp(self, idx=None, max_trials = 10):
        self.x0 = self.x0.reshape(-1)
        # self.GraspClient.init_grasp_nets(self.object_sparse_grid, \
        #                           self.object_voxel_dims, \
        #                           self.object_size)
        ik_t = 0
        net_t = 0
        wTo = invert_trans(self.obj_world_mat)
        for iter_ in range(max_trials):
            #grasp_prior_ = self.GraspClient.query_grasp_prior(self.object_sparse_grid, \
            #                                                  self.object_voxel_dims, \
            #                                                  self.object_size)
            #if iter_ < 2:
                #grasp_prior_ = self.GraspClient.get_mean(2)
            #else:
            t0 = time.time()
            if idx is None:
                grasp_prior_ = self.GraspClient.sample_prior()
            else:
                grasp_prior_ = self.GraspClient.sample_explicit(idx)
            self.raw_prior = grasp_prior_[:6]
            #prior_cost_ = -self.GraspClient.query_grasp_probabilities(grasp_prior_[:6], \
            #                                            [grasp_prior_[6]], \
            #                                            self.object_sparse_grid, \
            #                                            self.object_voxel_dims, \
            #                                            self.object_size)[0]

            prior_cost_ = self.GraspClient.get_grasp_prior(grasp_prior_)
            t1 = time.time()
            
            #grasp_prior_pose_ = convert_array_to_pose(grasp_prior_[:6], "object_pose")
            world_grasp_pose_ = mat_to_vec(TransformPoseMat(grasp_prior_, wTo))
            world_grasp_pose_ = convert_array_to_pose(world_grasp_pose_, "world").pose
            #g_trans_, g_quat_ = pose_to_array(world_grasp_pose_)
            #self.xt = pose_to_6dof(world_grasp_pose_)
            
            #qg_0_ = self.arm_palm_KDL.ik(g_trans_, g_quat_)
            qg_0_ = get_IK(world_grasp_pose_, 'left')
            t2= time.time()
            ik_t += t2 - t1
            net_t += t1 - t0
            if qg_0_ is None:
                print("Unreachable grasp, retrying...")
            else:
                qg_0_f_ = np.arctan2(np.sin(qg_0_), np.cos(qg_0_))
                qg_0_ = qg_0_f_
                self.x0[self.G_IDX] = qg_0_[:]
                self.x0[self.PRESHAPE] = grasp_prior_[6]
                xg_0_ = self.joint_to_grasp(self.x0)
                #prior_cost_fk_ = -self.GraspClient.query_grasp_probabilities(xg_0_, \
                #                                        [grasp_prior_[6]], \
                #                                        self.object_sparse_grid, \
                #                                        self.object_voxel_dims, \
                #                                        self.object_size)[0]
                _, lkh, prior_cost_fk_ = self.GraspClient.get_grasp_probs(grasp_prior_, self.Wt_Post)
                #print("Received Prior: {}\n After IK FK and Trans: {}".format(grasp_prior_[3:6], xg_0_[3:6]))
                #print("Equivalence: ", test_3d_rotation_equivalence(grasp_prior_[3:6], xg_0_[3:6]))
                print("Cost at Prior: {}\n After IK: {}\n Error: {}".format(prior_cost_, prior_cost_fk_, prior_cost_ - prior_cost_fk_))
                if (prior_cost_fk_ < -3):
                    self.x0 = self.x0.reshape(-1,1)
                    return
                else:
                    print('Found feasible but bad grasp, retrying...')
        if qg_0_ is None:
            print("Unable to generate reachable grasp prior, proceeding with zero joint init")
            self.x0 = self.x0 * 0
        print(f'ik time: {ik_t}. net time: {net_t}')
        self.x0 = self.x0.reshape(-1)

    def gripper_position_grads(self, q, gripper_pts):
        '''
        Verified works
        '''
        q = self._process_q(q)

        tRw, tPw = decompose_transform(self.ptable_world_mat)

        oRw, oPw = decompose_transform(self.obj_world_mat)
        
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

        oJRg_ = oRw @ batch_vector_skew_mat(omega_g) @ wRg
        q_grads_g = (oJRg_ @ gripper_pts.T).T + oRw @ linvel_g
        
        tJRg_ = tRw @ batch_vector_skew_mat(omega_p) @ wRg
        q_grads_p = (tJRg_ @ gripper_pts.T).T + tRw @ linvel_p

        return q_grads_p, q_grads_g

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

    def place_orientation_grads(self, q, full=False):
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

        if not full:
            return tJRo_p[:,2,2].T, tJRo_g[:,2,2].T
        
        t_omega_g = tRg @ gJRw @ gRw.T @ tRg.T

        t_omega_p = tRw @ batch_vector_skew_mat(omega_p) @ tRw.T

        g_rpy_jac = []
        p_rpy_jac = []
        for i in range(t_omega_g.shape[0]):
            t_omega = t_omega_g[i]
            t_omega_v = np.array([t_omega[2][1], t_omega[0][2], t_omega[1][0]])
            g_rpy_dot = angular_velocity_to_rpy_dot(t_omega_v, txo[3:])
            #print('dot', g_rpy_dot)
            g_rpy_jac.append(g_rpy_dot)

            t_omega = t_omega_p[i]
            t_omega_v = np.array([t_omega[2][1], t_omega[0][2], t_omega[1][0]])
            p_rpy_dot = angular_velocity_to_rpy_dot(t_omega_v, txo[3:])
            p_rpy_jac.append(p_rpy_dot)
        #print('jac', g_rpy_jac)
        g_rpy_jac = np.vstack(g_rpy_jac).T
        p_rpy_jac = np.vstack(p_rpy_jac).T

        return p_rpy_jac, g_rpy_jac

    def object_pose_from_joints(self, q):
        q = self._process_q(q)

        gTo = invert_trans(self.joint_to_grasp(q, asmat=True))
        tTg = self.joint_to_place(q, asmat=True)
        tTo = tTg @ gTo
        
        return mat_to_vec(tTo)

    def add_collision_constraints(self):
        Collision_Constraint.static_init(self)
        Reflex_Collision.static_init(self)
        Grasp_Collision.static_init(self)
        obj_pts = self.Obj_Grid.binned_pts#_raw_grid_idx_to_pts()
        gripper_pts = Reflex_Collision.get_gripper_points()
        publish_as_point_cloud(gripper_pts, topic='reflex_binned', frame_id = 'reflex_palm_link')
        num_pts = obj_pts.shape[0]
        print(f"Adding {num_pts} object collision checks per object")
        num_gpts = gripper_pts.shape[0]
        print(f"Adding {num_gpts} robot collision checks per object")
        if self.Wt_G:
            for j in range(num_gpts):
                self.inequality_constraints.append(Grasp_Collision(j, 'grasp_obj'))
                for key in self.GEnv_List.keys():
                    self.inequality_constraints.append(Grasp_Collision(j, key))
                break
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
        return fitness

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
        publish_env_cloud(self.Env_List)

    def get_env_poses(self):
        obj_poses = {}
        for key in self.Env_List.keys():
            obj_poses[key] = convert_array_to_pose(mat_to_vec(self.Env_List[key].Trans),\
                                                   self.place_frame).pose
        return obj_poses

    def new_id(self):
        return str(len(self.Env_List))

    def plot_joint_iterates(self, iterates):
        g_joints = iterates[:, self.G_IDX]
        p_joints = iterates[:, self.P_IDX]
        iter_range = range(len(iterates))
        jlo, jhi = self.arm_palm_KDL.get_joint_limits()
        colors = ['#7e1e9c', # purple
                  '#15b01a', # green
                  '#f97306', # orange
                  '#e50000', # red
                  '#0165fc', # blue
                  '#dbb40c', # mustard
                  '#fe01b1'  # pink
        ]
        for i in range(7):
            pyplot.plot(iter_range, [jlo[i]] * len(iterates), color=colors[i], linestyle='dotted', label="joint_" + str(i+1) + "_limit")
            pyplot.plot(iter_range, [jhi[i]] * len(iterates), color=colors[i], linestyle='dotted', label="joint_" + str(i+1) + "_limit")
            pyplot.plot(iter_range, g_joints[:,i], color=colors[i], linestyle='-', label="grasp_joint_" + str(i+1))
        pyplot.legend()
        pyplot.show()
        for i in range(7):
            pyplot.plot(iter_range, [jlo[i]] * len(iterates), color=colors[i], linestyle='dotted', label="joint_" + str(i+1) + "_limit")
            pyplot.plot(iter_range, [jhi[i]] * len(iterates), color=colors[i], linestyle='dotted', label="joint_" + str(i+1) + "_limit")
            pyplot.plot(iter_range, p_joints[:,i], color=colors[i], linestyle='-', label="place_joint_" + str(i+1))
        pyplot.legend()
        pyplot.show()

    def delete(self):
            Grasp_Collision.pblm = None
            Reflex_Collision.pblm = None
            Collision_Constraint.pblm = None


    def project_joint_limits(self, q):
        joint_lows = np.array(self.min_bounds) * 0.9
        #joint_lows[-1] = self.min_bounds[-1]
        joint_lows = self._process_q(joint_lows)
        joint_highs = np.array(self.max_bounds) * 0.9
        joint_highs = self._process_q(joint_highs)
        #joint_highs[-1] = self.max_bounds[-1]
        q[:14] = np.maximum(q,joint_lows)[:14]
        q[:14] = np.minimum(q,joint_highs)[:14]
        return q
        
    def sub_sample(self, num_samples=100):
        print('Refining')
        q = self._process_q(self.x0)
        q = self.project_joint_limits(q)
        oTg = self.joint_to_grasp(self.x0, asmat=True)
        grasp_cost, place_cost = self.cost(q, split=True)
        fitness = max(self.evaluate_constraints(q).values())
        tx_p_ = self.object_pose_from_joints(q)
        wTt = invert_trans(self.ptable_world_mat)
        for i in range(num_samples):
            tx = np.zeros(6)
            tx[:2] = np.random.uniform(-1e-2, 1e-2, 2)
            tx[5] = np.random.uniform(-np.pi/2, np.pi/2)
            tx += tx_p_
            #tx[2] = max(self.object_size[2]/2, tx[2])
            #tx[2] = min(1e-2, tx[2])
            if self.place_cost(tx) > place_cost:
                print('Not better...')
                continue
            tTo = vector2mat(tx[:3], tx[3:])
            tTg = tTo @ oTg
            place_prior_ = mat_to_vec(wTt @ tTg)
            world_place_pose_ = convert_array_to_pose(place_prior_, "world").pose
            #world_place_pose_ = TransformPoseStamped(place_prior_pose_,"world", self.tf_b, self.tf_listener)
            p_trans_, p_quat_ = pose_to_array(world_place_pose_)
            qp_0_ = get_IK(world_place_pose_, 'left')#self.arm_palm_KDL.ik(p_trans_, p_quat_)
            if qp_0_ is None:
                print('Not reachable...')
                continue
            qp_0_ = np.arctan2(np.sin(qp_0_), np.cos(qp_0_))
            q[self.P_IDX] = qp_0_[:]
            q = self.project_joint_limits(q)
                #self.x0 = self.x0.reshape(-1,1)
                #xp_fk = np.array(self.joint_to_place(self.x0))
                #print('Init Place Pose Error:\n', xp_fk - np.array(place_prior_))
                #print('Best feasible: {val_pt[0]}, {val_pt[1]}, {val_pt[5]}')
            txo = self.object_pose_from_joints(q)
            sub_fit = max(self.evaluate_constraints(q).values())
            sub_grasp_cost, sub_place_cost = self.cost(q, split=True)
            if sub_fit < max(fitness, 1e-3) and sub_grasp_cost < grasp_cost and sub_place_cost < place_cost:
                print(f'Found better sub sample at offset f{txo - tx_p_}')
                tx_p_ = txo
                fitness = sub_fitness
                grasp_cost = sub_grasp_cost
                place_cost = sub_place_cost
                #tTo_ck = vector2mat(txo[:3], txo[3:])
                #print('initp', tTo_ck[2, 2])
        return q


    def validate_grasp_collision(self, q):
        err = -1000
        for cc in self.inequality_constraints:
            if type(cc) == Grasp_Collision:
               err = max(cc.error_core(q), err)
        return err
                
        
        
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
        
