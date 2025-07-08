import PyKDL
from kdl_parser_py.urdf import treeFromParam
from urdf_parser_py.urdf import URDF
import rospy
import numpy as np
import logging

from gpu_sd_map.ros_transforms_lib import batch_vector_skew_mat
from gpu_sd_map.transforms_lib import angular_velocity_to_rpy_dot

def pos_quat_to_kdl(pos, quat):
    kdl_rot_ = PyKDL.Rotation.Quaternion(quat[0], quat[1], quat[2], quat[3])
    kdl_vec_ = PyKDL.Vector(pos[0], pos[1], pos[2])
    return PyKDL.Frame(kdl_rot_, kdl_vec_)

def kdl_frame_to_list(frame):
    out = list(frame.p)
    out = out + list(frame.M.GetRPY())
    return out

def kdl_jac_to_mat(jac, mat):
    assert mat.shape == (jac.rows(), jac.columns())
    for i_ in range(jac.rows()):
        for j_ in range(jac.columns()):
            mat[i_, j_] = jac[i_ ,j_]
    return mat

def pose_difference(pose1, pose2):
    f1 = pos_quat_to_kdl(pose1[:3], pose1[3:])
    f2 = pos_quat_to_kdl(pose2[:3], pose2[3:])
    fd = PyKDL.diff(f1, f2)
    return list([fd.vel.x(), fd.vel.y(), fd.vel.z()]) + list([fd.rot.x(), fd.rot.y(), fd.rot.z()])

class ChainKDL:
    def __init__(self, chain):
        self.chain = chain
        self.FK = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self.Jacobian = PyKDL.ChainJntToJacSolver(self.chain)
        self.IK_Vel = PyKDL.ChainIkSolverVel_pinv(self.chain)
        self.IK = PyKDL.ChainIkSolverPos_NR(self.chain, self.FK, self.IK_Vel)
        self.jin_ = PyKDL.JntArray(self.chain.getNrOfJoints())
        self.jout_ = PyKDL.JntArray(self.chain.getNrOfJoints())
        self.JSEED_ = PyKDL.JntArray(self.chain.getNrOfJoints())
        self.out_frame_ = PyKDL.Frame()
        self.jac_output_ = PyKDL.Jacobian(self.chain.getNrOfJoints())
        self.jac_mat_ = np.zeros((6, self.chain.getNrOfJoints()))

    def _validate_assign_joint_in(self, q):
        assert len(q) == self.chain.getNrOfJoints(), "Input number of joint angles doesn't match"
        for i in range(self.chain.getNrOfJoints()):
            self.jin_[i] = q[i]

    def forward_frame(self, q):
        self.out_frame_ = self.out_frame_.Identity()
        self._validate_assign_joint_in(q)
        self.FK.JntToCart(self.jin_, self.out_frame_)
        return self.out_frame_
            
    def forward(self, q):
        self.out_frame_ = self.out_frame_.Identity()
        self._validate_assign_joint_in(q)
        self.FK.JntToCart(self.jin_, self.out_frame_)
        return kdl_frame_to_list(self.out_frame_)
    
    def get_jacobian(self, q, mat=True):
        self._validate_assign_joint_in(q)
        self.jac_output_ = PyKDL.Jacobian(self.chain.getNrOfJoints())
        self.Jacobian.JntToJac(self.jin_, self.jac_output_)
        if not mat:
            return self.jac_output_
        kdl_jac_to_mat(self.jac_output_, self.jac_mat_)
        return self.jac_mat_

    def inverse(self, pos, quat):
        self.jout_ = PyKDL.JntArray(self.chain.getNrOfJoints())
        kdl_frame_ = pos_quat_to_kdl(pos, quat)
        if self.IK.CartToJnt(self.JSEED_, kdl_frame_, self.jout_) >= 0:
            return [self.jout_[i] for i in range(self.jout_.rows())]
        return None
        
        
class ManipulatorKDL:
    def __init__(self, robot_description = "robot_description", base_link="world", end_link="reflex_palm_link"):
        self.base_link = base_link
        self.end_link = end_link
        with suppress(suppress_stderr=True):
            self.tree = treeFromParam(robot_description)
            self.main_chain = self.create_chain(base_link, end_link)
            self.arm_urdf = URDF.from_parameter_server(robot_description)

    def fk(self, q):
        return self.main_chain.forward(q)

    def ik(self, x, q):
        return self.main_chain.inverse(x, q)

    def jacobian(self, q):
        return self.main_chain.get_jacobian(q)

    def get_dof(self):
        return self.main_chain.chain.getNrOfJoints()

    def create_chain(self, base_link="world", end_link="reflex_palm_link"):
        chain_ = self.tree[1].getChain(base_link, end_link)
        return ChainKDL(chain_)

    def get_joint_limits(self):
        lower_limits = [0.0] * self.main_chain.chain.getNrOfJoints()
        upper_limits = [0.0] * self.main_chain.chain.getNrOfJoints()
        joint_names_ = self.arm_urdf.get_chain(self.base_link, self.end_link, fixed = False, links = False)
        for id_, jn_ in enumerate(joint_names_):
            joint_ = self.arm_urdf.joint_map[jn_]
            if joint_.limit is not None:
                lower_limits[id_] = joint_.limit.lower
                upper_limits[id_] = joint_.limit.upper
            else:
                rospy.logwarn("Joint limit not set in urdf, defaulting to 0 limit (joint frozen)")
        return lower_limits, upper_limits

    def get_joint_names(self):
        joint_names = self.arm_urdf.get_chain(self.base_link, self.end_link, fixed = False, links = False)
        return joint_names

    def get_joint_by_name(self, jn):
        return self.arm_urdf.joint_map[jn]


    def space_jacobian(self, q, rot, trans):
        jac = self.jacobian(q)
        b_omega = jac[3:]
        b_linvel = jac[:3]
        s_omega = rot @ batch_vector_skew_mat(b_omega) @ rot.T
        s_rpy_jac = []
        for omega in s_omega[:]:
            s_omega_v = np.array([omega[2][1], omega[0][2], omega[1][0]])
            rpy_dot = angular_velocity_to_rpy_dot(s_omega_v, [0., 0., 0.])
            s_rpy_jac.append(rpy_dot)
        s_rpy_jac = np.vstack(s_rpy_jac).T
        s_jac = np.vstack([rot @ b_linvel, s_rpy_jac])
        return s_jac
        


class suppress:
    def __init__(self, *, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        import sys, os
        devnull = open(os.devnull, "w")

        # Suppress streams
        if self.suppress_stdout:
            self.original_stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self.original_stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args, **kwargs):
        import sys
        # Restore streams
        if self.suppress_stdout:
            sys.stdout = self.original_stdout

        if self.suppress_stderr:
            sys.stderr = self.original_stderr
