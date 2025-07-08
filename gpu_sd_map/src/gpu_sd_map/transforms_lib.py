#!/usr/bin/env python
'''
Library of useful functions related to homogeneous transformations
'''

import numpy as np

def rot2mat(rot, translation = [0.0, 0.0, 0.0]):
    '''
    Given a 3 x 3 rotation matrix, make it a homogeneous transform
    
    Params:
    rot - 3 x 3 rotation matrix (float)
    translation - 1 x 3 translation (float)(optinal)
    
    Output:
    4 x 4 homogeneous transform

    Status: Working
    
    Testing: Structure Verified
    '''
    assert rot.shape == (3, 3), "The Rotation matix is of incorrect shape {}, expected (3, 3)".format(rot.shape)
    transform = np.eye(4)
    transform[0:3, 3] = translation
    transform[0:3, 0:3] = rot
    return transform

def axis_planar_projection(axis):
    cardinals = { "x":0, "y":1, "z":2 }
    plane = [0, 1, 2]
    if isinstance(axis, str):
        axis = cardinals[axis]
    plane.remove(axis)
    return plane, axis

def rotation_element_assignment_(c, s, axis):
    '''
    Assigns values to identity matrix to get rotation matrix. Based on elementary rotations
    are just a combination of 2 values (sin and cos) with varied negations
        
    Params:
    c - cos value (use -sin for derivate matrix)
    s - sin value (use cos for derivate matrix)
    axis - str (x/y/z) or int (0 - 2)
        
    Output:
    3 x 3 rotation matrix

    Status: Working

    Testing: Verified
    
    TODO:
    - Eliminate the if else for negating sin, find a logical approach
    '''

    cardinals = { "x":0, "y":1, "z":2 }
    plane = [0, 1, 2]
    if isinstance(axis, str):
        axis = cardinals[axis]
    plane.remove(axis)
    plane, axis = axis_planar_projection(axis)
    R = np.eye(3)
    R[plane[0],plane[0]] = c
    R[plane[1],plane[1]] = c
    R[plane[0],plane[1]] = s
    R[plane[1],plane[0]] = s
    if (plane[0],plane[1]) == (0,2):
        R[plane[1],plane[0]] = -R[plane[1],plane[0]]
    else:
        R[plane[0],plane[1]] = -R[plane[0],plane[1]]
    return R

def get_elementary_rotation_(theta, axis):
    '''
    Outputs elementary rotation matrix for the given cardinal axis
        
    Params:
    theta - angle in radians
    axis - str (x/y/z) or int (0 - 2)
        
    Output:
    3 x 3 rotation matrix
        
    Status: Working

    Testing: Verified
        
    TODO:
    - Eliminate the if else for negating sin, find a logical approach
    '''
        
    R = rotation_element_assignment_(np.cos(theta), np.sin(theta), axis)
    return R

def get_rotation_matrix(r, p, y):
    '''
    Builds 3d rotation matrix for give roll, pitch, yaw
        
    Params:
    r - roll (radians)
    p - pitch (radians)
    y - yaw (radians)
    
    Output:
    3 x 3 matrix of 3d rotation matrix
    
    Status: Working

    Testing: Verified
    '''
    
    #Rotation at x
    Rx = get_elementary_rotation_(r, 'x')
    #Rotation at y
    Ry = get_elementary_rotation_(p, 'y')
    #Rotation at z
    Rz = get_elementary_rotation_(y, 'z')
    
    rotation = Rz @ Ry @ Rx #Why was it in this order again? add reason in journal
    return rotation

def vector2mat(translation = [0.0, 0.0, 0.0], rpy = [0.0, 0.0, 0.0]):
    transform = np.eye(4)
    transform[0:3, 3] = translation
    transform[0:3, 0:3] = get_rotation_matrix(rpy[0], rpy[1], rpy[2])
    return transform

def vector2matinv(translation = [0.0, 0.0, 0.0], rpy = [0.0, 0.0, 0.0]):
    transform = np.eye(4)
    rotation = get_rotation_matrix(rpy[0], rpy[1], rpy[2])
    transform[0:3, 3] = - rotation.T @ translation
    transform[0:3, 0:3] = rotation.T
    return transform

def vector_similarity(a, b):
    '''
    Returns a score between [-1, 1]
    If result:
    -> 0 : The vectors are not similar and are orthogonal to each other
    -> 1 : The vectors are similar or in the same direction
    ->-1 : The vectors are in oppisite directions
    Performs dot produnt on the unit vectors of a,b
    '''
    a = a.reshape(1,-1)
    b = b.reshape(-1,1)
    assert a.shape[1] == b.shape[0], "The vector lengths {} , {} do not match".format(a.shape, b.shape)
    al = np.linalg.norm(a)
    bl = np.linalg.norm(b)
    if al * bl == 0:
        return 0
    return np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b))

def jacobian_similarity(j1, j2):
    res = np.zeros((2,j1.shape[1]))
    for i in range(j1.shape[1]):
       res[0,i] = vector_similarity(j1[:3,i],j2[:3,i])
       res[1,i] = vector_similarity(j1[3:,i],j2[3:,i])
    return res
        

def get_derivate_elementary_rotation(theta, axis):
    '''
    Outputs derivate of elementary rotation for the cardinal axis
    
    Params:
    theta - angle in radians
    axis - str (x/y/z) or int (0 - 2)
        
    Output:
    3 x 3 rotation matrix
        
    Status: Working

    Testing: Downstream Verified
    '''
    cardinals = { "x":0, "y":1, "z":2 }
    if isinstance(axis, str):
        axis = cardinals[axis]
    R = rotation_element_assignment_(-np.sin(theta), np.cos(theta), axis)
    R[axis][axis] = 0 #rotation_element_assignment_() assigns on an identity base
    return R

def get_rotation_jacobian(r, p, y):
    '''
    Generate the point wise jacobian for each rotation axis
        
    Params:
    r - roll (radians)
    p - pitch (radians)
    y - yaw (radians)
    
    Output:
    (3 x 3) x 3 vector of rotation matrices
        
    Status: Working

    Testing: Downstream Verified

    TODO:
    - Optimal output format?
    '''
    cardinals = { "x":0, "y":1, "z":2 }
    #Rotation at x
    Rx = get_elementary_rotation_(r, 'x')
    dRx = get_derivate_elementary_rotation(r, 'x')
    dRx[0][0] = 0
    #Rotation at y
    Ry = get_elementary_rotation_(p, 'y')
    dRy = get_derivate_elementary_rotation(p, 'y')
    dRy[1][1] = 0
    #Rotation at z
    Rz = get_elementary_rotation_(y, 'z')
    dRz = get_derivate_elementary_rotation(y, 'z')
    dRz[2][2] = 0
    
    JRx = Rz @ Ry @ dRx
    JRy = Rz @ dRy @ Rx
    JRz = dRz @ Ry @ Rx
    
    return (JRx, JRy, JRz)

def verify_rpy_dot(rpy_dot, omega, rpy):
    JR = np.array(get_rotation_jacobian(rpy[0], rpy[1], rpy[2]))
    JR = np.einsum('ijk,i->jk', JR, rpy_dot)
    JRRT = JR @ get_rotation_matrix(rpy[0], rpy[1], rpy[2]).T
    return abs(JRRT - omega)

def euler_to_angular_velocity(rpy, dt=10):
    JR = np.array(get_rotation_jacobian(rpy[0], rpy[1], rpy[2]))
    rpy_dot = np.array(rpy)
    JR = np.einsum('ijk,i->jk', JR, rpy_dot)
    JRRT = JR @ get_rotation_matrix(rpy[0], rpy[1], rpy[2]).T
    #print(JRRT)
    return np.array([JRRT[2][1], JRRT[0][2], JRRT[1][0]])

def angular_velocity_to_rpy_dot(omega, rpy):
    '''
    This functions converts angular velocities - omega (3 x 1) to
    rpy angle rates (3 x 1). Using the relation:

    rpy_dot = A_inv @ omega

                  [[ c_g/c_b       s_g/c_b       0 ]

    where A_inv =  [-s_g           c_g           0 ]

                   [ c_g*s_b/c_b   s_g*s_b/c_b   1 ]]

    Derieved from the relation
    R_dot @ R.T = [omega]
    R = Rz(g) @ Ry(b) @ Rx(a)

    Since A_inv doesn't have full rank, it has a degenrate case
    when c_b = 0 
    '''

    a,b,g = rpy

    c_b = np.cos(b)
    s_b = np.sin(b)
    c_g = np.cos(g)
    s_g = np.sin(g)

    if c_b == 0:
        A_inv = np.array([[ 0.0     ,  0.0,  0.0 ], \
                          [ -1.0/s_g,  0.0,  0.0 ], \
                          [ 0.0     ,  0.0,  1.0 ]])
        if s_g == 0:
            A_inv[1,0] = 0.0
            A_inv[1,1] = 1.0/c_g

    else:
        A_inv = np.array([[ c_g/c_b    ,  s_g/c_b    ,  0.0 ], \
                          [ -s_g       ,  c_g        ,  0.0 ], \
                          [ c_g*s_b/c_b,  s_g*s_b/c_b,  1.0 ]])

    # Wrap angles to range [-pi, pi]
    #print(A_inv)
    rpy_dot = A_inv @ omega
    rpy_dot = np.arctan2(np.sin(rpy_dot), np.cos(rpy_dot))

    return A_inv @ omega
    
def invert_trans(mat):
    ret = np.eye(4)
    rot = mat[0:3, 0:3].T
    trans = -rot @ mat[0:3, 3]
    ret[0:3, 3] = trans
    ret[0:3, 0:3] = rot
    return ret

def transform_points(points, transform):
    '''
    Applies transforms to the set of 3D points

    Params:
    points - N x 3 array of points in 3D (float).
    transform - 4 x 4 homogeneous transform (float).

    Output:
    trans_points - N x 3 array of points in 3D (float).
    
    Status: Implemented

    Testing:
    '''
    assert points.shape[1] == 3, "points are not in 3D, Expected (N, 3), received {}".format(points.shape)
    assert transform.shape == (4, 4), "transform dimension is not (4, 4), received {}".format(transform.shape)
    points_1_ = np.append(points[:], np.ones([points.shape[0], 1]), 1)
    #print(points_1_.shape)
    trans_points = transform.dot(points_1_.T).T[:,:3]
    return trans_points

def decompose_transform(mat):
    return mat[:3,:3].reshape(3,3), mat[:3,3].reshape(3,1)
    
    
def skew_symmetric_mat(vec):
    sk_ = np.zeros((3,3))
    sk_[0,2] = vec[1]
    sk_[2,0] = -sk_[0,2]
    sk_[1,0] = vec[2]
    sk_[0,1] = -sk_[1,0]
    sk_[2,1] = vec[0]
    sk_[1,2] = -sk_[2,1]
    return sk_

def skew_symmetric_mat_einsum(vec):
    a = np.eye(3)
    r = np.array([[0,1,0],\
                  [0,0,1],\
                  [1,0,0]])
    a = np.einsum('ij,i->ij', a, vec)
    a = np.einsum('ij,jk -> ki',a,r )
    a = np.einsum('ij,jk -> ki',a,r.T )
    return a - a.T

def batch_vector_skew_mat(vec):
    A = np.eye(3)
    R = np.array([[0,1,0],\
                  [0,0,1],\
                  [1,0,0]])

    assert vec.shape[0] == 3 or vec.shape[1] == 3, \
        "Expected input batch of vectors of shape 3xN "
    if vec.shape[0] != 3:
        vec = vec.T
    vec = vec.reshape(3,-1)
    
    SV = np.einsum('ij,ik -> kij', A, vec)
    SV = np.einsum('aij,jk -> aki', SV, R)
    SV = np.einsum('aij,jk -> aki', SV, R.T)
    return SV - np.einsum('ijk->ikj', SV)

def TransformPoseMat(xarr, trans):
    return trans @ vector2mat(xarr[:3], xarr[3:])


def yaw_mat(mat, yaw):
    rtrans = rot2mat(get_rotation_matrix(0.0, 0.0, yaw))
    pos = mat[:3, 3]
    res = rtrans @ mat
    res[:3, 3] = pos
    return res
    
    
