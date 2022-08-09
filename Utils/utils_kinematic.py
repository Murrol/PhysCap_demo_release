import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, axis_angle_to_quaternion,\
    quaternion_to_axis_angle, quaternion_multiply, quaternion_raw_multiply, quaternion_to_matrix, \
    matrix_to_euler_angles, quaternion_invert, quaternion_apply

# from PyIK import BVH
# from PyIK import Quaternions
# from PyIK import Animation as PyAnimation


def create_dir(dst_dir):
    is_exists = os.path.exists(dst_dir)
    if not is_exists:
        os.makedirs(dst_dir)
    return is_exists


# Write Data
def write_pkl(pkl_file, data):
    with open(pkl_file, 'wb') as fw:
        pickle.dump(data, fw)
    return data


# Read Data
def read_pkl(pkl_file):
    with open(pkl_file, 'rb') as fr:
        data = pickle.load(fr)
    return data


# def concatenate_bvh(bvh_files, concatenate_bvh_file, s=200, e=-200):
#     bvh_file = bvh_files[0]
#     animations = Animations(bvh_file)
#     orients = animations.orients
#     offsets = animations.offsets
#     parents = animations.parents
#     positions = animations.local_positions
#     rotations = animations.local_rotations
#     frame_num = animations.local_positions.shape[0]
#     positions = positions[s:e]
#     rotations = rotations[s:e]
#     for bvh_file in bvh_files:
#         animations = Animations(bvh_file)
#         frame_num = animations.local_positions.shape[0]
#         positions = np.concatenate([positions, animations.local_positions[s:e]], axis=0)
#         rotations = np.concatenate([rotations, animations.local_rotations[s:e]], axis=0)
#     rotations = Quaternions.Quaternions(rotations)
#     anim = PyAnimation.Animation(rotations, positions, orients, offsets, parents)
#     BVH.save(concatenate_bvh_file, anim, frametime=1.0 / 30.0, names=animations.names)


def load_multiple_bvh(bvh_files):
    multi_animations = {}
    for bvh_file in bvh_files:
        # print('here', bvh_file)
        if '\\' in bvh_file:
            action = bvh_file.split('\\')[-1].split('_')[0]  # walk3
        else:
            action = bvh_file.split('/')[-1].split('_')[0]  # walk3
        # print(action)
        multi_animations[action] = Animations(bvh_file)
        # print(action, multi_animations[action].names, multi_animations[action].frametime)
    return multi_animations


class Animations:
    def __init__(self, bvh_file):
        animations_src, names, frametime = BVH.load(bvh_file)
        self.animations_src = animations_src
        self.names = names
        self.frametime = frametime
        self.lens = animations_src.rotations.shape[0]
        self.offsets = torch.Tensor(animations_src.offsets)  # (J, 3) ndarray
        self.orients = torch.Tensor(animations_src.orients.qs)  # (J) Quaternions
        self.parents = torch.LongTensor(animations_src.parents)  # [-1  0  1  2  3  0  5  6  7  0  9 10 11 12 11 14 15 16 11 18 19 20]
        self.local_rotations = torch.Tensor(animations_src.rotations.qs)  # (F, J) Quaternions
        self.local_positions = torch.Tensor(animations_src.positions) # (F, J, 3) ndarray
        # self.world_rotations = PyAnimation.rotations_global(animations_src)  # (F, J) Quaternions
        # self.world_positions = PyAnimation.positions_global(animations_src)  # (F, J, 3) ndarray


def length(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a tensor axis(axes)

    :param x: tensor
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = torch.sqrt(torch.sum(x * x, dim=axis, keepdims=keepdims))
    return lgth


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)

    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = F.normalize(x, dim=axis)
    return res


def quat_normalize(x, eps=1e-8):
    """
    Normalizes a quaternion tensor

    :param x: data tensor
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quaternions tensor
    """
    res = normalize(x, eps=eps)
    return res


def angle_axis_to_quat(angle, axis): ###TODO
    """
    Converts from and angle-axis representation to a quaternion representation

    :param angle: angles tensor
    :param axis: axis tensor
    :return: quaternion tensor
    """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q


def euler_to_quat(e, order='zyx', world=False):
    """

    Converts from an euler representation to a quaternion representation

    :param e: euler tensor, radians
    :param order: order of euler rotations
    :return: quaternion tensor
    """
    if world:
        e = torch.flip(e, dims=(-1,))
        order = order[::-1]

    mat = euler_angles_to_matrix(e, convention=order.upper())
    quat = matrix_to_quaternion(mat)
    return quat

def quat_inv(q):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    res = quaternion_invert(q)
    return res


def quat_fk(lrot, lpos, parents): ###TODO
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(quat_mul(gr[parents[i]], lrot[..., i:i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)
    return res


def quat_ik(grot, gpos, parents): ###TODO
    """
    Performs Inverse Kinematics (IK) on global quaternions and global positions to retrieve local representations

    :param grot: tensor of global quaternions with shape (..., Nb of joints, 4)
    :param gpos: tensor of global positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of local quaternion, local positions
    """
    res = [
        torch.cat([
            grot[..., :1, :],
            quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ], dim=-2),
        torch.cat([
            gpos[..., :1, :],
            quat_mul_vec(
                quat_inv(grot[..., parents[1:], :]),
                gpos[..., 1:, :] - gpos[..., parents[1:], :]),
        ], dim=-2)
    ]

    return res


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    res =  quaternion_raw_multiply(x, y)

    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    # print(x.size)
    res = quaternion_apply(q, x)

    return res


def quat_slerp(x, y, a): ###TODO
    """
    Perfroms spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor
    :param y: quaternion tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """
    len = np.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = np.zeros_like(x[..., 0]) + a
    amount0 = np.zeros(a.shape)
    amount1 = np.zeros(a.shape)

    linear = (1.0 - len) < 0.01
    omegas = np.arccos(len[~linear])
    sinoms = np.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = np.sin(a[~linear] * omegas) / sinoms
    res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y

    return res


def quat_between(x, y):
    """
    Quaternion rotations between two 3D-vector arrays

    :param x: tensor of 3D vectors
    :param y: tensor of 3D vetcors
    :return: tensor of quaternions
    """
    # print(x.shape, y.shape)
    x = x.type(y.dtype)
    x, y = torch.broadcast_tensors(x, y)
    res = torch.cat([
        torch.sqrt(torch.sum(x * x, dim=-1) * torch.sum(y * y, dim=-1)).unsqueeze(-1) +
        torch.sum(x * y, dim=-1).unsqueeze(-1),
        torch.cross(x, y)], dim=-1)
    return res


def quat_to_euler(q, order='zyx'):
    mat = quaternion_to_matrix(q)
    es = matrix_to_euler_angles(mat, convention=order.upper())
    return es


def quat_to_matrix(qs):
    m = quaternion_to_matrix(qs)
    return m


def interpolate_local(lcl_r_mb, lcl_q_mb, n_past, n_future): ###TODO
    """
    Performs interpolation between 2 frames of an animation sequence.

    The 2 frames are indirectly specified through n_past and n_future.
    SLERP is performed on the quaternions
    LERP is performed on the root's positions.

    :param lcl_r_mb:  Local/Global root positions (B, T, 1, 3)
    :param lcl_q_mb:  Local quaternions (B, T, J, 4)
    :param n_past:    Number of frames of past context
    :param n_future:  Number of frames of future context
    :return: Interpolated root and quats
    """
    # Extract last past frame and target frame
    start_lcl_r_mb = lcl_r_mb[:, n_past - 1, :, :][:, None, :, :]  # (B, 1, J, 3)
    end_lcl_r_mb = lcl_r_mb[:, -n_future, :, :][:, None, :, :]

    start_lcl_q_mb = lcl_q_mb[:, n_past - 1, :, :]
    end_lcl_q_mb = lcl_q_mb[:, -n_future, :, :]

    # LERP Local Positions:
    n_trans = lcl_r_mb.shape[1] - (n_past + n_future)
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    offset = end_lcl_r_mb - start_lcl_r_mb

    const_trans = np.tile(start_lcl_r_mb, [1, n_trans + 2, 1, 1])
    inter_lcl_r_mb = const_trans + (interp_ws)[None, :, None, None] * offset

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    inter_lcl_q_mb = np.stack(
        [(quat_normalize(quat_slerp(quat_normalize(start_lcl_q_mb), quat_normalize(end_lcl_q_mb), w))) for w in
         interp_ws], axis=1)

    return inter_lcl_r_mb, inter_lcl_q_mb


def remove_quat_discontinuities(rotations): ###TODO
    """

    Removing quat discontinuities on the time dimension (removing flips)

    :param rotations: Array of quaternions of shape (T, J, 4)
    :return: The processed array without quaternion inversion.
    """
    rots_inv = -rotations

    for i in range(1, rotations.shape[0]):
        # Compare dot products
        replace_mask = np.sum(rotations[i - 1: i] * rotations[i: i + 1], axis=-1) < np.sum(
            rotations[i - 1: i] * rots_inv[i: i + 1], axis=-1)
        replace_mask = replace_mask[..., np.newaxis]
        rotations[i] = replace_mask * rots_inv[i] + (1.0 - replace_mask) * rotations[i]

    return rotations


# Orient the data according to the las past keframe
def rotate_at_frame(X, Q, parents): ###TODO
    """
    Re-orients the animation data according to the last frame of past context.

    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """
    # Get global quats and global poses (FK)
    global_q, global_x = quat_fk(Q, X, parents)

    # key_glob_Q = global_q[:, n_past - 1: n_past, 0:1, :]  # (B, 1, 1, 4)
    # forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
    #              * quat_mul_vec(key_glob_Q, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    key_glob_Q = global_q[:, 0:1, 0:1, :]  # (B, 1, 1, 4)
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
              * quat_mul_vec(key_glob_Q, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    new_glob_Q = quat_mul(quat_inv(yrot), global_q)
    new_glob_X = quat_mul_vec(quat_inv(yrot), global_x)

    # back to local quat-pos
    Q, X = quat_ik(new_glob_Q, new_glob_X, parents)

    return X, Q


# Orient the data according to the las past keframe
def rotate_at_frame2(pos_loc, rot_loc, parents, n_past=10): ###TODO
    """
    Re-orients the animation data according to the last frame of past context.
    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """
    # Get global quats and global poses (FK)
    pos_loc = np.expand_dims(pos_loc, 0)
    rot_loc = np.expand_dims(rot_loc, 0)

    rot_glo, pos_glo = quat_fk(rot_loc, pos_loc, parents)

    key_rot_glo = rot_glo[:, n_past - 1: n_past, 0:1, :]  # (B, 1, 1, 4)
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
              * quat_mul_vec(key_rot_glo, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    rot_glo_re = quat_mul(quat_inv(yrot), rot_glo)
    pos_glo_re = quat_mul_vec(quat_inv(yrot), pos_glo)

    # back to local quat-pos
    rot_loc_re, pos_loc_re = quat_ik(rot_glo_re, pos_glo_re, parents)

    pos_loc_re = pos_loc_re[0]
    rot_loc_re = rot_loc_re[0]

    pos_glo_re = pos_glo_re[0]
    rot_glo_re = rot_glo_re[0]

    return pos_loc_re, rot_loc_re, pos_glo_re, rot_glo_re


# def get_foot_offset(pos, rot):
#     # 3 LeftFoot 4 LeftToe
#     # 7 RightFoot 8 RightToe
#     root_rot = rot[:, 0:1, :]  # Shape (T, J, 4)
#
#     forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, :] \
#               * quat_mul_vec(root_rot, np.array([0, 1, 0])[np.newaxis, np.newaxis, :])  # Shape (T, J, 3)
#     forward = normalize(forward)  # Shape (T, J, 3)
#     yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
#
#     l_foot = pos[:, 4:5, :] - pos[:, 0:1, :]  # (-1, 3)
#     r_foot = pos[:, 8:9, :] - pos[:, 0:1, :]  # (-1, 3)
#     print('Hips 0', pos[:, 0:1, :])
#     print('LeftToe 0', pos[:, 4:5, :])
#     print('l_foot 0', l_foot)
#     l_foot = quat_mul_vec(yrot, l_foot)
#     r_foot = quat_mul_vec(yrot, r_foot)
#     # print('l_foot 1', l_foot)
#     foot_offset = np.concatenate([l_foot, r_foot], axis=2).reshape(-1)
#     return foot_offset

def get_foot_offset(pos, rot, parents): ###TODO some hardcode
    # 3 LeftFoot 4 LeftToe
    # 7 RightFoot 8 RightToe

    rot, pos = quat_fk(rot, pos, parents)

    root_rot = rot[:, 0:1, :]  # Shape (T, J, 4)

    forward = torch.Tensor([1., 0., 1.])[None, None, :] \
              * quat_mul_vec(root_rot, torch.Tensor([0., 1., 0.])[None, None, :])  # Shape (T, J, 3)
    forward = normalize(forward)  # Shape (T, J, 3)
    yrot = quat_normalize(quat_between(torch.Tensor([1., 0., 0.]), forward))

    l_foot = pos[:, 4:5, :] - pos[:, 0:1, :]  # (-1, 3)
    r_foot = pos[:, 8:9, :] - pos[:, 0:1, :]  # (-1, 3)

    l_foot = quat_mul_vec(yrot, l_foot)
    r_foot = quat_mul_vec(yrot, r_foot)

    foot_offset = np.concatenate([l_foot, r_foot], axis=2).reshape(-1)
    return foot_offset


def rot2vct(rotations): ###TODO
    # rot_glo shape (T, J, 4)
    root_rot = rotations[:, 0, :]
    forward = quat_mul_vec(root_rot, torch.Tensor([0., 1., 0.])[None, :])
    x_forward = torch.Tensor([1., 0., 1.])[None, :]
    forward = x_forward * forward
    forward = normalize(forward)
    forward = forward[:, [0, 2]]
    return forward


def rotate_at_frame3(pos_loc, rot_loc, parents, n_past=10): ###TODO
    """
    Re-orients the animation data according to the last frame of past context.
    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """
    # Get global quats and global poses (FK)
    pos_loc = np.expand_dims(pos_loc, 0)  # extend batch-size [1, 1, J, 3]
    rot_loc = np.expand_dims(rot_loc, 0)  # extend batch-size [1, 1, J]

    rot_glo, pos_glo = quat_fk(rot_loc, pos_loc, parents)

    key_rot_glo = rot_glo[:, n_past, 0:1, :]  # (B, 1, 1, 4)
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
              * quat_mul_vec(key_rot_glo, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    rot_glo_re = quat_mul(quat_inv(yrot), rot_glo)
    pos_glo_re = quat_mul_vec(quat_inv(yrot), pos_glo)

    # back to local quat-pos
    rot_loc_re, pos_loc_re = quat_ik(rot_glo_re, pos_glo_re, parents)

    pos_loc_re = pos_loc_re[0]
    rot_loc_re = rot_loc_re[0]

    pos_glo_re = pos_glo_re[0]
    rot_glo_re = rot_glo_re[0]

    return pos_loc_re, rot_loc_re, pos_glo_re, rot_glo_re


def rotate_at_frame_by_vec(pos_loc, rot_loc, parents, vector): ###TODO
    pos_loc = np.expand_dims(pos_loc, 0)
    rot_loc = np.expand_dims(rot_loc, 0)
    rot_glo, pos_glo = quat_fk(rot_loc, pos_loc, parents)
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
              * vector[np.newaxis, np.newaxis, np.newaxis, :]
    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    rot_glo_re = quat_mul(quat_inv(yrot), rot_glo)
    pos_glo_re = quat_mul_vec(quat_inv(yrot), pos_glo)
    rot_loc_re, pos_loc_re = quat_ik(rot_glo_re, pos_glo_re, parents)
    pos_loc_re = pos_loc_re[0]
    rot_loc_re = rot_loc_re[0]
    pos_glo_re = pos_glo_re[0]
    rot_glo_re = rot_glo_re[0]
    return pos_loc_re, rot_loc_re, pos_glo_re, rot_glo_re


def rotate_to_frame(pos_loc, rot_loc, pos_loc_src, rot_loc_src, parents): ###TODO
    """
    Re-orients the animation data according to the last frame of past context.
    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """
    # Get global quats and global poses (FK)
    pos_loc = np.expand_dims(pos_loc, 0)
    rot_loc = np.expand_dims(rot_loc, 0)
    rot_glo, pos_glo = quat_fk(rot_loc, pos_loc, parents)

    pos_loc_src = np.expand_dims(pos_loc_src, 0)
    rot_loc_src = np.expand_dims(rot_loc_src, 0)
    rot_glo_src, pos_glo_src = quat_fk(rot_loc_src, pos_loc_src, parents)

    # Get Forward Rot
    key_rot_glo = rot_glo_src[:, -1:, 0:1, :]
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
              * quat_mul_vec(key_rot_glo, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    forward = normalize(forward)
    # print('forward before', forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    rot_glo_re = quat_mul(yrot, rot_glo)
    pos_glo_re = quat_mul_vec(yrot, pos_glo)

    # back to local quat-pos
    rot_loc_re, pos_loc_re = quat_ik(rot_glo_re, pos_glo_re, parents)
    pos_loc_re = pos_loc_re[0]
    rot_loc_re = rot_loc_re[0]
    pos_glo_re = pos_glo_re[0]
    rot_glo_re = rot_glo_re[0]

    # key_rot_glo = rot_glo_re[-1:, 0:1, :]
    # forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
    #           * quat_mul_vec(key_rot_glo, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    # forward = normalize(forward)
    # print('forward after', forward)

    return pos_loc_re, rot_loc_re, pos_glo_re, rot_glo_re


def extract_feet_contacts(pos, lfoot_idx, rfoot_idx, velfactor=0.02): ###TODO
    """
    Extracts binary tensors of feet contacts

    :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
    :param lfoot_idx: indices list of left foot joints
    :param rfoot_idx: indices list of right foot joints
    :param velfactor: velocity threshold to consider a joint moving or not
    :return: binary tensors of left foot contacts and right foot contacts
    """
    lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
    contacts_l = (np.sum(lfoot_xyz, axis=-1) < velfactor)

    rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
    contacts_r = (np.sum(rfoot_xyz, axis=-1) < velfactor)

    # Duplicate the last frame for shape consistency
    contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
    contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

    return contacts_l, contacts_r


# def rotation2bvh(file, rotations, position):
#     from phase_motion_matching.skeleton_ubi import offsets_ubi as offsets
#     from phase_motion_matching.skeleton_ubi import parents_ubi as parents
#     # from sptmm.skeleton_kma import offsets_kma as offsets
#     # from sptmm.skeleton_kma import parents_kma as parents
#     from PyIK.Quaternions import Quaternions
#     from PyIK.Animation import Animation
#     from PyIK.BVH import save
#     orients = Quaternions.id(1)
#     orients_final = np.array([[1, 0, 0, 0]]).repeat(len(offsets), axis=0)
#     orients.qs = np.append(orients.qs, orients_final, axis=0)
#     rotations_Quat = Quaternions(rotations)
#     # position = position[:, np.newaxis, :]
#     anim = Animation(rotations_Quat, position, orients, offsets, parents)
#     save(file, anim, frametime=1.0 / 30.0)


def standardization(M):
    M_std = torch.std(M, dim=0) + 1E-7
    M_mean = torch.mean(M, dim=0)
    _M = (M - M_mean) / M_std
    return _M, M_std, M_mean


def normalization(M):
    M_sum = torch.sum(M, dim=1)
    M_avg = torch.mean(M_sum)
    _M = M / M_avg
    return _M, M_avg
