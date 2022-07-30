import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import torch
import smplx
import cv2
from Utils.utils_mmphsd import render_model
import joblib
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot  
from Utils.core_utils import LabelMotionGetter_mmhpsd as LabelMotionGetter

import rbdl
import copy
from Utils.initialize import Initializer
import pybullet as p
from Utils.core_utils import KinematicUtil, angle_util, Core_utils,RefCorrect

data_dir = '/home/datassd/yuxuan'
action = 'subject01_group1_time1' 
start_frame_idx = 200
end_frame_idx = 400 #210#400
img_size = 256
scale = img_size / 1280.
cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
save_dir = '/home/yuxuan/demo'




def visualize_sythesis_events():
    num_frame = len(os.listdir('%s/data_event_out/pose_events/%s' % (data_dir, action))) - 1
    beta_list, theta_list, tran_list, joints2d_list, joints3d_list = [], [], [], [], []
    for idx in range(num_frame):
        beta, theta, tran, joints3d, joints2d = joblib.load(
                    '%s/data_event_out/pose_events/%s/pose%04i.pkl' % (data_dir, action, idx))
        beta_list.append(beta)
        theta_list.append(theta)
        tran_list.append(tran)
        joints2d_list.append(joints2d)
        joints3d_list.append(joints3d)
    
    # tmp = np.array(joints3d_list)
    # feet = tmp[:,10:12]
    # feet = np.round(feet, 2)
    # feet = np.reshape(feet, (-1, 2*3))
    # import scipy
    # print(scipy.stats.mode(feet, axis=0)[0].shape)
    # print(scipy.stats.mode(feet, axis=0)[0])
    # return

    qs = np.load('results/PhyCap_q_%s.npy' %action)

    #recover
    
    thetas = np.concatenate(theta_list, axis=0)
    trans = np.concatenate(tran_list, axis=0)
    motion_params = np.concatenate((trans, thetas), axis=1)
    AU = angle_util()
    CU = Core_utils()
    kui = KinematicUtil() 
    id_simulator = p.connect(p.DIRECT) 
    p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1) 
     
    model = rbdl.loadModel('asset/physcap.urdf'.encode()) 
    model.gravity = np.array([0., -9.81, 0.]) ###set in rbdl model, no gravity is set in pybullet
    id_robot = p.loadURDF('asset/physcap.urdf', [0, 0, 0.5], globalScaling=1, useFixedBase=False) #optimizing
    # if ((jointType==URDFFloatingJoint)||(jointType==URDFPlanarJoint)) warning: root joint type is float 
    # {
    #     printf("Warning: joint unsupported, creating a fixed joint instead.");
    # }
    id_robot_vnect = p.loadURDF('asset/physcap.urdf', [0, 0, 0.5], globalScaling=1, useFixedBase=False) #init kinematic
    skeleton_specific_base_offset =  np.array([-2.35437, -237.806, 26.4052]) #see physcap.skeleton
    _, _, jointIds, jointNames = kui.get_jointIds_Names(id_robot)
    LMG = LabelMotionGetter('asset/physcap.skeleton', motion_params, jointNames, skeleton_specific_base_offset)
    la_po_dic = LMG.get_dictionary()
    _partial_euler = LMG.cleaned_pose.T

    length = LMG.motion_params.shape[0]
    smpl_theta = np.reshape(LMG.motion_params[:, 3:], (-1, 3))
    result_trans = LMG.motion_params[:, :3][:, None]
    r = Rot.from_rotvec(smpl_theta)
    euler = r.as_euler('xyz')
    euler = np.reshape(euler, (length, -1))

    ini = Initializer()
    rbdl2bullet = ini.get_rbdl2bullet()
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]

    jointNames_reordered = [jointNames[i] for i in rbdl2bullet] #would be the same order as q

    # for i, n in zip(jointIds, jointNames):
    #     print(i, n, end=' ||| ')

    # print('='*50)
    # for i, n in zip(jointIds_reordered, jointNames_reordered):
    #     print(i, n, end=' ||| ')

    
    # print(len(LMG.dof_names)) #33
    # print(qs.shape) #1345,43
    recover_partial_euler = np.array([qs[:, 6:][:, jointNames_reordered.index(name.encode())] for name in LMG.dof_names if name.encode() in jointNames_reordered ])
    # print(_partial_euler.shape, recover_partial_euler.shape) #33, 27

    # for n, i, j  in zip(LMG.dof_names[6:], _partial_euler[206][6:], recover_partial_euler.T[206]):
    #     print(n, i, j)

    # print(jointNames)
    # print(LMG.dof_names)
    # print(recover_partial_euler.shape)
    # print(euler.shape)
    # print(len(LMG.clean_pose_idx))

    # euler[:, LMG.clean_pose_idx[6:]] = _partial_euler[:, 6:]#to verify NOTE LMG.clean_pose_idx[6:] need to substrast number of root trans, since it is originally applied to pose with concate trans and theta.

    # _rotv = np.array(theta_list)[:,0]
    # _r = Rot.from_rotvec(np.reshape(_rotv, (-1, 3)))
    # _euler = np.reshape(_r.as_euler('xyz'), (length, -1))
    # _pose = np.concatenate((LMG.motion_params[:, :3], _euler), axis=1)
    # _partial_euler_v = _pose[:, LMG.clean_pose_idx] #verified correctness
    # print(_partial_euler.shape, _partial_euler_v.shape)
    # for n, i, j  in zip(LMG.dof_names, _partial_euler[206], _partial_euler_v[206]):
    #     print(n, i, j)

    # euler = _euler#to verify

    idx = [x-3 for x in LMG.clean_pose_idx[6:]] #IMPORTANT !
    euler[:, idx] = recover_partial_euler.T
    euler[:, :3] = qs[:, 3: 6]
    euler = np.reshape(euler, (-1, 3))
    r = Rot.from_euler('xyz', euler)
    result_theta = np.reshape(r.as_rotvec(), (length, -1))[:, None]

    device = torch.device("cpu")
    smplmodel = smplx.create(os.path.join(data_dir, "smpl_model/models"),
                             model_type="smpl", gender="male", ext="pkl",
                             batch_size=1).to(device)

    frames = []

    tmp = []
    # for frame_idx in tqdm(range(num_frame)):
    for frame_idx in tqdm(range(start_frame_idx, end_frame_idx)):
        img_filename = '{}/data_event_out/full_pic_256/{}/fullpic{:04d}.jpg'.format(data_dir, action, frame_idx)
        img = cv2.resize(cv2.imread(img_filename)[:, :, ::-1], (img_size, img_size))

        # joints3d=target_joints3d, joints2d=target_joints2d, trans=target_trans, body_pose=theta, beta=beta
        beta = beta_list[frame_idx]
        theta = theta_list[frame_idx]
        trans = tran_list[frame_idx]
        with torch.no_grad():
            outputp = smplmodel(betas=torch.from_numpy(beta).float().to(device),
                                global_orient=torch.from_numpy(theta[:, :3]).float().to(device),
                                body_pose=torch.from_numpy(theta[:, 3:]).float().to(device),
                                transl=torch.from_numpy(trans).float().to(device),
                                return_verts=True)
        verts = outputp.vertices.detach().cpu().numpy()[0]

        # tmp.append(np.sort(verts[:, 1])[-10:])
        # print(theta[:, :3])
        
        

        render_img = (render_model(
            verts, smplmodel.faces, img_size, img_size, np.asarray(cam_intr[0:4]),
            np.zeros([3]), np.zeros([3]), near=0.1, far=20, img=None)[:, :, 0:3] * 255).astype(np.uint8)

        theta = result_theta[frame_idx]
        trans = result_trans[frame_idx]
        with torch.no_grad():
            outputp = smplmodel(betas=torch.from_numpy(beta).float().to(device),
                                global_orient=torch.from_numpy(theta[:, :3]).float().to(device),
                                body_pose=torch.from_numpy(theta[:, 3:]).float().to(device),
                                transl=torch.from_numpy(trans).float().to(device),
                                return_verts=True)
        verts = outputp.vertices.detach().cpu().numpy()[0]

        result_render_img = (render_model(
            verts, smplmodel.faces, img_size, img_size, np.asarray(cam_intr[0:4]),
            np.zeros([3]), np.zeros([3]), near=0.1, far=20, img=None)[:, :, 0:3] * 255).astype(np.uint8)

        frame = np.concatenate([img, render_img, result_render_img], axis=1)
        frames.append(frame)
    
    # tmp = np.concatenate(tmp, axis=0)
    # print(tmp)

    demo_name = '{}/physcap_{}_{}_{}.gif'.format(save_dir, action, start_frame_idx, end_frame_idx)
    imageio.mimsave(demo_name, frames, 'GIF', duration=1. / 15.)
    print('save as {}'.format(demo_name))


if __name__ == '__main__':
    visualize_sythesis_events()





