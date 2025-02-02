import joblib
import numpy as np
import os
# from stage3_mmphsd import sim_loop
from stage3_mmphsd_ordertest import sim_loop
from models.networks import ContactEstimationNetwork
import argparse
from stage2 import inferenceCon
import torch

###to run: python exp_mmphsd.py --contact_estimation 1 --save_path './results/' --floor_known 0 --humanoid_path asset/physcap.urdf --skeleton_filename asset/physcap.skeleton --contact_path results/contacts.npy --stationary_path results/stationary.npy
if __name__ == '__main__':

    ### config for fitting and contact calculations ###
    parser = argparse.ArgumentParser(description='arguments for predictions')
    parser.add_argument('--contact_estimation', type=int, default=1)
    parser.add_argument('--image_size',type=int, default=256)
    parser.add_argument('--floor_known', type=int, default=0)
    parser.add_argument('--model_path', default="models/ConStaNet_sample.pkl") 
    parser.add_argument('--floor_frame',  default="data/floor_frame.npy") 
    # parser.add_argument('--vnect_2d_path', default="data/sample_vnect_2ds.npy") 
    parser.add_argument('--humanoid_path', default='asset/physcap.urdf') 
    parser.add_argument('--skeleton_filename', default="asset/physcap.skeleton" )
    # parser.add_argument('--motion_filename', default="data/sample.motion")
    parser.add_argument('--floor_path', default="asset/plane.urdf") 
    parser.add_argument('--contact_path', default="results/contacts.npy") 
    parser.add_argument('--stationary_path', default="results/stationary.npy")
    parser.add_argument('--save_path', default='./results/')
    args = parser.parse_args()
    data_dir = '/home/datassd/yuxuan/data_event_out'
    action = 'subject01_group1_time1' 
    kinematic_2d_path = "data/kinematic_2ds.npy"
    num_frame = len(os.listdir('%s/pose_events/%s' % (data_dir, action))) - 1
    theta_list, tran_list, joints2d_list, joints3d_list = [], [], [], []
    for idx in range(num_frame):
        beta, theta, tran, joints3d, joints2d = joblib.load(
                    '%s/pose_events/%s/pose%04i.pkl' % (data_dir, action, idx))
        theta_list.append(theta[0])
        tran_list.append(tran[0])
        joints2d_list.append(joints2d)
        joints3d_list.append(joints3d)
    kinematic_2d = np.stack(joints2d_list, axis=0)[:,:,:-1] #(1345, 24, 2)
    np.save(kinematic_2d_path, kinematic_2d)

    ### Contact and Stationary Estimation ### don't need floor
    if args.contact_estimation: #if Ture, run stage two, vnect_2d used. (VNect uses a different joints index, see https://github.com/XinArkh/VNect)
        target_joints = ["head", "neck", "left_hip",  "left_knee", "left_ankle", "left_toe",  "right_hip", "right_knee", "right_ankle", "right_toe",  "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
        
        ###about to change the dict when using different 2d prediction skeleton###
        # vnect_dic = {"base": 14, "head": 0, "neck": 1, "left_hip": 11, "left_knee": 12, "left_ankle": 13, "left_toe": 16,"right_hip": 8,  "right_knee": 9, "right_ankle": 10, "right_toe": 15, "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7,  "right_shoulder": 2, "right_elbow": 3, "right_wrist": 4 }

        kinematic_dic = {"base": 0, "head": 15, "neck": 12, "left_hip": 1, "left_knee": 4, "left_ankle": 7, "left_toe": 10,"right_hip": 2,  "right_knee": 5, "right_ankle": 8, "right_toe": 11, "left_shoulder": 16, "left_elbow": 18, "left_wrist": 20,  "right_shoulder": 17, "right_elbow": 19, "right_wrist": 21 }
        
        window_size=10  
        ConNet = ContactEstimationNetwork(in_channels=32, num_features=512, out_channels=5, num_blocks=4).cuda() 
        ConNet.load_state_dict(torch.load(args.model_path))
        ConNet.eval()
        print("Stage II running ... ")
        # inferenceCon(target_joints,vnect_dic,ConNet,args.image_size,window_size,args.vnect_2d_path,args.save_path)
        inferenceCon(target_joints,kinematic_dic,ConNet,args.image_size,window_size,kinematic_2d_path,args.save_path)
        print("Done. Predictions were saved at "+args.save_path)

    ### Physics-based Optimization ###
    thetas = np.stack(theta_list, axis=0)
    trans = np.stack(tran_list, axis=0)
    motion_params = np.concatenate((trans, thetas), axis=1)

    path_dict={ 
            "floor_frame": args.floor_frame, 
            "humanoid_path": args.humanoid_path,
            "motion_params": motion_params,
            "skeleton_filename": args.skeleton_filename,
            "floor_path": args.floor_path,
            "contact_path": args.contact_path,
            "stationary_path": args.stationary_path,
            "save_path": args.save_path,
            "action": action
            } 
    print("Stage III running ... ") 
    sim_loop(path_dict,floor_known=args.floor_known)
    print("Done. Predictions were saved at "+args.save_path)
