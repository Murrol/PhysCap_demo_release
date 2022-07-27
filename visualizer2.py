from tkinter import N
import pybullet as p
import rbdl
import numpy as np
from Utils.core_utils import KinematicUtil#,angle_util,LabelMotionGetter,Core_utils
from Utils.initialize import Initializer
from scipy.spatial.transform import Rotation as Rot
import time
import argparse

parser = argparse.ArgumentParser(description='arguments for predictions')
parser.add_argument('--q_path',  default="./results/PhyCap_q_subject01_group1_time1.npy") 
args = parser.parse_args()
id_simulator = p.connect(p.GUI)
# p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=0)
p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
p.configureDebugVisualizer(flag=p.COV_ENABLE_SHADOWS, enable=0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) 

debugcamera_param = {'cameraDistance': 1,
        'cameraYaw': 170,
        'cameraPitch': -10,
        'cameraTargetPosition': [0, 0, 4]}
p.resetDebugVisualizerCamera(**debugcamera_param)

def isKeyTriggered(keys, key):
  o = ord(key)
  if o in keys:
    return keys[ord(key)] & p.KEY_WAS_TRIGGERED
  return False

def visualizer(id_robot, id_robot_ref, q, q_ref, jointIds_reordered, jointIds):
    kui.motion_update_specification(id_robot, jointIds_reordered, q[6:]) 
    # print(q_ref-q)
    kui.motion_update_specification(id_robot_ref, jointIds, q_ref[6:]) 
    r = Rot.from_euler('xyz', q[3:6])  # Rot.from_matrix()
    angle = r.as_euler('xyz') 
    # p.resetBasePositionAndOrientation(id_robot, [q[0], q[1], q[2]], p.getQuaternionFromEuler([angle[0], angle[1], angle[2]]))
    p.resetBasePositionAndOrientation(id_robot, q[: 3], p.getQuaternionFromEuler(q[3: 6]))
    # p.stepSimulation()

    ### angle_ref = r_ref.as_euler('xyz') ; [angle_ref[2], angle_ref[1], angle_ref[0]] == angle_ref = r_ref.as_euler('ZYX')
    p.resetBasePositionAndOrientation(id_robot_ref, q_ref[: 3], p.getQuaternionFromEuler(q_ref[3: 6]))
    p.stepSimulation()
    return 0

def data_loader(q_path):
    return np.load(q_path) 
    
 
if __name__ == '__main__': 
    fps = 25
    ini = Initializer()
    rbdl2bullet=ini.get_rbdl2bullet() 
    kui = KinematicUtil() 
    qs = data_loader(args.q_path)
    ref_qs = data_loader(args.q_path.replace('PhyCap_q', 'Ref_q'))
    print(ref_qs.shape)
    print(qs.shape)
    humanoid_path='./asset/physcap.urdf'
    
    model = rbdl.loadModel(humanoid_path.encode())
    id_robot = p.loadURDF(humanoid_path, [0, 0, 0.5], globalScaling=1, useFixedBase=False)
    id_robot_ref = p.loadURDF(humanoid_path, [0, 0, 0.5], globalScaling=1, useFixedBase=False)
    alpha = 0.7
    for j in range(-1, p.getNumJoints(id_robot_ref)):
        p.changeVisualShape(id_robot_ref, j, rgbaColor = [1, 1, 1, alpha])
    ini.remove_collisions(id_robot, id_robot_ref)
    # id_robot_ref = None
    # ref_qs = None
     
    _, _, jointIds, _ = kui.get_jointIds_Names(id_robot)
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]

    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "../demo0726.mp4")
    for q, ref_q in zip(qs, ref_qs): 
        visualizer(id_robot, id_robot_ref, q, ref_q, jointIds_reordered, jointIds)  
        time.sleep(0.1)
    p.disconnect()

    # animating = False
    # while(p.isConnected()):
    #     keys = p.getKeyboardEvents()
    #     if isKeyTriggered(keys, ' '): # Press spacebar to start animating
    #         animating = not animating
        
    #     if animating:
    #         for q, ref_q in zip(qs, ref_qs): 
    #             visualizer(id_robot, id_robot_ref, q, ref_q, jointIds_reordered, jointIds)  
    #             time.sleep(0.002)
