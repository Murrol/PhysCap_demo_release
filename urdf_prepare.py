import os
import sys
import json
import joblib
import trimesh
import subprocess
import numpy as np
# from smplx import SMPL, SMPLH, SMPLX
import smplx
from matplotlib import cm as mpl_cm, colors as mpl_colors
from scipy.spatial import cKDTree

SMPL_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand'
    ]

SKELETON = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 4],
    [2, 5],
    [3, 6],
    [4, 7],
    [5, 8],
    [6, 9],
    [7, 10],
    [8, 11],
    [9, 12],
    [9, 13],
    [9, 14],
    [12, 15],
    [13, 16],
    [14, 17],
    [16, 18],
    [17, 19],
    [18, 20],
    [19, 21],
    [20, 22],
    [21, 23]
]
def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)
    file_path = os.path.join(outdir, url.split('/')[-1])
    return file_path


def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros(n_vertices)

    for part_idx, (k, v) in enumerate(part_segm.items()):

        vertex_labels[v] = part_idx

    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3] = alpha
    vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

    return vertex_colors


def main(body_model='smpl', body_model_path='/home/datassd/yuxuan/smpl_model/models'):
    if body_model == 'smpl':
        part_segm_filepath = os.path.join(os.path.split(body_model_path)[0], 'assets-SMPL_body_segmentation/smpl/smpl_vert_segmentation.json')
    elif body_model == 'smplx':
        part_segm_filepath = os.path.join(os.path.split(body_model_path)[0], 'assets-SMPL_body_segmentation/smplx/smplx_vert_segmentation.json')
    elif body_model == 'smplh':
        part_segm_filepath = os.path.join(os.path.split(body_model_path)[0], 'assets-SMPL_body_segmentation/smpl/smpl_vert_segmentation.json')
    else:
        raise ValueError(f'{body_model} is not defined, \"smpl\", \"smplh\" or \"smplx\" are valid body models')

    body_model = smplx.create(model_path=body_model_path, model_type=body_model)
    part_segm = json.load(open(part_segm_filepath))
    
    part_segm['leftHand'] += part_segm.pop('leftHandIndex1')
    part_segm['rightHand'] += part_segm.pop('rightHandIndex1')#22 same as drecon
    print('num of parts:', len(part_segm.items()))




    vertices = body_model().vertices[0].detach().numpy()
    joints = body_model().joints[0].detach().numpy()

    print(joints)
    faces = body_model.faces

    parts_mesh = list()
    save_dict = dict()
    for part_idx, (k, v) in enumerate(part_segm.items()):
        mesh = trimesh.Trimesh(vertices[v], process=False)
        _mesh = mesh.convex_hull
        # parts_mesh.append(_mesh)
        save_dict[k] = _mesh
    
    joblib.dump(save_dict, './body_parts.pkl')
    joblib.dump({'skeleton': np.array(SKELETON), 'joints_name': SMPL_JOINT_NAMES, 'joints_position': joints}, './joints_info.pkl')
    parts_mesh = joblib.load('./body_parts.pkl').values()
    for idx, m in enumerate(parts_mesh):
        m.export('../demo/body_parts_%d.stl' %idx)
    scene = trimesh.Scene(parts_mesh)
    scene.show()
    scene.export('../demo/body_parts.stl')
    # vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[0])
    # mesh = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertex_colors)
    # mesh.show(background=(0,0,0,0))


if __name__ == '__main__':
    # main(sys.argv[1], sys.argv[2])
    main()