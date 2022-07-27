import sys
sys.path.append('../')
from . import calibration as util
import pickle
import numpy as np
import cv2
import matplotlib.pylab as plt
import trimesh


class MultiCamParams:
    def __init__(self, root_dir, cam_list):
        self.root_dir = root_dir
        self.cam_list = cam_list
        self.action = None
        self.intr = None
        self.kinect_extr = None
        self.extr = None
        self.trans_dict = None
        self.subject_calib_dict = {
            'subject01': '1024',
            'subject02': '1024',
            'subject03': '1024',
            'subject04': '1028',
            'subject05': '1028',
            'subject06': '1028',
            'subject07': '1028',
            'subject08': '1028',
            'subject09': '1028',
            'subject10': '1028',
            'subject11': '1101',
            'subject12': '1101',
            'subject13': '1101',
            'subject14': '1101',
            'subject15': '1101'
        }

    def set_action(self, action):
        self.action = action
        date = self.subject_calib_dict[self.action.split('_')[0]]
        calib_dir = '%s/calib_%s' % (self.root_dir, date)
        self.intr = pickle.load(open('%s/intrinsic_param.pkl' % calib_dir, 'rb'))
        # print(self.intr.keys())
        self.kinect_extr = pickle.load(open('%s/kinect_extrinsic_param.pkl' % calib_dir, 'rb'))
        # print(self.kinect_extr.keys())
        self.extr = pickle.load(open('%s/extrinsic_param_%s.pkl' % (calib_dir, date), 'rb'))
        # print(self.extr.keys())
        print('[%s] %s' % (action, calib_dir))
        self.trans_dict = self.extrinsic_transform()

    def extrinsic_transform(self):
        trans_dict = {}
        # depth to color within each kinect
        for cam in self.cam_list:
            if 'kinect' in cam:
                trans_dict['%s_cd' % cam] = \
                    util.Transform(r=self.kinect_extr['%s_d2c' % cam][0], t=self.kinect_extr['%s_d2c' % cam][1] / 1000)

        # extrinsic from other cam depth to azure_kinect_0 depth
        key = 'azure_kinect_0-azure_kinect_0'
        trans_dict[key] = util.Transform(r=np.eye(3), t=np.zeros([3]))
        key = 'azure_kinect_0-azure_kinect_1'
        trans_dict[key] = util.Transform(r=self.extr[key][0], t=self.extr[key][1] / 1000)
        key = 'azure_kinect_0-azure_kinect_2'
        trans_dict[key] = util.Transform(r=self.extr[key][0], t=self.extr[key][1] / 1000)

        key = 'azure_kinect_0-kinect_v2_1'
        T_tmp = util.Transform(r=self.extr['azure_kinect_1-kinect_v2_1'][0],
                               t=self.extr['azure_kinect_1-kinect_v2_1'][1] / 1000)
        trans_dict[key] = trans_dict['azure_kinect_0-azure_kinect_1'] * T_tmp
        key = 'azure_kinect_0-kinect_v2_2'
        T_tmp = util.Transform(r=self.extr['azure_kinect_2-kinect_v2_2'][0],
                               t=self.extr['azure_kinect_2-kinect_v2_2'][1] / 1000)
        trans_dict[key] = trans_dict['azure_kinect_0-azure_kinect_2'] * T_tmp

        key = 'event_camera-azure_kinect_0'
        trans_dict[key] = util.Transform(r=self.extr[key][0], t=self.extr[key][1] / 1000)
        key = 'polar-azure_kinect_0'
        trans_dict[key] = util.Transform(r=self.extr[key][0], t=self.extr[key][1] / 1000)
        return trans_dict

    def get_extrinsic_transform(self, source, target, depth_cam=True):
        if '%s-%s' % (target, source) in self.trans_dict.keys():
            T_depth = self.trans_dict['%s-%s' % (target, source)]
            if depth_cam or 'kinect' not in target:
                return T_depth
            else:
                T_color = self.trans_dict['%s_cd' % target] * T_depth
                return T_color
        elif '%s-%s' % (source, target) in self.trans_dict.keys():
            T_depth = self.trans_dict['%s-%s' % (source, target)].inv()
            if depth_cam or 'kinect' not in target:
                return T_depth
            else:
                T_color = self.trans_dict['%s_cd' % target] * T_depth
                return T_color
        else:
            raise ValueError('[error] %s-%s not exist.' % (target, source))

    def get_intrinsic_param(self, cam):
        if cam in self.intr.keys():
            return self.intr[cam]
        else:
            raise ValueError('[error] %s not exist.' % cam)


"""render"""


from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}


def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.01,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)


def simple_renderer(rn, verts, faces, yrot=np.radians(120)):

    # Rendered model color
    color = colors['pink']

    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))

    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge(
        (b_channel, g_channel, r_channel, alpha.astype(imtmp.dtype)))
    return im_RGBA


def render_model(verts, faces, w, h, cam_param, cam_t, cam_rt, near=0.5, far=25, img=None):
    f = cam_param[0:2]
    c = cam_param[2:4]
    rn = _create_renderer(w=w, h=h, near=near, far=far, rt=cam_rt, t=cam_t, f=f, c=c)
    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    imtmp = simple_renderer(rn, verts, faces)

    # If white bg, make transparent.
    if img is None:
        imtmp = get_alpha(imtmp)

    return imtmp


def render_depth_v(verts, faces, require_visi = False,
                   t = [0.,0.,0.], img_size=[448, 448], f=[400.0,400.0], c=[224.,224.]):
    from opendr.renderer import DepthRenderer
    rn = DepthRenderer()
    rn.camera = ProjectPoints(rt = np.zeros(3),
                              t = t,
                              f = f,
                              c = c,
                              k = np.zeros(5))
    rn.frustum = {'near': .01, 'far': 10000.,
                  'width': img_size[1], 'height': img_size[0]}
    rn.v = verts
    rn.f = faces
    rn.bgcolor = np.zeros(3)
    if require_visi is True:
        return rn.r, rn.visibility_image
    else:
        return rn.r


# others
def projection(xyz, intr_param, simple_mode=False):
    # xyz: [N, 3]
    # intr_param: (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
    assert xyz.shape[1] == 3
    fx, fy, cx, cy = intr_param[0:4]

    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_p = xyz[:, 0] / xyz[:, 2]
        y_p = xyz[:, 1] / xyz[:, 2]
        r2 = x_p ** 2 + y_p ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        b = b + (b == 0)
        d = a / b

        x_pp = x_p * d + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p ** 2)
        y_pp = y_p * d + p1 * (r2 + 2 * y_p ** 2) + 2 * p2 * x_p * y_p

        u = fx * x_pp + cx
        v = fy * y_pp + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)
    else:
        u = xyz[:, 0] / xyz[:, 2] * fx + cx
        v = xyz[:, 1] / xyz[:, 2] * fy + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)
