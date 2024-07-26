import json
from pathlib import Path
import argparse
import sys
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import open3d as o3d
import time
from utils.align import get_extrinsic

from utils.obj import load_obj
from utils.transform import intrinsic2proj

torch.set_grad_enabled(False)
def get_config(opt):
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    return config


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)     # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(curr_pose_pred, curr_pose_gt):
    euc_t = np.linalg.norm(curr_pose_pred['t'] - curr_pose_gt['t'])
    error_t = angle_error_vec(curr_pose_pred['t'], curr_pose_gt['t'])
    error_t = np.minimum(error_t, 180 - error_t)    # ambiguity of E estimation
    error_R = angle_error_mat(curr_pose_pred['R'], curr_pose_gt['R'])
    
    print('t_gt  ', curr_pose_gt['t'])
    print('t_pred', curr_pose_pred['t'])
    print('R_gt  ----------')
    print(curr_pose_gt['R'])
    print('R_pred----------')
    print(curr_pose_pred['R'])
    print('euc_t = ', euc_t)     # error in norm
    print('err_t = ', error_t)   # error in angle
    print('err_R = ', error_R)

    return error_t, error_R


def get_text(kpts0, kpts1, mkpts0, stem0, stem1, matching):
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]
    
    # Display extra parameter info.
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {}:{}'.format(stem0, stem1),
    ]

    return text, small_text


def get_filter_idx(mkpts0, kps_filter):
    tmp_dict = {}
    for i in range(mkpts0.shape[0]):
        tmp_dict[tuple(list(mkpts0[i]))] = i
        
    ind_ls = []
    for i in range(kps_filter.shape[0]):
        ind_ls.append(tmp_dict[tuple(list(kps_filter[i]))])
    
    return np.array(ind_ls)


def get_seg_mask(mask_path):
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  
    img[img>0] = 1
    seg_mask = img.astype(bool)

    return seg_mask


def get_depth_ycb(depth_path, resize, scale): 
    depth_img = cv2.imread(str(depth_path), -1)
    depth_scale=0.1                                         # for YCB-V data
    depth_img = np.float32(depth_img * depth_scale / 1000)  # depth value in meters!
    if scale != (1.0, 1.0):                                 
        depth_img = cv2.resize(depth_img, resize)           # resize = (w, h) 

    return depth_img


def backproj(uv_kps, depth, camK, scale=1):
    """ 
    Params: 
        uv_kps: nx2 array, each row is (x,y);
        depth: hxw array, depth image; if depth is not scaled in meters, please set the param `scale`!
        camK: 3x3 array, the camera intrinsics;
    Output:
        pts: the point cloud
        z_mask: point mask, in case there is no valid depth value.
    """ 
    uv_kps = uv_kps.astype(int)
    intrinsics_inv = np.linalg.inv(camK) 
    ones = np.ones([uv_kps.shape[0], 1])
    uv_grid = np.concatenate((uv_kps, ones), axis=1)    # [num, 3]
    xyz = uv_grid @ intrinsics_inv.T                    # [num, 3]

    z = depth[uv_kps[:,1], uv_kps[:,0]].astype(np.float32)
    z_mask = (z > 0)
    pts = xyz * z[:, np.newaxis]  # / xyz[:, -1:]
    
    return pts * scale, z_mask


def estimate_pose_ransac(pcd0, pcd1):
    """ matched pcd0/pcd1: nx3 array. """

    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd0))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd1))
    corres_arr = np.stack([np.arange(pcd0.shape[0]), np.arange(pcd0.shape[0])], axis=1)
    corres = o3d.utility.Vector2iVector(corres_arr)
    
    max_corres_distance = 0.02
    # estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    # criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=1000, confidence=0.999)
    
    start = time.time()
    reg = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source, target, corres, max_corres_distance,
            # estimation_method, ransac_n=3, criteria=criteria
    )
    print('\ntime elapses: ', time.time() - start)
    print('fitness: ', reg.fitness)                  # The overlapping area (# of inlier correspondences / # of points in source). Higher is better.
    print('inlier_rmse: ', reg.inlier_rmse, '\n')    # RMSE of all inlier correspondences. Lower is better.
    # print(reg.transformation)

    ret = (reg.transformation[:3,:3], reg.transformation[:3,3], reg.fitness)

    return ret
    

class PoseEstimator():
    def __init__(self, opt, camK0, camK1) -> None:
        self.camK0 = camK0
        self.camK1 = camK1
        config = get_config(opt)
        self.matching = Matching(config).eval().cuda()
    
    def umeyama(self, img0, img1, mask0, depth0, depth1):
        # Perform the matching.
        pred = self.matching({'image0': img0, 'image1': img1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # build kps_mask 
        kps_mask = np.zeros(img0.shape[-2:], dtype=bool)
        index = mkpts0.astype(int)
        kps_mask[index[:,1], index[:,0]] = 1

        # Look through seg_mask list to estimate pose
        comb_mask = kps_mask & mask0
        inds = np.nonzero(comb_mask)
        mkpts0_filter = np.stack([inds[1], inds[0]], axis=1).astype(np.float32)
        filter_ind = get_filter_idx(mkpts0, mkpts0_filter)
        mkpts1_filter = mkpts1[filter_ind]
        mconf_filter = mconf[filter_ind]

        # backproject to get point cloud
        mkpts0_pcd, z_mask0 = backproj(mkpts0_filter, depth0, self.camK0)
        mkpts1_pcd, z_mask1 = backproj(mkpts1_filter, depth1, self.camK1)
        z_mask = z_mask0 & z_mask1

        # Estimate camera delta pose.
        ret = estimate_pose_ransac(mkpts0_pcd[z_mask], mkpts1_pcd[z_mask])
        if ret:
            R, t, inliers = ret
        else:
            R, t = np.eye(3), np.zeros(3)

        current_pose = {'R': R, 't': t}

        results_extra = {
            'mconf':mconf_filter[z_mask], 
            'mkpts0':mkpts0_filter[z_mask], 
            'mkpts1':mkpts1_filter[z_mask], 
            'kpts0':kpts0, 
            'kpts1':kpts1
        }

        return current_pose, results_extra

    def epipolar(self, img0, img1, mask0, depth0, depth1):
        # Perform the matching.
        pred = self.matching({'image0': img0, 'image1': img1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # build kps_mask 
        kps_mask = np.zeros(img0.shape[-2:], dtype=bool)
        index = mkpts0.astype(int)
        kps_mask[index[:,1], index[:,0]] = 1

        # Look through seg_mask list to estimate pose
        comb_mask = kps_mask & mask0
        inds = np.nonzero(comb_mask)
        mkpts0_filter = np.stack([inds[1], inds[0]], axis=1).astype(np.float32)
        filter_ind = get_filter_idx(mkpts0, mkpts0_filter)
        mkpts1_filter = mkpts1[filter_ind]
        mconf_filter = mconf[filter_ind]

        # Estimate camera delta pose.
        thresh = 1.    # In pixels relative to resized image size.
        ret = estimate_pose(mkpts0_filter, mkpts1_filter, self.camK0, self.camK1, thresh)
        if ret:
            R, t, inliers = ret
            proj0 = np.zeros([3, 4], dtype=np.float32)
            proj0[:3, :3] = np.eye(3)
            proj1 = np.zeros([3, 4], dtype=np.float32)
            proj1[:3, :3] = R
            proj1[:3, -1] = t
            # proj1 = self.camK1 @ proj1
            pts0 = (mkpts0_filter[inliers] - self.camK0[[0, 1], [2, 2]][None]) / self.camK0[[0, 1], [0, 1]][None]
            pts1 = (mkpts1_filter[inliers] - self.camK1[[0, 1], [2, 2]][None]) / self.camK1[[0, 1], [0, 1]][None]
            pts3d = cv2.triangulatePoints(proj0, proj1, pts0.T, pts1.T).T
            pts3d = pts3d[:, :3] / pts3d[:, 3:]
            
            pts3d_depth0, z_mask0 = backproj(mkpts0_filter[inliers], depth0, self.camK0)
            pts3d_depth1, z_mask1 = backproj(mkpts1_filter[inliers], depth1, self.camK1)
            pts3d_depth0 = pts3d_depth0[z_mask0 & z_mask1]
            pts3d_depth1 = pts3d_depth1[z_mask0 & z_mask1]
            rtt = np.dot(R.T, t)
            # scale = np.sum(pts3d_depth0[z_mask0] * pts3d[z_mask0]) / np.sum(pts3d[z_mask0] * pts3d[z_mask0])
            scale = np.dot(pts3d_depth1 @ R - pts3d_depth0, rtt).sum() / np.sum(rtt * rtt) / pts3d_depth1.shape[0]
            t *= scale
        else:
            R, t = np.eye(3), np.zeros(3)

        current_pose = {'R': R, 't': t}

        results_extra = {'mconf':mconf_filter, 'mkpts0':mkpts0_filter, 'mkpts1':mkpts1_filter, 
                        'kpts0':kpts0, 'kpts1':kpts1, }

        return current_pose, results_extra
        

@torch.no_grad()
def estimate_rel_pose(intrinsics, img1, img2, mask1, depth1, depth2, umeyama=False, vis=False):
    opt = lambda: None
    opt.nms_radius = 4
    opt.keypoint_threshold = 5e-3
    opt.max_keypoints = 1024
    opt.superglue = 'indoor'
    opt.sinkhorn_iterations = 20
    opt.match_threshold = 0.2
    api = PoseEstimator(opt, intrinsics, intrinsics)

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    img1_tensor = torch.from_numpy(img1 / 255).cuda().float()[None, None]
    img2_tensor = torch.from_numpy(img2 / 255).cuda().float()[None, None]

    if umeyama:
        curr_pose_pred, results_extra = api.umeyama(img1_tensor, img2_tensor, mask1, depth1, depth2)   # 基于3D-3D的ransac匹配
    else:
        curr_pose_pred, results_extra = api.epipolar(img1_tensor, img2_tensor, mask1, depth1, depth2)
    
    pose = np.eye(4)
    pose[:3, :3] = curr_pose_pred['R']
    pose[:3, -1] = curr_pose_pred['t']
    
    if vis:
        out = np.full((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1] + 10), 255, np.uint8)
        out[:img1.shape[0], :img1.shape[1]] = img1
        out[:img2.shape[0], img1.shape[1] + 10:] = img2
        out = np.stack([out]*3, -1)
        
        kpts1, kpts2 = np.round(results_extra['kpts0']).astype(int), np.round(results_extra['kpts1']).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        color = cm.jet(results_extra['mconf'])
        for x, y in kpts1:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts2:
            cv2.circle(out, (x + 10 + img1.shape[1], y), 2, black, -1,
                        lineType=cv2.LINE_AA)
            cv2.circle(out, (x + 10 + img1.shape[1], y), 1, white, -1,
                        lineType=cv2.LINE_AA)

        mkpts1, mkpts2 = np.round(results_extra['mkpts0']).astype(int), np.round(results_extra['mkpts1']).astype(int)
        color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
        for (x0, y0), (x1, y1), c in zip(mkpts1, mkpts2, color):
            c = c.tolist()
            cv2.line(out, (x0, y0), (x1 + 10 + img1.shape[1], y1),
                    color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + 10 + img1.shape[1], y1), 2, c, -1,
                    lineType=cv2.LINE_AA)
        cv2.imshow('keypoints', out)
        cv2.waitKey()
    return pose

import os
def bcot_track(conn):
    if os.name == 'nt':
        from tracking.build.Release.tracking import RBOTTracker
    else:
        from tracking.build.tracking import RBOTTracker
    init_args = conn.recv()
    tracker = RBOTTracker(*init_args)
    
    msgs = conn.recv()
    while msgs[0] != 'end':
        if msgs[0] == 'add_model':
            tracker.add_model(*msgs[1:])
        elif msgs[0] == 'init':
            tracker.init(*msgs[1:])
        elif msgs[0] == 'track':
            rel_pose = tracker.track(*msgs[1:])
            conn.send(rel_pose)
        msgs = conn.recv()
    del tracker
    
    
if __name__ == '__main__':
    from build.Release.tracking import RBOTTracker
    img1 = cv2.imread('data/scenes/test/cam0/rgb/000145.png')
    img2 = cv2.imread('data/scenes/test/cam0/rgb/000155.png')
    mask1 = cv2.imread('data/scenes/test/cam0/mask/000145.png', cv2.IMREAD_GRAYSCALE) > 0
    depth1 = cv2.imread('data/scenes/test/cam0/depth/000145.png', cv2.IMREAD_ANYDEPTH) / 1000
    depth2 = cv2.imread('data/scenes/test/cam0/depth/000155.png', cv2.IMREAD_ANYDEPTH) / 1000

    pose = np.eye(4)
    pose[:3, :3] = np.array([
        [
            0.7464556042371445,
            0.6630212853134936,
            -0.0566286687482838
        ],
        [
            0.6616522154202268,
            -0.7485809003658376,
            -0.04292995966663101
        ],
        [
            -0.07085461687473502,
            -0.005423175150768849,
            -0.9974719106014058
        ]
    ])
    pose[:3, -1] = np.array([
        -0.2035722505631478,
        -0.18595798204025393,
        -0.03365204269100678
    ])
    # pose = flip_yz @ extrinsics @ pose

    flip_yz = np.eye(4)
    flip_yz[1:3, 1:3] *= -1
        
    intrinsics = []
    extrinsics = []
    for i in range(3):
        cam_info = json.load(open('data/scenes/test/cam{}/cam_info.json'.format(i)))
        intrinsic = np.array(cam_info['intrinsics'])
        intrinsic[0, 2] = 640 - intrinsic[0, 2]
        extrinsic = np.array(cam_info['extrinsics'])
        
        intrinsics.append(intrinsic.astype(np.float32))
        extrinsics.append((extrinsic @ flip_yz).astype(np.float32))
    tracker = RBOTTracker(intrinsics, extrinsics, 640, 480)
    
    obj = load_obj(Path('data/models/all/trashcan7'))
    vertices = []
    triangles = []
    cnt = 0
    for node in obj:
        mesh = node.mesh.simplify_quadric_decimation(len(node.mesh.triangles) // 100)
        vertices.append(np.array(mesh.vertices))
        triangles.append(np.array(mesh.triangles) + cnt)
        cnt += vertices[-1].shape[0]
        
    tracker.add_model(np.concatenate(vertices, 0).astype(np.float32), np.concatenate(triangles, 0).astype(np.int32), pose.astype(np.float32))
    print('added model')
    for img_id in range(145, 166):
        imgs = []
        for cam_id in range(3):
            img = cv2.imread('data/scenes/test/cam{}/rgb/{:06d}.png'.format(cam_id, img_id))
            imgs.append((img / 255.).astype(np.float32))
        if img_id == 145:
            tracker.init(imgs)
        else:
            rel_pose = tracker.track(imgs)
            print(np.linalg.inv(extrinsics[0]) @ rel_pose @ extrinsics[0])
    # pose = estimate_rel_pose(intrinsics, img1, img2, mask1, depth1, depth2, vis=False)
    # print(pose)
    
    # extrinsic1 = get_extrinsic(np.loadtxt(f'data/recorded/0.txt'), Path('data/recorded/rgb/cam0/{:06d}.png'.format(145)))
    # extrinsic2 = get_extrinsic(np.loadtxt(f'data/recorded/0.txt'), Path('data/recorded/rgb/cam0/{:06d}.png'.format(155)))
    # pose_gt = np.linalg.inv(extrinsic1) @ extrinsic2
    
    # print(np.arccos((np.trace(pose[:3, :3].T @ pose_gt[:3, :3]) - 1) / 2) / np.pi * 180)
    # print(pose_gt)