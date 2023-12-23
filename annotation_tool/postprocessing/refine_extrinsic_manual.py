import numpy as np
import open3d as o3d
import cv2
import json
import os
from pathlib import Path
from utils.transform import backproject
import copy
from utils.miscellaneous import avg_poses
from tqdm import tqdm
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import make_matching_plot_fast
import torch
import matplotlib.cm as cm
import matlab.engine
import seaborn as sns


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])
    
def downsample(pc, res):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    _, _, idxs = pcd.voxel_down_sample_and_trace(res, pcd.get_min_bound(), pcd.get_max_bound())
    res = []
    for idx in idxs:
        res.append(np.random.choice(np.array(idx)))
    return np.array(res)


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask[:, 0] > 0)
    return ret


def visualize_scenes():
    scene_vis = o3d.visualization.Visualizer()
    scene_vis.create_window(width=1920, height=1080)
    
    root = 'data/scenes/scene_0/video_0'
    intrinsics = np.load(Path(root) / 'intrinsics.npy')
    extrinsics = np.load(Path(root) / 'extrinsics.npy')
    
    rel_extrinsics1 = []
    rel_extrinsics2 = []
    for img_id in tqdm(range(0, len(list((Path(root) / 'cam0/rgb').glob('*.png'))), 10)):
        scene_pcs = []
        for i in range(3):
            img_data = cv2.imread(os.path.join(root, 'cam{}/rgb/rgb{:04d}.png'.format(i, img_id)))[..., :3][:, :, ::-1]
            depth_data = cv2.imread(os.path.join(root, 'cam{}/depth/depth{:04d}.png'.format(i, img_id)), cv2.IMREAD_ANYDEPTH)[:] * 1e-3
            depth_data[depth_data > 2] = 0
            pc, idxs = backproject(depth_data, intrinsics[i])
            # extrinsics[1:3] *= -1
            pc = (pc - extrinsics[i][:3, -1]) @ extrinsics[i][:3, :3]
            sub_idx = downsample(pc, 5e-3)
            pc = pc[sub_idx]
            # pc[..., 1:] *= -1
            y, x = idxs
            x = x[sub_idx]
            y = y[sub_idx]
            
            scene_pc = o3d.geometry.PointCloud()
            scene_pc.points = o3d.utility.Vector3dVector(pc)
            scene_pc.colors = o3d.utility.Vector3dVector(img_data[y, x] / 255.)
            scene_pc.estimate_normals()
            scene_pcs.append(scene_pc)
    
    for i in range(3):
        scene_vis.add_geometry(scene_pcs[i])
    
    while True:
        scene_vis.poll_events()
        scene_vis.update_renderer()



if __name__ == '__main__':
    eng = matlab.engine.start_matlab()
    s = eng.genpath('./TFT_vs_Fund')
    eng.addpath(s, nargout=0)
    
    torch.set_grad_enabled(False)
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().cuda()
        

    for root in list(sorted(Path('data/videos/scene_0').glob('video_6'))):
        intrinsics = np.load(root / 'intrinsics.npy')
        
        Corresp = []
        palette = sns.color_palette("bright", 100)
        # len(list((root / 'cam0/rgb').glob('*.png'))) - 1
        for img_id in tqdm([0, len(list((root / 'cam0/rgb').glob('*.png'))) - 1]):
            bgrs = {}
            vis_bgrs = {}
            
            # automatic compute
            pred_sp = {}
            for i in range(3):
                bgr = cv2.imread(str(root / f'cam{i}/rgb/rgb{img_id:04d}.png'))
                gray = (torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)) / 255.).float().cuda()[None, None]
                pred_curr = matching.superpoint({'image': gray})
                pred_sp = {**pred_sp, **{k+str(i): v for k, v in pred_curr.items()}, **{'image'+str(i): gray}}

            valid0 = np.ones((pred_sp['keypoints0'][0].shape[0],), dtype=bool)
            triplet_matches = [np.arange(pred_sp['keypoints0'][0].shape[0])]
            for j in range(1, 3):
                pred = matching({'image0': pred_sp['image0'], 
                                'image1': pred_sp[f'image{j}'],
                                'keypoints0': pred_sp['keypoints0'],
                                'descriptors0': pred_sp['descriptors0'],
                                'scores0': pred_sp['scores0'],
                                'keypoints1': pred_sp[f'keypoints{j}'],
                                'descriptors1': pred_sp[f'descriptors{j}'],
                                'scores1': pred_sp[f'scores{j}']})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred_sp['keypoints0'][0].cpu().numpy(), pred_sp[f'keypoints{j}'][0].cpu().numpy()
                matches, conf = pred['matches0'], pred['matching_scores0']
                valid = np.where((matches > -1) & (conf > 0.5))[0]
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]

                _, _, mask = estimate_pose(mkpts0, mkpts1, intrinsics[0], intrinsics[1], 0.5)
                mkpts0 = mkpts0[mask]
                mkpts1 = mkpts1[mask]
                mconf = mconf[mask]
                valid = valid[mask]

                tmp_mask = np.zeros((valid0.shape[0],), dtype=bool)
                tmp_mask[valid] = True
                valid0 = valid0 & tmp_mask

                triplet_matches.append(matches)
            
            pts = {i: pred_sp[f'keypoints{i}'][0].cpu().numpy()[triplet_matches[i][valid0]].astype(int).tolist() for i in range(3)}
            # pts = {i: [] for i in range(3)}
            def callback(event, x, y, flags, param):
                vis_bgrs, bgrs, pts, i = param
                for j in range(3):
                    vis_bgrs[j][...] = bgrs[j]
                if event == cv2.EVENT_LBUTTONDOWN:
                    pts[i].append([x, y])
                elif event == cv2.EVENT_MOUSEMOVE:
                    pt = np.array([x, y])
                    if len(pts[i]) == 0:
                        return
                    p_idx = np.argmin(np.linalg.norm(np.array(pts[i]) - pt, axis=-1))
                    
                    if np.linalg.norm(np.array(pts[i][p_idx]) - pt) < 10:
                        for j in range(3):
                            try:
                                cv2.circle(vis_bgrs[j], pts[j][p_idx], 4, (204, 204, 204), 2)
                            except:
                                pass
                elif event == cv2.EVENT_RBUTTONDOWN:
                    pt = np.array([x, y])
                    if len(pts[i]) == 0:
                        return
                    p_idx = np.argmin(np.linalg.norm(np.array(pts[i]) - pt, axis=-1))
                    
                    if np.linalg.norm(np.array(pts[i][p_idx]) - pt) < 10:
                        for j in range(3):
                            try:
                                del pts[j][p_idx]
                            except:
                                pass
                for j in range(3):
                    for k, pt in enumerate(pts[j]):
                        color = np.array(palette[k]) * 255
                        cv2.circle(vis_bgrs[j], pt, 3, (int(color[0]), int(color[1]), int(color[2])), -1)
                
            for i in range(3):
                bgr = cv2.imread(str(root / f'cam{i}/rgb/rgb{img_id:04d}.png'))
                gray = (torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)) / 255.).float().cuda()[None, None]
                cv2.namedWindow(f'image{i}')
                bgrs[i] = bgr
                vis_bgrs[i] = bgr.copy()
                cv2.setMouseCallback(f'image{i}', callback, [vis_bgrs, bgrs, pts, i])
            
            while True:
                for i in range(3):
                    cv2.imshow(f'image{i}', vis_bgrs[i])
                if cv2.waitKey(1) == ord('q'):
                    break
            
            Corresp.append(np.concatenate([np.array(pts[i]) for i in range(3)], -1).T)
        
        CalM = np.concatenate(intrinsics)
        Corresp = np.concatenate(Corresp, -1)

        print(Corresp.shape)
        R_t_2, R_t_3, Reconst = eng.FaugPapaTFTPoseEstimation(matlab.double(Corresp.tolist()), matlab.double(CalM.tolist()), nargout=3)

        R_t_2 = np.asarray(R_t_2)
        R_t_3 = np.asarray(R_t_3)
        Reconst = np.asarray(Reconst)
        # Compute the errors
        # reprojection error
        repr_err = eng.ReprError([matlab.double(np.concatenate([intrinsics[0], np.zeros((3, 1))], -1).tolist()),
                                matlab.double((intrinsics[1] @ R_t_2).tolist()),
                                matlab.double((intrinsics[2] @ R_t_3).tolist())], matlab.double(Corresp.tolist()), matlab.double(Reconst.tolist()))
        print('Reprojection error is {}'.format(repr_err))

        R_t_ref, Reconst_ref, iter, repr_err = eng.BundleAdjustment(matlab.double(CalM.tolist()), 
                                                                matlab.double(np.concatenate([np.concatenate([np.eye(3), np.zeros((3, 1))], -1), R_t_2, R_t_3]).tolist()), 
                                                                matlab.double(Corresp.tolist()), 
                                                                matlab.double(Reconst.tolist()), 
                                                                nargout=4)
        print('Reprojection error is {} after Bundle Adjustment'.format(repr_err))

        R_t_ref = np.asarray(R_t_ref)
        print(R_t_ref.shape)

        extrinsics = np.stack([np.eye(4) for _ in range(3)])
        extrinsics[1][:3] = R_t_ref[3:6]
        extrinsics[2][:3] = R_t_ref[6:9]

        trans = np.concatenate([extrinsics[1][:3, -1], extrinsics[2][:3, -1]])

        extrinsics_raw = np.load(root / 'extrinsics.npy')
        trans_raw = np.concatenate([extrinsics_raw[1][:3, -1], extrinsics_raw[2][:3, -1]])
        
        scale = np.dot(trans_raw, trans) / np.dot(trans, trans)

        extrinsics[1][:3, -1] *= scale
        extrinsics[2][:3, -1] *= scale
        np.save(root / 'extrinsics_refined.npy', extrinsics)