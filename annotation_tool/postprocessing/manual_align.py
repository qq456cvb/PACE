import numpy as np
import open3d as o3d
import cv2
import json
import os
from pathlib import Path
from utils.transform import backproject
import utils.transform as transform
import copy
from utils.miscellaneous import avg_poses
from tqdm import tqdm
from functools import partial
from utils.config import Config
import string

def downsample(pc, res):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    _, _, idxs = pcd.voxel_down_sample_and_trace(res, pcd.get_min_bound(), pcd.get_max_bound())
    res = []
    for idx in idxs:
        res.append(np.random.choice(np.array(idx)))
    return np.array(res)


def align_3d(root):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name='Segmented Scene',
        width=1920,
        height=1080,
        left=480,
        top=270)
    vis.get_render_option().background_color = [0.0, 0.0, 0.0]
    vis.get_render_option().point_size = 5

    intrinsics = np.load(Path(root) / 'intrinsics.npy')
    extrinsics = np.load(Path(root) / 'extrinsics_refined.npy')
    
    anno_ids = [2]
    
    # for img_id in tqdm(range(0, len(list((Path(root) / 'cam0/rgb').glob('*.png'))), 10)):
    img_id= 120
    scene_pcs = {}
    for i in [0] + anno_ids:
        img_data = cv2.imread(os.path.join(root, 'cam{}/rgb_marker/rgb{:04d}.png'.format(i, img_id)))[..., :3][:, :, ::-1]
        depth_data = cv2.imread(os.path.join(root, 'cam{}/depth/depth{:04d}.png'.format(i, img_id)), cv2.IMREAD_ANYDEPTH)[:] * 1e-3
        depth_data[depth_data > 2] = 0
        pc, idxs = backproject(depth_data, intrinsics[i])
        
        pc = (pc - extrinsics[i][:3, -1]) @ extrinsics[i][:3, :3]
        
        sub_idx = downsample(pc, 5e-3)
        pc = pc[sub_idx]
        
        y, x = idxs
        x = x[sub_idx]
        y = y[sub_idx]
        
        scene_pc = o3d.geometry.PointCloud()
        scene_pc.points = o3d.utility.Vector3dVector(pc)
        scene_pc.colors = o3d.utility.Vector3dVector(img_data[y, x] / 255.)
        scene_pcs[i] = scene_pc
        vis.add_geometry(scene_pc)

    trans_interval = 1e-3
    rot_interval = 0.1 / 180 * np.pi
    refined_extrinsics = extrinsics.copy()
    def update_pc(key, pc_id, vis):
        # trans = extrinsics[pc_id][:3, -1]
        # scene_pcs[pc_id].translate((trans * delta).tolist())
        
        # # update extrinsic
        # trans_mat = np.eye(4)
        # trans_mat[:3, -1] = -trans * delta
        # refined_extrinsics[pc_id] = refined_extrinsics[pc_id] @ trans_mat
        
        delta = np.eye(4)
        if key == ord('Q'):
            delta[:3, 3] += refined_extrinsics[pc_id][:3, 0] * trans_interval
        elif key == ord('E'):
            delta[:3, 3] -= refined_extrinsics[pc_id][:3, 0] * trans_interval
        elif key == ord('W'):
            delta[:3, 3] += refined_extrinsics[pc_id][:3, 2] * trans_interval
        elif key == ord('S'):
            delta[:3, 3] -= refined_extrinsics[pc_id][:3, 2] * trans_interval
        elif key == ord('D'):
            delta[:3, 3] += refined_extrinsics[pc_id][:3, 1] * trans_interval
        elif key == ord('A'):
            delta[:3, 3] -= refined_extrinsics[pc_id][:3, 1] * trans_interval
        elif key == ord('Z'):
            delta[:3, :3] = transform.rotation_around(refined_extrinsics[pc_id][:3, 0], rot_interval)
        elif key == ord('X'):
            delta[:3, :3] = transform.rotation_around(refined_extrinsics[pc_id][:3, 0], -rot_interval)
        elif key == ord('C'):
            delta[:3, :3] = transform.rotation_around(refined_extrinsics[pc_id][:3, 1], rot_interval)
        elif key == ord('V'):
            delta[:3, :3] = transform.rotation_around(refined_extrinsics[pc_id][:3, 1], -rot_interval)
        elif key == ord('B'):
            delta[:3, :3] = transform.rotation_around(refined_extrinsics[pc_id][:3, 2], rot_interval)
        elif key == ord('N'):
            delta[:3, :3] = transform.rotation_around(refined_extrinsics[pc_id][:3, 2], -rot_interval)
        refined_extrinsics[pc_id] = refined_extrinsics[pc_id] @ delta
        scene_pcs[pc_id].transform(np.linalg.inv(delta))
        vis.update_geometry(scene_pcs[pc_id])
        
    def save():
        np.save(Path(root) / 'extrinsics_refined.npy', refined_extrinsics)
        print('saved')
    
    img_datas = {}
    depth_datas = {}
    for i in anno_ids:
        img_data = cv2.imread(os.path.join(root, 'cam{}/rgb_marker/rgb{:04d}.png'.format(i, img_id)))[..., :3][:, :, ::-1]
        depth_data = cv2.imread(os.path.join(root, 'cam{}/depth/depth{:04d}.png'.format(i, img_id)), cv2.IMREAD_ANYDEPTH)[:] * 1e-3
        depth_data[depth_data > 2] = 0
        img_datas[i] = img_data
        depth_datas[i] = depth_data
        
    for ch in string.ascii_uppercase:
        vis.register_key_callback(ord(ch), partial(update_pc, ord(ch), anno_ids[0]))
    # rgb = cv2.imread(os.path.join(root, 'cam0/rgb_marker/rgb{:04d}.png'.format(img_id)))[..., :3][:, :, ::-1]
    imgs = {0: cv2.imread(os.path.join(root, 'cam0/rgb_marker/rgb{:04d}.png'.format(img_id)))[..., :3][:, :, ::-1]}
    import time        
    while True:
        # vis.register_key_callback(ord("Q"), partial(update_pc, 2e-3, 1))
        # vis.register_key_callback(ord("W"), partial(update_pc, -2e-3, 1))
        
        # vis.register_key_callback(ord("E"), partial(update_pc, 2e-3, 2))
        # vis.register_key_callback(ord("R"), partial(update_pc, -2e-3, 2))
        
        # vis.register_key_callback(ord("S"), save)
       
        
        
        for i in anno_ids:
            img_data, depth_data = img_datas[i], depth_datas[i]
            pc, idxs = backproject(depth_data, intrinsics[i])
            
            pc = (pc - refined_extrinsics[i][:3, -1]) @ refined_extrinsics[i][:3, :3]
            pc_color = img_data[idxs[0], idxs[1]]
            
            proj = (intrinsics[0] @ pc.T).T
            proj = (proj[:, :2] / proj[:, 2:]).astype(int)
            
            img = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
            proj = np.maximum(np.minimum(proj, [Config.FRAME_WIDTH - 1, Config.FRAME_HEIGHT - 1]), [0, 0])
            img[proj[:, 1], proj[:, 0]] = pc_color
            
            imgs[i] = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        
        img = np.clip(1 / 2. * imgs[0] + 1 / 2. * imgs[anno_ids[0]], 0, 255.).astype(np.uint8)
        cv2.imshow('img', img[..., ::-1])
        if cv2.waitKey(1) == 27:
            break
        vis.poll_events()
        vis.update_renderer()
    save()
        
        
def align_2d(root):
    intrinsics = np.load(Path(root) / 'intrinsics.npy')
    extrinsics = np.load(Path(root) / 'extrinsics_refined.npy')
    
    # for img_id in tqdm(range(0, len(list((Path(root) / 'cam0/rgb').glob('*.png'))), 10)):
    
    for img_id in [0]:
        imgs = [cv2.imread(os.path.join(root, 'cam0/rgb_marker/rgb{:04d}.png'.format(0, img_id)))[..., :3][:, :, ::-1]]
        pcs = [None]
        colors = [None]
        for i in range(1, 3):
            img_data = cv2.imread(os.path.join(root, 'cam{}/rgb_marker/rgb{:04d}.png'.format(i, img_id)))[..., :3][:, :, ::-1]
            depth_data = cv2.imread(os.path.join(root, 'cam{}/depth/depth{:04d}.png'.format(i, img_id)), cv2.IMREAD_ANYDEPTH)[:] * 1e-3
            depth_data[depth_data > 2] = 0
            pc, idxs = backproject(depth_data, intrinsics[i])
            
            pcs.append(pc)
            
            pc = (pc - extrinsics[i][:3, -1]) @ extrinsics[i][:3, :3]
            pc_color = img_data[idxs[0], idxs[1]]
            
            proj = (intrinsics[0] @ pc.T).T
            proj = (proj[:, :2] / proj[:, 2:]).astype(int)
            
            img = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
            proj = np.maximum(np.minimum(proj, [Config.FRAME_WIDTH - 1, Config.FRAME_HEIGHT - 1]), [0, 0])
            img[proj[:, 1], proj[:, 0]] = pc_color
            
            imgs.append(cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)))
            colors.append(pc_color)
    
    img = np.clip(1 / 2. * imgs[0] + 1 / 4. * imgs[1] + 1 / 4. * imgs[2], 0, 255.).astype(np.uint8)
    cv2.imshow('img', img[..., ::-1])
    
    refined_extrinsics = extrinsics.copy()
    def update_pc(delta, pc_id):
        trans = extrinsics[pc_id][:3, -1]
        
        # update extrinsic
        trans_mat = np.eye(4)
        trans_mat[:3, -1] = -trans * delta
        refined_extrinsics[pc_id] = refined_extrinsics[pc_id] @ trans_mat
        pc = (pcs[pc_id] - refined_extrinsics[pc_id][:3, -1]) @ refined_extrinsics[pc_id][:3, :3]
        
        proj = (intrinsics[0] @ pc.T).T
        proj = (proj[:, :2] / proj[:, 2:]).astype(int)
        
        img = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
        proj = np.maximum(np.minimum(proj, [Config.FRAME_WIDTH - 1, Config.FRAME_HEIGHT - 1]), [0, 0])
        img[proj[:, 1], proj[:, 0]] = colors[pc_id]
        
        imgs[pc_id] = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
            
    k = 0
    while k != 27:
        k = cv2.waitKey()
        
        if k == ord('q'):
            update_pc(2e-3, 1)
        elif k == ord('w'):
            update_pc(-2e-3, 1)
        elif k == ord('e'):
            update_pc(2e-3, 2)
        elif k == ord('r'):
            update_pc(-2e-3, 2)
        elif k == ord('s'):
            np.save(Path(root) / 'extrinsics_refined.npy', refined_extrinsics)
            print('saved')
            break
            
        img = np.clip(1 / 2. * imgs[0] + 1 / 4. * imgs[1] + 1 / 4. * imgs[2], 0, 255.).astype(np.uint8)
        cv2.imshow('img', img[..., ::-1])

if __name__ == '__main__':
    for root in list(sorted(Path('data/videos/scene_7').glob('video_6'))):
        align_3d(root)
    
