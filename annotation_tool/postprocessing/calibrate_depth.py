import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import os
from tqdm import tqdm
from pyrender import DirectionalLight, SpotLight, Mesh, Scene, OffscreenRenderer, Camera, RenderFlags, MetallicRoughnessMaterial, IntrinsicsCamera
import trimesh
import json
import torch
from utils.transform import intrinsic2proj
import nvdiffrast.torch as dr


def get_pose(item):
    pose = np.eye(4)
    pose[:3, :3] = np.array(item['m2c_R'])
    pose[:3, -1] = np.array(item['m2c_t'])
    return pose


def render(glctx, proj, pos, pos_idx, uv, uv_idx, tex):
    # Setup TF graph for reference.
    pos_clip    = (proj @ pos[0].T).T[None]
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip.contiguous(), pos_idx.contiguous(), resolution=[720, 1280])

    texc, _ = dr.interpolate(uv, rast_out, uv_idx)
    color = dr.texture(tex, texc, filter_mode='linear')
    # texc, texd = dr.interpolate(uv, rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
    # color = dr.texture(tex, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=1)
    return color

if __name__ == '__main__':
    
    colormap = plt.get_cmap('jet')
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
            
            depth_data = ((depth_data - 0.5) * 2. * 255.).astype(np.uint8)
            
            
            heatmap = (colormap(depth_data) * 255.).astype(np.uint8)[:,:,:3]
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            
            fused = (img_data[..., ::-1] * 0.5 + heatmap * 0.5).astype(np.uint8)
            
            # cv2.imshow('heatmap', heatmap)
            cv2.imshow('fused', fused)
            if cv2.waitKey() == ord('q'):
                exit()
    
    # glctx = dr.RasterizeGLContext()
    # root = 'data/scenes/scene_0/video_0'
    # intrinsics = np.load(Path(root) / 'intrinsics.npy')[0]
    # cam_pose = np.eye(4)
    # cam = IntrinsicsCamera(intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
    # mesh = trimesh.load('data/models_aligned_lowres/can/001.obj', force='mesh', process=True)
    # scale_mat = np.eye(4)
    # scale_mat[:3, :3] *= 1e-3
    # mesh.apply_transform(scale_mat)

    # r = OffscreenRenderer(viewport_width=1280, viewport_height=720)
    
    # direc_l = DirectionalLight(color=np.ones(3), intensity=3)
    # spot_l = SpotLight(color=np.ones(3), intensity=0,
    #                 innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
    
    # scene = Scene(ambient_light=np.array([0., 0., 0., 0.0]), bg_color=np.zeros((3,)))
    # scene.add(cam, pose=cam_pose)
    # scene.add(direc_l, pose=cam_pose)
    # scene.add(spot_l, pose=cam_pose)
    
    # content = json.load(open('data/scenes/scene_0/video_0/cam1/pose/0000.json', 'r'))
    # mesh_pose = get_pose(content['1'])
    # # mesh_pose[1:3] *= -1  # to opengl

    # mesh.apply_transform(mesh_pose)
    
    # node = scene.add(Mesh.from_trimesh(mesh), pose=np.eye(4))
    
    # rendered, render_depth = r.render(scene)
    
    # img_data = cv2.imread(os.path.join(root, 'cam1/rgb/rgb{:04d}.png'.format(0)))[..., :3][:, :, ::-1]
    # fused = (img_data * 0.5 + rendered * 0.5).astype(np.uint8)[..., ::-1]
    # cv2.imshow('fused', fused)
    # cv2.waitKey()
    
    
    # pos = np.array(mesh.vertices)
    # pos = np.concatenate([pos, np.ones((pos.shape[0], 1), dtype=np.float32)], -1)
    # pos = torch.from_numpy(pos).float().cuda()[None]
    # uv = torch.from_numpy(np.array(mesh.visual.uv)).float().cuda()[None]
    # print(uv)
    # tex = torch.from_numpy(cv2.imread('data/models_aligned_lowres/can/001.jpg')[::-1, :, ::-1] / 255.).float().cuda()[None]
    # pos_idx = torch.from_numpy(mesh.faces).int().cuda()
    # uv_idx = pos_idx
    
    # intrinsics_gl = cam.get_projection_matrix(1280, 720)
    # vp = torch.from_numpy(intrinsic2proj(intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2], 1280, 720)).cuda().float()
    # rendered = (render(glctx, vp, pos, pos_idx, uv, uv_idx, tex).cpu().numpy() * 255.).astype(np.uint8)[0][::-1]
    
    # fused = (img_data * 0.5 + rendered * 0.5).astype(np.uint8)[..., ::-1]
    # cv2.imshow('fused2', fused)
    # cv2.waitKey()