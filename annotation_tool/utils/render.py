import json
from pathlib import Path
import cv2
import nvdiffrast.torch as dr
import numpy as np
from trimesh import Trimesh
import torch
import trimesh
from utils.align import get_extrinsic
from utils.transform import intrinsic2proj
from collections import OrderedDict
from torch.optim import Adam, SGD

class AnnoScene:
    def __init__(self) -> None:
        self.glctx = dr.RasterizeCudaContext()
        self.pos = OrderedDict()
        self.pos_idx = OrderedDict()
        self.color = OrderedDict()
        self.color_idx = OrderedDict()
        self.mask = OrderedDict()
        
        self.width = None
        self.height = None
        
        self.rgbs = []
        self.masks = []
        self.projs = []
        self.extrinsics = []
        
    def clear(self):
        self.pos = OrderedDict()
        self.pos_idx = OrderedDict()
        self.color = OrderedDict()
        self.color_idx = OrderedDict()
        self.mask = OrderedDict()
        
        self.rgbs = [np.zeros((self.height, self.width, 3), dtype=np.uint8) for _ in range(len(self.projs))]
        self.masks = [np.zeros((self.height, self.width), dtype=np.uint8) for _ in range(len(self.projs))]
    
    def add_frame(self, intrinsic, extrinsic, width, height):
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = intrinsic2proj(fx, fy, cx, cy, width, height)
        
        if self.width is None:
            self.width = width
            self.height = height
        else:
            assert(self.width == width, "does not support multi camera with different image size")
        
        proj = intrinsic @ extrinsic
        self.extrinsics.append(extrinsic)
        self.projs.append(torch.from_numpy(proj).cuda().float())
        self.masks.append(np.zeros((self.height, self.width), dtype=np.uint8))
        self.rgbs.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
    
    @torch.no_grad()
    def update(self):
        if len(self.pos) == 0:
            self.rgbs = [np.zeros((self.height, self.width, 3), dtype=np.uint8) for _ in range(len(self.projs))]
            self.masks = [np.zeros((self.height, self.width), dtype=np.uint8) for _ in range(len(self.projs))]
            return
        cnt = 0
        
        pos = []
        color = []
        mask = []
        pos_idx = []
        
        for obj in self.pos:
            pos.append((torch.from_numpy(obj.pose).float().cuda() @ self.pos[obj].T).T)
            color.append(self.color[obj])
            mask.append(self.mask[obj])  # offset by 1
            pos_idx.append(self.pos_idx[obj] + cnt)
            cnt += self.pos[obj].shape[0]
        pos = torch.cat(pos)
        pos_idx = torch.cat(pos_idx)
        color = torch.cat(color)
        mask = torch.cat(mask)
        
        pos = torch.stack([(self.projs[i] @ pos.T).T.contiguous() for i in range(len(self.projs))])
        rast_out, _ = dr.rasterize(self.glctx, pos, pos_idx, resolution=[self.height, self.width])
        color   , _ = dr.interpolate(color[None, ...], rast_out, pos_idx)
        mask, _ = dr.interpolate(mask[None, ...], rast_out, pos_idx)
        
        color = (color.cpu().numpy() * 255).astype(np.uint8)
        mask = (mask[..., 0].cpu().numpy() + 0.5).astype(np.uint8)
        
        self.rgbs = color
        self.masks = mask
        
    def __len__(self):
        return len(self.pos)
        
    def add(self, obj, base_idx):
        idx = base_idx
        def add_rec(node):
            nonlocal idx
            if node.is_leaf():
                node.mesh.transform(np.linalg.inv(node.pose))
                vtx_pos = torch.from_numpy(np.array(node.mesh.vertices)).cuda().float()
                vtx_pos = torch.cat([vtx_pos, torch.ones((vtx_pos.shape[0], 1)).to(vtx_pos)], -1)
                pos_idx = torch.from_numpy(np.array(node.mesh.triangles)).cuda().int()
                vtx_col = torch.from_numpy(np.array(node.mesh.vertex_colors)).cuda().float()
                col_idx = pos_idx
                node.mesh.transform(node.pose)
                
                mask = torch.full((vtx_pos.shape[0], 1), idx, dtype=torch.float32, device='cuda')
                
                self.pos[node]  = vtx_pos
                self.pos_idx[node] = pos_idx
                self.color[node] = vtx_col
                self.color_idx[node] = col_idx
                self.mask[node] = mask
                
                idx += 1
            else:
                for child in node.children:
                    add_rec(child)
                    
        add_rec(obj)
        
    def remove(self, obj):
        for child in obj:
            del self.pos[child]
            del self.pos_idx[child]
            del self.color[child]
            del self.color_idx[child]
            del self.mask[child]
    
    def get_rgb(self, frame_idx):
        return self.rgbs[frame_idx]
    
    def get_mask(self, frame_idx):
        return self.masks[frame_idx]

    @torch.no_grad()
    def render_depth(self, node, frame_id=0):
        if len(node) == 0:
            return
        cnt = 0
        
        pos = []
        pos_idx = []
        depth = []
        
        for obj in node:
            pos.append((torch.from_numpy(obj.pose).float().cuda() @ self.pos[obj].T).T)
            pos_idx.append(self.pos_idx[obj] + cnt)
            depth.append(-(torch.from_numpy(self.extrinsics[frame_id]).float().cuda() @ self.pos[obj].T).T[..., 2:3])
            cnt += self.pos[obj].shape[0]
        pos = torch.cat(pos)
        pos_idx = torch.cat(pos_idx)
        depth = torch.cat(depth)
        depth_idx = pos_idx
        
        pos = torch.stack([(self.projs[frame_id] @ pos.T).T.contiguous()])
        rast_out, _ = dr.rasterize(self.glctx, pos, pos_idx, resolution=[self.height, self.width])
        
        depth, _ = dr.interpolate(depth[None, ...], rast_out, depth_idx)
        
        depth = depth[0, ..., 0].cpu().numpy()
        
        return depth

def compose_img(imgs, masks, alphas):
    curr = np.zeros_like((imgs[0]))
    for img, mask, alpha in zip(imgs, masks, alphas):
        if alpha > 1 - 1e-7:
            curr[mask] = img[mask]
            continue
        if alpha < 1e-7:
            continue
        curr[mask] = curr[mask] * (1 - alpha) + img[mask] * alpha
    return curr
    

def render_silhouette(mask, color, size=3):
    mask = mask.astype(np.uint8)
    kernel = np.ones((size, size), np.uint8)
    dilated = cv2.dilate(mask, kernel)
    res = dilated - mask
    img = res[..., None] * np.array(color, dtype=np.uint8)
    return img
    

if __name__ == '__main__':
    # mask = np.zeros((480, 640), dtype=bool)
    # mask[240:360, 240:360] = True
    # img = render_silhouette(mask, (0, 0, 255), 5)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    cube = trimesh.load('cube.obj', force='mesh')

    glctx = dr.RasterizeGLContext()
    img_id = 148
    for i in range(3):
        intrinsics = np.loadtxt(f'data/recorded/{i}.txt')
        extrinsics = get_extrinsic(intrinsics, Path('data/recorded/rgb/cam{}/{:06d}.png'.format(i, img_id)))
        
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        intrinsics = intrinsic2proj(fx, fy, cx, cy, 640, 480)
        
        extrinsics[1:3] *= -1
        
        proj = intrinsics @ extrinsics
        
        rgb = cv2.imread('data/recorded/rgb/cam{}/{:06d}.png'.format(i, img_id))
        
        pos = cube.vertices * 0.02
        pos = np.concatenate([pos, np.ones((pos.shape[0], 1), dtype=np.float32)], -1)
        pos = (proj @ pos.T).T
        print(pos)
        
        pos = torch.from_numpy(pos).float().cuda()[None]
        
        pos_idx = torch.from_numpy(cube.faces).int().cuda()
        col_idx = pos_idx
        
        col = np.zeros((1, pos.shape[1], 3), dtype=np.float32)
        col[:] = np.array([0, 0, 1])
        col = torch.from_numpy(col).float().cuda()
        rast_out, _ = dr.rasterize(glctx, pos.contiguous(), pos_idx.contiguous(), resolution=[480, 640])
        color   , _ = dr.interpolate(col, rast_out, col_idx)
        color = (color.cpu().numpy() * 255).astype(np.uint8)[0][::-1]
        print(color.max(), color.min())
        vis = (rgb * 0.5 + color * 0.5).astype(np.uint8)
        cv2.imshow('img', vis)
        cv2.waitKey()
        