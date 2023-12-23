import numpy as np
from utils.render import AnnoScene
from utils.config import Config
from pathlib import Path
from utils.io import load_anno
import cv2
from tqdm import tqdm
import open3d as o3d

if __name__ == '__main__':
    root = Path('data/videos/scene_9/video_0')
    intrinsics = np.load(root / 'intrinsics.npy')
    extrinsics = np.load(root / 'extrinsics_refined.npy')
    
    flip_yz = np.eye(4)
    flip_yz[1:3, 1:3] *= -1
    
    for fn in tqdm(list((root / 'cam0/pose').glob('*.json'))):
        annotations = load_anno(fn)
        pose0 = annotations[(3, 3)].pose
        verts = np.array(o3d.io.read_triangle_mesh(annotations[(3, 3)].path).vertices) * 1e-3
        
        for i in range(3):
            pose = extrinsics[i] @ flip_yz @ pose0  # this is the pose, opengl2opencv
            
            pc = (pose[:3, :3] @ verts.T).T + pose[:3, -1]
            proj = (intrinsics[i] @ pc.T).T
            proj = (proj[:, :2] / proj[:, 2:]).astype(int)
            
            img = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
            proj = np.maximum(np.minimum(proj, [Config.FRAME_WIDTH - 1, Config.FRAME_HEIGHT - 1]), [0, 0])
            img[proj[:, 1], proj[:, 0]] = 255
            cv2.imshow(f'proj{i}', img)
            
        if cv2.waitKey() == 27:
            break
        
        