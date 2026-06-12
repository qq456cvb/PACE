from utils.render import AnnoScene
import numpy as np
from utils.config import Config
from pathlib import Path
from utils.io import load_anno
import cv2
from tqdm import tqdm


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Render instance masks for all cameras from saved pose annotations.')
    parser.add_argument('root', help='video folder, e.g. data/videos/scene_0/video_0')
    args = parser.parse_args()

    root = Path(args.root)
    intrinsics = np.load(root / 'intrinsics.npy')
    extrinsics = np.load(root / 'extrinsics_refined.npy')
    anno_scene = AnnoScene()
    for i in range(3):
        flip_yz = np.eye(4)
        flip_yz[1:3, 1:3] *= -1
        anno_scene.add_frame(intrinsics[i], flip_yz @ extrinsics[i] @ flip_yz, Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
        
    for fn in tqdm(list((root / 'cam0/pose').glob('*.json'))):
        annotations = load_anno(fn)
        for (start, _), obj in annotations.items():
            anno_scene.add(obj, start)
        anno_scene.update()
        
        for i in range(3):
            mask = anno_scene.get_mask(i)[::-1]
            (fn.parent.parent.parent / f'cam{i}' / 'mask').mkdir(exist_ok=True)
            cv2.imwrite(str(fn.parent.parent.parent / f'cam{i}' / 'mask' / (fn.stem + '.png')), mask)
        anno_scene.clear()