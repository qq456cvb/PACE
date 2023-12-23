from pathlib import Path
import cv2
from inpainting.inpaint import detect
from utils.config import Config
import numpy as np
from tqdm import tqdm
from utils.miscellaneous import avg_poses
from PIL import Image


def filter(intrinsics, extrinsics, bg_imgs, rel_imgs, record_imgs):
    db = []
    id1 = None
    for bg_img in tqdm(bg_imgs):
        res = detect(intrinsics[0], bg_img, marker_length=Config.MARKER_SIZE)
        if res is not None:
            cam_pose = np.linalg.inv(list(res.values())[0]['extrinsic'])  # in id1 frame
            id1 = list(res.keys())[0]
            db.append({
                'cam_pose': cam_pose,
                'img': bg_img,
                'corner': list(res.values())[0]['corner']
            })
    
    id2 = None
    poses = []
    for rel_img in tqdm(rel_imgs):
        res = detect(intrinsics[0], rel_img, marker_length=Config.MARKER_SIZE)
        pose1, pose2 = None, None
        if len(res) == 2:
            for marker_id in res:
                if marker_id == id1:
                    pose1 = res[id1]['extrinsic']
                else:
                    id2 = marker_id
                    pose2 = res[id2]['extrinsic']
            poses.append(np.linalg.inv(pose1) @ pose2)  # 2 to 1
    rel_pose = avg_poses(poses)
    
    for record_img in tqdm(record_imgs):
        res = detect(intrinsics[0], record_img[0], marker_length=Config.MARKER_SIZE)
        if res is not None and id2 in res:
            cam_pose0 = rel_pose @ np.linalg.inv(res[id2]['extrinsic'])
            pose = np.linalg.inv(cam_pose0)
            
            for i in range(3):
                ex = extrinsics[i] @ pose  # id1 projection in each frame

                marker_size = Config.MARKER_SIZE / 1000.
                corners = np.array(
                    [
                        [-0.5 * marker_size, 0.5 * marker_size, 0, 1],
                        [0.5 * marker_size, 0.5 * marker_size, 0, 1],
                        [0.5 * marker_size, -0.5 * marker_size, 0, 1],
                        [-0.5 * marker_size, -0.5 * marker_size, 0, 1],
                    ]
                )
                proj = (ex @ corners.T).T
                proj = (intrinsics[i] @ proj[:, :3].T).T
                proj = proj[:, :2] / proj[:, 2:3]
                
                # dist = np.array([np.linalg.norm(np.linalg.inv(ex)[:3, -1] - p['cam_pose'][:3, -1]) for p in self.db])
                dist = np.array([np.arccos((np.trace(np.linalg.inv(ex)[:3, :3].T @ p['cam_pose'][:3, :3]) - 1.) / 2) for p in db])
                best_idx = np.argmin(dist)

                im_src = db[best_idx]['img']
                im_dst = record_img[i]
                h, status = cv2.findHomography(db[best_idx]['corner'], proj)
                # Warp source image to destination based on homography
                im_src = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
                
                mask = np.zeros((im_dst.shape[0], im_dst.shape[1]), np.uint8)
                ratio = 1.5 / 2.
                a4corners = np.array(
                    [
                        [-marker_size * ratio, marker_size * ratio, 0, 1],
                        [marker_size * ratio, marker_size * ratio, 0, 1],
                        [marker_size * ratio, -marker_size * ratio, 0, 1],
                        [-marker_size * ratio, -marker_size * ratio, 0, 1],
                    ]
                )
                
                proj = (extrinsics[i] @ res[id2]['extrinsic'] @ a4corners.T).T
                proj = (intrinsics[i] @ proj[:, :3].T).T
                proj = proj[:, :2] / proj[:, 2:3]
                proj = proj.astype(int)
                cv2.fillConvexPoly(mask, proj, 255)
                br = cv2.boundingRect(mask) # bounding rect (x,y,width,height)
                centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
                # im_inpaint[mask > 0] = im_out[mask > 0]
                bbox = Image.fromarray(mask).getbbox()
                record_img[i] = cv2.seamlessClone(np.array(Image.fromarray(im_src).crop(bbox)), 
                                                  im_dst, 
                                                  np.array(Image.fromarray(mask).crop(bbox)), 
                                                  centerOfBR, 
                                                  cv2.MIXED_CLONE)
                vis = np.array(Image.fromarray(im_src).crop(bbox))
                vis = vis * (np.array(Image.fromarray(mask).crop(bbox)) > 0).astype(np.uint8)[..., None]
                vis2 = im_dst * (mask > 0).astype(np.uint8)[..., None]
                cv2.imshow(f'img{i}', record_img[i])
                cv2.imshow(f'src{i}', vis)
                cv2.imshow(f'dst{i}', vis2)
                cv2.waitKey()


if __name__ == '__main__':
    root = Path('data/videos/scene_9/video_0')
    intrinsics = np.load(root / 'intrinsics.npy')
    extrinsics = np.load(root / 'extrinsics_refined.npy')

    bg_imgs = [cv2.imread(str(fn)) for fn in sorted(Path(root / 'aux1').glob('bg*.png'))]
    rel_imgs = [cv2.imread(str(fn)) for fn in sorted(Path(root / 'aux1').glob('rel*.png'))]
    record_imgs = []
    for i in range(3):
        record_imgs.append([cv2.imread(str(fn)) for fn in sorted(Path(root / f'cam{i}/rgb_marker').glob('*.png'))])
    record_imgs = list(map(list, zip(*record_imgs)))
    print(len(record_imgs))
    filter(intrinsics, extrinsics, bg_imgs, rel_imgs, record_imgs)