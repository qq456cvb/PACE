import json
from pathlib import Path
import cv2
import numpy as np
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from utils import transform

class PnPSolver:
    OP2D = 2
    OP3D = 3
    def __init__(self) -> None:
        self.clear()
        
    def add2D(self, pt):
        self.pts2D.append(pt)
        self.stack.append(self.OP2D)
        
    def add3D(self, pt):
        self.pts3D.append(pt)
        self.stack.append(self.OP3D)
    
    def undo(self):
        if len(self.stack) == 0:
            return None
        op = self.stack.pop()
        if op == self.OP2D:
            self.pts2D.pop()
        else:
            self.pts3D.pop()
        return op
        
    def clear(self):
        self.pts2D = []
        self.pts3D = []
        self.stack = []
    
    def align(self, cam_intrinsics, opengl=False):
        _, rvec, trans = cv2.solvePnP(np.stack(self.pts3D).astype(np.float32), 
                                      np.stack(self.pts2D).astype(np.float32), 
                                      cam_intrinsics,
                                      None,
                                      flags=cv2.SOLVEPNP_EPNP
                                      )
        
        if rvec is None or trans is None:
            return None
        rot, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, -1] = trans[:, 0]
        if opengl:
            pose[1:3] *= -1
        return pose
    

def get_extrinsic(intrinsic, img_path, marker_length=150):
    dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
    arucoParams = aruco.DetectorParameters_create()
    arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    arucoParams.cornerRefinementWinSize = 5
    img = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(
        img_gray, aruco_dict, parameters=arucoParams)

    # print(img.shape[:2])
    if not ids is None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, intrinsic, dist_coeffs)
        
        t = tvec[0]
        r = cv2.Rodrigues(rvec[0])[0]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = r
        extrinsic[:3, -1] = t / 1000
        
        if False:
            draw_img = aruco.drawDetectedMarkers(img, corners, ids,
                                                 (0, 255, 0))
            print(corners)
            for r, t in zip(rvec, tvec):
                draw_img = cv2.drawFrameAxes(draw_img, intrinsic, dist_coeffs,
                                          r, t, 100)
            cv2.imshow("Detected markers", cv2.resize(draw_img, (0, 0), fx=0.25, fy=0.25))
            cv2.waitKey(0)
            
        return extrinsic
    return None




            
if __name__ == '__main__':
    # pts3d = np.random.randn(3, 3)
    # pts2d = np.random.randn(3, 2) * 100
    
    # intrinsic = np.array(
    #     [
    #         [525, 0, 340],
    #         [0, 525, 240],
    #         [0, 0, 1],
    #     ], dtype=np.float32
    # )
    # _, rvec, trans = cv2.solvePnP(pts3d.astype(np.float32), 
    #                             pts2d.astype(np.float32), 
    #                             intrinsic, 
    #                             None,
    #                             flags=cv2.SOLVEPNP_EPNP
    #                             )
    
    img_id = 148
    # extrinsics = []
    # for i in range(3):
    #     extrinsic = get_extrinsic(np.loadtxt(f'data/recorded/{i}.txt'), Path('data/recorded/rgb/cam{}/{:06d}.png'.format(i, img_id)))
    #     extrinsics.append(extrinsic)
    
    # exit()
    # get_extrinsic(np.loadtxt(f'data/recorded/0.txt'), Path('data/test.jpg'))
    # extrinsic1 = get_extrinsic(np.loadtxt(f'data/recorded/0.txt'), Path('data/recorded/rgb/cam0/{:06d}.png'.format(145)))
    # extrinsic2 = get_extrinsic(np.loadtxt(f'data/recorded/0.txt'), Path('data/recorded/rgb/cam0/{:06d}.png'.format(166)))
    # print(np.linalg.inv(extrinsic1) @ extrinsic2)
    scene_vis = o3d.visualization.Visualizer()
    scene_vis.create_window(width=1920, height=1080)
    
    
    for i in range(3):
        img_data = cv2.imread('data/scenes/test/cam{}/rgb/{:06d}.png'.format(i, img_id))[..., :3][:, :, ::-1]
        cam_info = json.load(open('data/scenes/test/cam{}/cam_info.json'.format(i)))
        depth_data = cv2.imread('data/scenes/test/cam{}/depth/{:06d}.png'.format(i, img_id), cv2.IMREAD_ANYDEPTH)[:] * 1e-3
        depth_data[depth_data > 1] = 0
        pc, idxs = transform.backproject(depth_data, np.array(cam_info['intrinsics']))
        extrinsics = np.array(cam_info['extrinsics'])
        # extrinsics[1:3] *= -1
        pc = (pc - extrinsics[:3, -1]) @ extrinsics[:3, :3] 
        # pc[..., 1:] *= -1
        
        scene_pc = o3d.geometry.PointCloud()
        scene_pc.points = o3d.utility.Vector3dVector(pc)
        scene_pc.colors = o3d.utility.Vector3dVector(img_data[idxs[0], idxs[1]] / 255.)
        scene_vis.add_geometry(scene_pc)
    
    
    while True:
        scene_vis.poll_events()
        scene_vis.update_renderer()
    

    # extrinsics[0] = extrinsics[0] @ np.linalg.inv(extrinsics[1])
    # extrinsics[2] = extrinsics[2] @ np.linalg.inv(extrinsics[1])
    # extrinsics[1] = np.eye(4)
    # for i in range(3):
    #     intrinsic = np.loadtxt(f'data/recorded/{i}.txt')
    #     cam_info = dict(
    #         width=640,
    #         height=480,
    #         depth_scale=1e-3,
    #         intrinsics=intrinsic.tolist(),
    #         extrinsics=extrinsics[i].tolist()  # opencv coordinate
    #     )
    #     json.dump(cam_info, open('data/scenes/test/cam{}/cam_info.json'.format(i), 'w'))
