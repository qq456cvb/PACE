from pathlib import Path
import cv2
import numpy as np
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs
import sys
import os
sys.path.append(os.path.abspath("."))
from utils.config import Config


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


def get_extrinsic(intrinsic, img, marker_length=150):     # 临测for采集视频with两个marker
    dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
    arucoParams = aruco.DetectorParameters_create()
    arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    arucoParams.cornerRefinementWinSize = 5
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
        
        if False: # False: # 
            draw_img = aruco.drawDetectedMarkers(img, corners, ids,
                                                 (0, 255, 0))
            # print(corners)
            for r, t in zip(rvec, tvec):
                draw_img = cv2.drawFrameAxes(draw_img, intrinsic, dist_coeffs,
                                          r, t, 100)
            # cv2.imshow("Detected markers", cv2.resize(draw_img, (0, 0), fx=0.25, fy=0.25))
            cv2.imshow("Detected markers", draw_img)
            cv2.waitKey(0)
            
        return extrinsic, corners
    return None, None



            
if __name__ == '__main__':
    pipelines = []
    intrinsics = []
    for i, sn in enumerate(['038522062547', '039422060546', '104122063678']):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(sn)
        config.enable_stream(rs.stream.depth, Config.FRAME_WIDTH, Config.FRAME_HEIGHT, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, Config.FRAME_WIDTH, Config.FRAME_HEIGHT, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append(pipeline)
        
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        intrinsic = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        mtx = [intrinsic.width,intrinsic.height,intrinsic.ppx,intrinsic.ppy,intrinsic.fx,intrinsic.fy]
        camIntrinsics = np.array([[mtx[4],0,mtx[2]],
                                    [0,mtx[5],mtx[3]],
                                    [0,0,1.]])
        intrinsics.append(camIntrinsics)

    # intrinsic save
    np.save('data/intrinsics.npy', np.stack(intrinsics))

    start = False
    obj_pts = [[] for _ in range(3)]
    img_pts = [[] for _ in range(3)]
    try:
        while True:
            colors = []
            depths = []
            for pipeline in pipelines:
                # Camera 1
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                colors.append(color_image)
                depths.append(depth_image)

            # Stack all images horizontally
            images = np.concatenate(colors, 1)

            # Show images from both cameras
            cv2.namedWindow('RealSense')
            cv2.imshow('RealSense', cv2.resize(images, (0, 0), fx=0.25, fy=0.25))
            cv2.waitKey(1)

            # Save images and depth maps from both cameras by pressing 's'
            ch = cv2.waitKey(25)
            if ch == 27:
                exit()
            if ch == ord('s'):
                if not start:
                    print('starting')
                else:
                    print('stopping')
                    break
                start = not start
            if start:
                three_corners = []
                ress = []
                for i in range(3):
                    res, corners = get_extrinsic(intrinsics[i], colors[i], marker_length=Config.MARKER_SIZE)
                    ress.append(res)
                    three_corners.append(corners)
                    
                if np.any([res is None for res in ress]):
                    continue
                for i, corners in enumerate(three_corners):
                    img_pts[i].append(corners[0])
                    obj_pts[i].append(np.array([
                        [-Config.MARKER_SIZE / 2000., Config.MARKER_SIZE / 2000., 0], 
                        [Config.MARKER_SIZE / 2000., Config.MARKER_SIZE / 2000., 0], 
                        [Config.MARKER_SIZE / 2000., -Config.MARKER_SIZE / 2000., 0], 
                        [-Config.MARKER_SIZE / 2000., -Config.MARKER_SIZE / 2000., 0]
                    ]))

    finally:
        for pipeline in pipelines:
            pipeline.stop()
    
    for i in range(3):
        img_pts[i] = np.concatenate(img_pts[i]).astype(np.float32)
        obj_pts[i] = np.stack(obj_pts[i]).astype(np.float32)
        print(img_pts[i].shape)
        print(obj_pts[i].shape)
    rel_exs = []
    for j in range(1, 3):
        (
            retval,
            cameraMatrix1,
            distCoeffs1,
            cameraMatrix2,
            distCoeffs2,
            R,
            T,
            E,
            F,
        ) = cv2.stereoCalibrate(
            obj_pts[0],
            img_pts[0],
            img_pts[j],
            intrinsics[0],
            None,
            intrinsics[j],
            None,
            (Config.FRAME_WIDTH, Config.FRAME_WIDTH),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        ex = np.eye(4)
        ex[:3, :3] = R
        ex[:3, -1] = T[:, 0]
        rel_exs.append(ex)
    
    np.save('data/extrinsics.npy', np.stack(
        [np.eye(4), rel_exs[0], rel_exs[1]]
    ))
    exit()
    img_id_ls = [0, 5, 10]  # [148, 158, 166]
    rotvec1_ls, rotvec2_ls = [], []
    transvec1_ls, transvec2_ls = [], []
    for img_id in img_id_ls:
        extrinsics = []
        for i in range(3):
            # extrinsic = get_extrinsic(np.loadtxt(f'imgs2/{i+1}.txt'), Path('imgs2/cam{}_{:06d}.png'.format(i+1, img_id)))
            extrinsic = get_extrinsic_tmp(np.loadtxt(f'data/imgs/{i+1}.txt'), Path('data/imgs/cam{}_{:06d}.png'.format(i+1, img_id)))
            extrinsics.append(extrinsic)

        extrinsics[1] = extrinsics[1] @ np.linalg.inv(extrinsics[0])   # 对应Tcw, 把cam0下的坐标转到cam1下，即世界坐标到cam1的转换
        extrinsics[2] = extrinsics[2] @ np.linalg.inv(extrinsics[0])   # 对应Tcw
        extrinsics[0] = np.eye(4)                                      # 设定该相机坐标系，就是世界坐标系，自然它的外参就是单位阵！
        
        print(extrinsics[1])
        r1 = Rotation.from_matrix(extrinsics[1][:3,:3])
        r2 = Rotation.from_matrix(extrinsics[2][:3,:3])
        print(r1.as_rotvec() * 180 / np.pi)
        rotvec1_ls.append(r1.as_rotvec())
        rotvec2_ls.append(r2.as_rotvec())
        transvec1_ls.append(extrinsics[1][:3, -1])
        transvec2_ls.append(extrinsics[2][:3, -1])

    rotvec1 = np.stack(rotvec1_ls).mean(0)
    rotvec2 = np.stack(rotvec2_ls).mean(0)
    transvec1 = np.stack(transvec1_ls).mean(0)
    transvec2 = np.stack(transvec2_ls).mean(0)

    r1 = Rotation.from_rotvec(rotvec1)
    r2 = Rotation.from_rotvec(rotvec2)
    extrinsics[1][:3,:3] = r1.as_matrix()
    extrinsics[1][:3,-1] = transvec1
    extrinsics[2][:3,:3] = r2.as_matrix()
    extrinsics[2][:3,-1] = transvec2

    print('=====')
    print(extrinsics[1])


    for i in range(3):
        # intrinsic = np.loadtxt(f'imgs2/{i+1}.txt')
        intrinsic = np.loadtxt(f'data/imgs/{i+1}.txt')
        cam_info = dict(
            width=Config.FRAME_WIDTH,  # 640, #
            height=Config.FRAME_HEIGHT,  # 480, #
            depth_scale=1e-3,
            intrinsics=intrinsic.tolist(),
            extrinsics=extrinsics[i].tolist()  # opencv coordinate
        )
        # json.dump(cam_info, 
        #     open(f'D:\\3_Codes\\annotator-main\\data_new\\scene_003\\cam{i+1}\\cam_info.json', 'w'))

    
