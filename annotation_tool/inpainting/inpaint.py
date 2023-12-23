from copy import copy, deepcopy
from io import BytesIO
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath("."))
from time import sleep, time
import cv2
import numpy as np
from cameras.Camera import Camera, Camera4multi
from tqdm import tqdm
import cv2.aruco as aruco
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QSizePolicy, QLabel, \
    QTreeWidgetItem, QFileDialog, QDialog, QDialogButtonBox, QVBoxLayout, QScrollArea, QPushButton, QStyle
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QFile, QThread, QObject, QTimer, Qt, QEvent, QPoint, QUrl
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QWindow, QPainter, QImage, QPixmap
from PyQt5 import uic
import sys
import skvideo.io
import tempfile
import pyrealsense2 as rs
from utils.config import Config
from utils.miscellaneous import avg_poses

save_root="./data/videos/"

def detect(intrinsic, img, marker_length = 150):
    dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
    arucoParams = aruco.DetectorParameters_create()
    arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    arucoParams.cornerRefinementWinSize = 5
    # img = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(
        img_gray, aruco_dict, parameters=arucoParams)  # -1, 1, 1, 1, 1, -1, -1, -1
    
    res = {}
    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, intrinsic, dist_coeffs)
        
        for rvec, tvec, marker_id, corner in zip(rvecs, tvecs, ids, corners):
            t = tvec[0]
            r = cv2.Rodrigues(rvec[0])[0]
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = r
            extrinsic[:3, -1] = t / 1000
            # draw_img = cv2.drawFrameAxes(img.copy(), intrinsic, dist_coeffs, r, t, 100)
            res[marker_id[0]] = {
                'extrinsic': extrinsic,
                'corner': corner[0]
            }
            # cv2.imshow('img', draw_img)
            # cv2.waitKey()
        return res
    else:
        return None


class VideoThread(QThread):
    img_signal = pyqtSignal(list, list)
    bg_signal = pyqtSignal(np.ndarray)
    rel_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.reset()
        self.cameras = [Camera4multi(cid, H=Config.FRAME_HEIGHT, W=Config.FRAME_WIDTH) for cid in ['038522062547', '039422060546', '104122063678']]

    def reset(self):
        self.recorded_imgs = []
        self.bg_flag = False
        self.rel_flag = False
        self.recording = False
        self.paused = False
        self.bg_img = None
        self.rel_img = None
        
    def run(self):
        t = time()
        while self._run_flag:
            sleep(0.1)
            if self.paused:
                continue
            # print('get frame')
            for i in range(3):
                self.cameras[i].get_frames()
            colors = []
            depths = []
            for i in range(3):
                color, depth = self.cameras[i].get_frames_data()
                # color = np.random.randint(0, 256, (720, 1280, 3), np.uint8)
                # depth = np.random.randint(0, 2000, (720, 1280), np.uint16)
                colors.append(color.copy())
                depths.append(depth.copy())
            
            if self.bg_flag:
                print('setting bg')
                self.bg_img = colors[0]
                self.bg_flag = False
                self.bg_signal.emit(colors[0])
            if self.rel_flag:
                print('setting rel')
                self.rel_img = colors[0]
                self.rel_flag = False
                self.rel_signal.emit(colors[0])
            if self.recording:
                self.recorded_imgs.append([colors, depths])
            
            if time() - t > 0.05:
                t = time()
                self.img_signal.emit(colors, depths)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
        
        
class Window(QMainWindow):
    
    def __init__(self):
        super().__init__()
        uic.loadUi(os.fspath(Path(__file__).resolve().parent / "inpaint.ui"), self)
        self.msg = QLabel()
        self.statusBar().addWidget(self.msg)
        
        self.toggle_button = self.findChild(QObject, 'toggleButton')
        self.toggle_button.clicked.connect(self.toggle_cam)
        def set_bg():
            self.th.bg_flag = True
        def set_rel():
            self.th.rel_flag = True
        self.bgButton = self.findChild(QObject, 'bgButton')
        self.bgButton.clicked.connect(set_bg)
        self.relButton = self.findChild(QObject, 'relButton')
        self.relButton.clicked.connect(set_rel)
        self.record_button = self.findChild(QObject, 'recordButton')
        self.record_button.clicked.connect(self.record)
        self.video_label = self.findChild(QObject, 'videoLabel')
        self.bg_label = self.findChild(QObject, 'bgLabel')
        self.rel_label = self.findChild(QObject, 'relLabel')
        self.showMaximized()
        self.get_frame = None
        
        self.ok_button = self.findChild(QObject, 'okButton')
        self.cancel_button = self.findChild(QObject, 'cancelButton')
        
        self.common_buttons = self.findChild(QObject, 'commonLayout')
        self.confirm_buttons = self.findChild(QObject, 'confirmLayout')
        
        for i in range(self.confirm_buttons.count()):
            self.confirm_buttons.itemAt(i).widget().setEnabled(False)
        
        timer = QTimer(self)
        timer.timeout.connect(self.draw_video)
        self.fps = 10
        timer.start(1000 / self.fps)
        
        self.cam_open = False
        self.curr_frame = None
        self.th = None
        self.bg_img = None
        self.rel_img = None
        
        self.video_widget = self.findChild(QObject, 'videoWidget')
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setNotifyInterval(100)
        self.mediaPlayer.setVideoOutput(self.video_widget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        
        self.playButton = self.findChild(QObject, 'playButton')
        self.playButton.setFixedHeight(30)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        
        self.positionSlider = self.findChild(QObject, 'videoSlider')
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        self.stacked_widget = self.findChild(QObject, 'stackedWidget')
        
        # self.intrinsic = np.array(
        #     [[612.89178467,   0.,         317.58123779],
        #     [  0.,         612.09307861, 238.63554382],
        #     [  0.,           0.,           1.        ]]
        # )
        self.scale = 0.25
        self.intrinsics = np.load('data/intrinsics.npy')
        for i in range(len(self.intrinsics)):
            self.intrinsics[i][:2, :3] *= self.scale
        
        self.extrinsics = np.load('data/extrinsics.npy')
        self.curr_idx = 0
        self.t = 0
        
        self.bgPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.bgPlayer.setVideoOutput(self.findChild(QObject, 'bgVideo'))
        self.bgPlayer.stateChanged.connect(lambda : self.bgPlayer.play() if self.bgPlayer.state() == QMediaPlayer.StoppedState else None)
        
        self.relPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.relPlayer.setVideoOutput(self.findChild(QObject, 'relVideo'))
        self.relPlayer.stateChanged.connect(lambda : self.relPlayer.play() if self.relPlayer.state() == QMediaPlayer.StoppedState else None)
        
        self.bg_video = []
        self.rel_video = []

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_A:
            self.ok_button.click()
        elif key == Qt.Key.Key_S:
            self.cancel_button.click()
        elif key == Qt.Key.Key_D:
            self.toggle_button.click()
        elif key == Qt.Key.Key_Z:
            self.bgButton.click()
        elif key == Qt.Key.Key_X:
            self.relButton.click()
        elif key == Qt.Key.Key_C:
            self.record_button.click()
        
    def composite(self, colors, depths, f=None):
        colors = [colors[1], colors[0], colors[2]]
        depths = [depths[1], depths[0], depths[2]]
        if f is not None:
            for i in range(len(colors)):
                colors[i] = cv2.resize(colors[i], (0, 0), fx=f, fy=f)
            for i in range(len(depths)):
                depths[i] = cv2.resize(depths[i], (0, 0), fx=f, fy=f)
        colors = np.concatenate(colors, 1)
        depths = np.concatenate([cv2.cvtColor(np.clip(depth / 2000. * 255., 0, 255.).astype(np.uint8), cv2.COLOR_GRAY2BGR) for depth in depths], 1)
        return np.concatenate([colors, depths], 0)
        
    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusbar.showMessage("Error: " + self.mediaPlayer.errorString())

    def closeEvent(self, event):
        if self.th is not None:
            self.th.stop()
        event.accept()
        
    def record(self):
        if self.th.recording:
            self.th.recording = False
            self.stacked_widget.setCurrentIndex(0)
            file_name = os.path.join(tempfile.gettempdir(), 'tmp.avi')
            frames = []
            video = deepcopy(self.th.recorded_imgs)
            for i in tqdm(range(len(video))):
                frame = self.composite(video[i][0], video[i][1], f=self.scale)
                # cv2.imshow('frame', frame)
                frames.append(frame)
                # print(frame.shape)
            
            # print(len(frames))
            skvideo.io.vwrite(file_name, np.stack(frames)[..., ::-1], inputdict={'-framerate':str(15)})
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
            self.swap_buttons()
            self.play()
            self.record_button.setText('Record')
            
            # save video
            def ok():
                # name = QFileDialog.getSaveFileName(self, 'Save File', ".", "Video Files (*.mp4)")[0]
                # dialog = QFileDialog(self)
                # name = dialog.getExistingDirectory(self, 'Select a Directory')
                cur_list=os.listdir(save_root)
                max_num=-1
                for j in range(len(cur_list)):
                    idx=int(cur_list[j].split('_')[1])
                    if idx > max_num:
                        max_num=idx
                name = save_root + "video_{}/".format(max_num+1)
                print(name)
                if name is not None and len(name) > 0:
                    # create directory
                    os.makedirs(name)
                    [Path(name, f'cam{i}').mkdir(exist_ok=True) for i in range(3)]
                    [Path(name, f'cam{i}', 'rgb').mkdir(exist_ok=True) for i in range(3)]
                    [Path(name, f'cam{i}', 'depth').mkdir(exist_ok=True) for i in range(3)]
                    # os.makedirs(os.path.join(name,"aux"),exist_ok=True)
                    Path(name, 'aux1').mkdir(exist_ok=True)
                    
                    final_colors = [[] for _ in range(3)]
                    final_depths = [[] for _ in range(3)]
                    for colors, depths in video:
                        for i in range(3):
                            final_colors[i].append(colors[i])
                            final_depths[i].append(depths[i])
                    
                    for i in range(3):
                        # skvideo.io.vwrite(os.path.join(name, f'color{i}.avi'), np.stack(final_colors[i])[..., ::-1])
                        # skvideo.io.vwrite(os.path.join(name, f'depth{i}.mp4'), np.stack(final_depths[i]))
                        for j in tqdm(range(len(final_colors[i]))):
                            self.statusbar.showMessage('writing frame {} for camera {}'.format(j, i))
                            cv2.imwrite(str(Path(name, f'cam{i}', 'rgb', 'rgb{:04d}.png'.format(j))), final_colors[i][j])
                            cv2.imwrite(str(Path(name, f'cam{i}', 'depth', 'depth{:04d}.png'.format(j))), final_depths[i][j])
                        # np.save(os.path.join(name, f'rgb{i}.npy'), np.stack(final_colors[i])[..., ::-1])
                        # np.save(os.path.join(name, f'depth{i}.npy'), np.stack(final_depths[i]))
                    
                    # save relative pose
                    # np.save(Path(name, 'aux', 'rel.npy'), self.rel)
                    intrinsics = deepcopy(self.intrinsics)
                    for i in range(3):
                        intrinsics[i][:2, :3] /= self.scale
                    np.save(Path(name, 'intrinsics.npy'), intrinsics)
                    np.save(Path(name, 'extrinsics.npy'), self.extrinsics)
                    # save bg image
                    for j in tqdm(range(len(self.bg_video))):
                        cv2.imwrite(str(Path(name, 'aux1', 'bg{:04d}.png'.format(j))), self.bg_video[j])
                    
                    for j in tqdm(range(len(self.rel_video))):
                        cv2.imwrite(str(Path(name, 'aux1', 'rel{:04d}.png'.format(j))), self.rel_video[j])

                    self.statusbar.showMessage(f'Video saved to {name}')
                    self.ok_button.clicked.disconnect()
                    self.cancel_button.clicked.disconnect()
                    self.swap_buttons()
                    self.stacked_widget.setCurrentIndex(1)
                    self.th.reset()
                    os.remove(os.path.join(tempfile.gettempdir(), 'tmp.avi'))
                    self.id1 = self.id2 = None
                
            def cancel():
                self.ok_button.clicked.disconnect()
                self.cancel_button.clicked.disconnect()
                self.swap_buttons()
                self.stacked_widget.setCurrentIndex(1)
                self.th.recorded_imgs = []
                os.remove(os.path.join(tempfile.gettempdir(), 'tmp.avi'))
                self.id1 = self.id2 = None
                
            self.ok_button.clicked.connect(ok)
            self.cancel_button.clicked.connect(cancel)
            
        else:
            self.th.recording = True
            self.record_button.setText('Stop')
        
    def swap_buttons(self):
        enabled = self.confirm_buttons.itemAt(0).widget().isEnabled()
        for i in range(self.confirm_buttons.count()):
            self.confirm_buttons.itemAt(i).widget().setEnabled(not enabled)
        for i in range(self.common_buttons.count()):
            self.common_buttons.itemAt(i).widget().setEnabled(enabled)
    
    def draw_video(self):
        if self.get_frame is not None:
            img = self.get_frame()
            self.video_label.setPixmap(QPixmap.fromImage(img))
            
    def toggle_cam(self):
        self.cam_open = not self.cam_open
        if self.cam_open:
            self.th = VideoThread()
            self.th.img_signal.connect(self.update_image)
            self.th.bg_signal.connect(self.set_bg)
            self.th.rel_signal.connect(self.set_rel)
            self.th.start()
            self.toggle_button.setText('Close Cam')
        else:
            self.th.img_signal.disconnect()
            self.th.bg_signal.disconnect()
            self.th.rel_signal.disconnect()
            self.th.stop()
            self.get_frame = None
            self.toggle_button.setText('Open Cam')
            
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    @pyqtSlot(np.ndarray)
    def set_bg(self, img):
        if self.th.recording:
            self.th.recording = False
            file_name = os.path.join(tempfile.gettempdir(), 'bg.avi')
            frames = []
            video = self.th.recorded_imgs
            for i in tqdm(range(len(video))):
                frames.append(cv2.resize(video[i][0][0], (0, 0), fx=self.scale, fy=self.scale))
            
            skvideo.io.vwrite(file_name, np.stack(frames)[..., ::-1], inputdict={'-framerate':str(15)})
            
            # b = BytesIO(open(os.path.join(tempfile.gettempdir(), 'tmp.avi'), 'rb').read())
            # self.bg_buf = QtCore.QBuffer()
            # self.bg_buf.setData(b.getvalue())
            # self.bg_buf.open(QtCore.QIODevice.ReadOnly)
            
            
            # self.bgPlayer.setMedia(QMediaContent(), self.bg_buf)
            self.bgPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
            self.bgPlayer.play()
            self.bgButton.setText('Background')
            self.bg_video = [img[0][0].copy() for img in self.th.recorded_imgs]
            # os.remove(os.path.join(tempfile.gettempdir(), 'tmp.avi'))
            while len(self.th.recorded_imgs) > 0:
                self.th.recorded_imgs = []
            
            self.db = []
            for img in tqdm(frames[::5]):
                self.id1 = None
                res = detect(self.intrinsics[0], img, marker_length=Config.MARKER_SIZE)
                if res is not None:
                    cam_pose = np.linalg.inv(list(res.values())[0]['extrinsic'])  # in id1 frame
                    if self.id1 is None:
                        self.id1 = list(res.keys())[0]
                    self.db.append({
                        'cam_pose': cam_pose,
                        'img': img,
                        'corner': list(res.values())[0]['corner']
                    })
                    # for pt in list(res.values())[0]['corner'].astype(int):
                    #     cv2.circle(img, (pt[0], pt[1]), 2, (0, 0, 255), -1)
                    # cv2.imshow('img', img)
                    # cv2.waitKey()
        else:
            # clear media
            self.bgPlayer.stop()
            self.bgPlayer.setMedia(QMediaContent())
            self.th.recording = True
            self.bgButton.setText('Stop')
        # img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale)
        # def ok():
        #     nonlocal img
        #     # colors, depths = self.curr_frame
        #     # img = colors[0]
        #     # self.db = []
        #     # self.id1 = None
        #     # res = detect(self.intrinsics[0], img, marker_length=marker)
        #     # if res is not None:
        #     #     cam_pose = np.linalg.inv(list(res.values())[0]['extrinsic'])  # in id1 frame
        #     #     self.id1 = list(res.keys())[0]
        #     #     self.db.append({
        #     #         'cam_pose': cam_pose,
        #     #         'img': img,
        #     #         'corner': list(res.values())[0]['corner']
        #     #     })
        #     # else:
        #     #     print('fail to detect corners')
    
    @pyqtSlot(np.ndarray)
    def set_rel(self, img):
        if self.th.recording:
            self.th.recording = False
            file_name = os.path.join(tempfile.gettempdir(), 'rel.avi')
            frames = []
            video = self.th.recorded_imgs
            poses = []
            for i in tqdm(range(len(video))):
                frames.append(cv2.resize(video[i][0][0], (0, 0), fx=self.scale, fy=self.scale))
            
            skvideo.io.vwrite(file_name, np.stack(frames)[..., ::-1], inputdict={'-framerate':str(15)})
            
            self.rel_video = [img[0][0].copy() for img in self.th.recorded_imgs]
            # b = BytesIO(open(os.path.join(tempfile.gettempdir(), 'tmp.avi'), 'rb').read())
            # self.bg_buf = QtCore.QBuffer()
            # self.bg_buf.setData(b.getvalue())
            # self.bg_buf.open(QtCore.QIODevice.ReadOnly)
            
            for img in tqdm(frames[::5]):
                self.id2 = None
                res = detect(self.intrinsics[0], img, marker_length=Config.MARKER_SIZE)
                pose1, pose2 = None, None
                if len(res) == 2:
                    for marker_id in res:
                        if marker_id == self.id1:
                            pose1 = res[self.id1]['extrinsic']
                        else:
                            self.id2 = marker_id
                            pose2 = res[self.id2]['extrinsic']
                    poses.append(np.linalg.inv(pose1) @ pose2)  # 2 to 1
            self.rel = avg_poses(poses)

            # self.bgPlayer.setMedia(QMediaContent(), self.bg_buf)
            self.relPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
            self.relPlayer.play()
            self.relButton.setText('Relative Pose')
            # os.remove(os.path.join(tempfile.gettempdir(), 'tmp.avi'))
            while len(self.th.recorded_imgs) > 0:
                self.th.recorded_imgs = []
            
        else:
            # clear media
            self.relPlayer.stop()
            self.relPlayer.setMedia(QMediaContent())
            self.th.recording = True
            self.relButton.setText('Stop')
            # colors, depths = self.curr_frame
            # img = colors[0]
            # self.id2 = None
            # res = detect(self.intrinsics[0], img, marker_length=marker)
            # pose1, pose2 = None, None
            # if len(res) == 2:
            #     for marker_id in res:
            #         if marker_id == self.id1:
            #             pose1 = res[self.id1]['extrinsic']
            #         else:
            #             self.id2 = marker_id
            #             pose2 = res[self.id2]['extrinsic']
            #     self.rel = np.linalg.inv(pose1) @ pose2  # 2 to 1
            # else:
            #     print('fail to detect two markers')
    
    @pyqtSlot(list, list)
    def update_image(self, colors, depths):
        if time() - self.t < 1 / self.fps + 0.01:
            return
        
        self.t = time()
        f = self.scale
        for i in range(len(colors)):
            colors[i] = cv2.resize(colors[i], (0, 0), fx=f, fy=f)
        for i in range(len(depths)):
            depths[i] = cv2.resize(depths[i], (0, 0), fx=f, fy=f)
        self.curr_frame = colors, depths
        
        if hasattr(self, 'id2'):
            # print('here')
            res = detect(self.intrinsics[0], colors[0], marker_length=Config.MARKER_SIZE)
            if res is not None and self.id2 in res:
                cam_pose0 = self.rel @ np.linalg.inv(res[self.id2]['extrinsic'])
                pose = np.linalg.inv(cam_pose0)
                
                for i in range(3):
                    ex = self.extrinsics[i] @ pose  # id1 projection in each frame

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
                    proj = (self.intrinsics[i] @ proj[:, :3].T).T
                    proj = proj[:, :2] / proj[:, 2:3]

                    # for pt in proj.astype(int):
                    #     cv2.circle(colors[i], (pt[0], pt[1]), 2, (0, 0, 255), -1)

                    # for pt in self.db[0]['corner'].astype(int):
                    #     cv2.circle(colors[i], (pt[0], pt[1]), 2, (255, 0, 0), -1)
                    
                    # # print(self.intrinsics[0])
                    # cv2.imshow('img', colors[0])
                    # cv2.waitKey()
                    
                    # dist = np.array([np.linalg.norm(np.linalg.inv(ex)[:3, -1] - p['cam_pose'][:3, -1]) for p in self.db])
                    dist = np.array([np.arccos((np.trace(np.linalg.inv(ex)[:3, :3].T @ p['cam_pose'][:3, :3]) - 1.) / 2) for p in self.db])
                    best_idx = np.argmin(dist)

                    im_src = self.db[best_idx]['img']
                    im_dst = colors[i]
                    h, status = cv2.findHomography(self.db[best_idx]['corner'], proj)
                    # Warp source image to destination based on homography
                    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
                    
                    mask = np.zeros((im_dst.shape[0], im_dst.shape[1]), np.uint8)
                    ratio = 1.7 / 2.
                    a4corners = np.array(
                        [
                            [-marker_size * ratio, marker_size * ratio, 0, 1],
                            [marker_size * ratio, marker_size * ratio, 0, 1],
                            [marker_size * ratio, -marker_size * ratio, 0, 1],
                            [-marker_size * ratio, -marker_size * ratio, 0, 1],
                        ]
                    )
                    
                    proj = (self.extrinsics[i] @ res[self.id2]['extrinsic'] @ a4corners.T).T
                    proj = (self.intrinsics[i] @ proj[:, :3].T).T
                    proj = proj[:, :2] / proj[:, 2:3]
                    proj = proj.astype(int)
                    cv2.fillConvexPoly(mask, proj, 255)
                    br = cv2.boundingRect(mask) # bounding rect (x,y,width,height)
                    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
                    # im_inpaint[mask > 0] = im_out[mask > 0]
                    colors[i] = cv2.seamlessClone(im_out, im_dst, mask, centerOfBR, cv2.NORMAL_CLONE)
        img = cv2.cvtColor(self.composite(colors, depths), cv2.COLOR_BGR2RGB)
        self.get_frame = lambda: QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).scaledToWidth(self.video_label.width())

        
if __name__ == '__main__' :
    app = QApplication([])
    app.setStyleSheet("QLabel{font-size: 14pt;}")
    widget = Window()
    widget.show()
    sys.exit(app.exec_())
    # im_src = cv2.imread('book2.jpg')
    # # Four corners of the book in source image
    # pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])
    # # Read destination image.
    # im_dst = cv2.imread('book1.jpg')
    # # Four corners of the book in destination image.
    # pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]])
    # # Calculate Homography
    # h, status = cv2.findHomography(pts_src, pts_dst)
    # # Warp source image to destination based on homography
    # im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    # cv2.imshow("Source Image", im_src)
    # cv2.imshow("Destination Image", im_dst)
    # cv2.imshow("Warped Source Image", im_out)
    # cv2.waitKey(0)
