# This Python file uses the following encoding: utf-8
from copy import deepcopy
import json
from multiprocessing import Process
from multiprocessing.connection import Pipe
import os

from pathlib import Path
import pickle
import sys
from time import time
from typing import List
import cv2
try:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
except:
    pass
import trimesh
# import win32gui
import numpy as np
from kornia.morphology import dilation

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QSizePolicy, QLabel, \
    QTreeWidgetItem, QFileDialog, QDialog, QDialogButtonBox, QVBoxLayout, QScrollArea, QInputDialog
from PyQt5.QtCore import QFile, QThread, QObject, QTimer, Qt, QEvent, QPoint
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QWindow, QPainter, QImage
from PyQt5 import uic
import open3d as o3d
import pyqtgraph as pg
from utils.align import PnPSolver, get_extrinsic
from scipy.spatial.transform import Rotation as R
import seaborn as sns
from scipy.stats import special_ortho_group
import torch
import nvdiffrast.torch as dr
from utils.io import load_anno, parse_arti_obj_hier, save_anno
from utils.miscellaneous import next_available_start
from utils.obj import ObjectNode, load_obj
from utils.pose import bcot_track, estimate_rel_pose
from utils.seg import SegScene
import utils.transform as transform
from utils.render import AnnoScene, compose_img, render_silhouette
import xmltodict
from utils.config import Config
from tqdm import tqdm
from XMem.track import get_tracker, resize_mask
from PIL import Image
from subprocess import Popen, PIPE
import cv2.aruco as aruco
from functools import partial
from utils.ui import ImageSelector, MessageDialog, create_image_sel_dialog
import torch.nn.functional as F
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
pg.setConfigOption('imageAxisOrder', 'row-major')
import kornia.feature as KF
# pg.setConfigOption('leftButtonPan', False)  # if False, then dragging the left mouse button draws a rectangle


# Handle high resolution displays:
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

from time import perf_counter


class catchtime:
    def __init__(self, name) -> None:
        self.name = name
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'{self.name}: {self.time:.3f} seconds'
        print(self.readout)
        
from PyQt5.QtCore import pyqtSignal, pyqtSlot
# class UpdateThread(QThread):
#     finished = pyqtSignal()
    
#     def run(self, *args, **kwargs):
#         print('running')
#         self.parent.blocked = True
#         print('updating')
#         self.parent.update_anno(*args, **kwargs)
#         self.finished.emit()
        
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.load_ui()
        
        self.matcher = KF.LoFTR('indoor').eval().cuda()
        
        self.msg = QLabel()
        self.statusBar().addWidget(self.msg)
        self.time1 = time()
        self.time2 = time()
        
        self.pnp_palette = sns.color_palette("bright", 20)
        
        self.adding_model = False
        self.pnp_frame = None
        self.button_add = self.findChild(QObject, 'addmodel')
        self.button_add.clicked.connect(self.add_model)
        
        self.button_cancel = self.findChild(QObject, 'canceladd')
        self.button_cancel.clicked.connect(self.cancel_add)
        
        self.button_prev = self.findChild(QObject, 'prev')
        self.button_prev.clicked.connect(self.prev_frame)
        
        self.button_next = self.findChild(QObject, 'next')
        self.button_next.clicked.connect(self.next_frame)
        
        self.button_prev10 = self.findChild(QObject, 'prev10')
        self.button_prev10.clicked.connect(self.prev_10frame)
        
        self.button_next10 = self.findChild(QObject, 'next10')
        self.button_next10.clicked.connect(self.next_10frame)
        
        self.line_edit = self.findChild(QObject, 'lineEdit')
        self.button_jmp = self.findChild(QObject, 'jumpButton')
        self.button_jmp.clicked.connect(lambda _: self.jump_frame(self.frame_candidates.index('{:04d}'.format(int(self.line_edit.text())))))
        
        self.button_genseg = self.findChild(QObject, 'genSeg')
        self.button_genseg.clicked.connect(self.generate_seg)
        
        self.label_scene = self.findChild(QLabel, 'sceneid')
        self.label_frame = self.findChild(QLabel, 'frameid')
        
        self.action_open = self.findChild(QObject, 'actionOpen')
        self.action_open.triggered.connect(self.openScene)
        
        self.action_extrap = self.findChild(QObject, 'actionExtrinsic_extrapolation')
        self.action_extrap.triggered.connect(self.extrapolate)
        
        self.action_track = self.findChild(QObject, 'actionTracking')
        self.action_track.triggered.connect(self.track)
        
        self.action_seg = self.findChild(QObject, 'actionSegmentation_generation')
        self.action_seg.triggered.connect(self.generate_seg_all)
        
        self.action_bundletrack = self.findChild(QObject, 'actionBundle_track')
        self.action_bundletrack.triggered.connect(self.bundle_track)
        
        self.tab = self.findChild(QObject, 'tabWidget')
        self.tab.currentChanged.connect(self.tabChanged)
        
        self.obj_tree = self.findChild(QObject, 'objView')
        self.obj_tree.currentItemChanged.connect(self.dbSelectionChanged)
        
        self.anno_tree = self.findChild(QObject, 'annoView')
        self.anno_tree.currentItemChanged.connect(self.annoSelectionChanged)
        
        self.loadObjDatabase()
        
        self.pnp_solver = PnPSolver()
        # self.obj_tree.expandAll()
        
        self.setup_viewer3d()
        
        self.grid = self.findChild(QObject, 'graphicsView')
        self.grid.ci.setContentsMargins(0, 0, 0, 0)
        
        self.curr_selection = None
        self.cand_selection = (None, None)
        
        self.viewpoint = None
        
        # self.installEventFilter(self)
        # FIXME: load me from file
        self.cfg = {
            'trans_interval': 1e-3,
            'rot_interval': 0.5,  # in degree
        }
        self.annotations = {}
        self.obj = None
        self.canon_mesh = None
        self.anno_scene = None
        self.seg_scenes = {i: self.findChild(SegScene, f'segScene{i}') for i in Config.CAMS_TO_ANNO}
        # self.loadScene(Path('data/scenes/test'))
        # self.reset()
        # self.loadFrame(self.frame_candidates[0])
        self.showMaximized()
        self.loaded = []
        self.update_th = QThread(parent=self)
        def set_blocked():
            print('finished')
            self.blocked = False
        self.update_th.finished.connect(set_blocked)
    
    def jump_frame(self, target_idx):
        target_idx = min(max(target_idx, 0), len(self.frame_candidates) - 1)
        if target_idx == self.frame_candidates.index(self.curr_frame_idx):
            return
        self.reset()
        self.loadFrame(self.frame_candidates[target_idx], update_all=target_idx not in self.loaded)
        
    def prev_frame(self):
        self.jump_frame(self.frame_candidates.index(self.curr_frame_idx) - 1)
        
    def prev_10frame(self):
        self.jump_frame(self.frame_candidates.index(self.curr_frame_idx) - 10)
    
    def next_frame(self):
        self.jump_frame(self.frame_candidates.index(self.curr_frame_idx) + 1)
        
    def next_10frame(self):
        self.jump_frame(self.frame_candidates.index(self.curr_frame_idx) + 10)
        
    def reset(self):
        self.annotations = {}
        
        self.canon_mesh = None
        self.reset_obj_mesh(None)
        
        self.curr_selection = None
        self.cand_selection = (None, None)
        
        self.pnp_solver.clear()
        self.adding_model = False
        self.pnp_frame = None
        
        self.anno_scene.clear()
        
        self.scene_vis.clear_geometries()
        if len(self.img_frames) > 0:
            try:
                self.img_frames[0].scene().sigMouseClicked.disconnect()
                self.img_frames[0].scene().sigMouseMoved.disconnect()
            except:
                pass
        
    def openScene(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_path is None or len(folder_path) == 0:
            return
        self.loadScene(Path(folder_path))
        self.reset()
        self.loadFrame(self.frame_candidates[0])
        
        
    def loadScene(self, path : Path):
        if not os.listdir(str(path)):
            return
        self.scene_root = path
        self.scene_id  = path.stem
        self.anno_scene = AnnoScene()
        self.cameras = {}
        self.cam_infos = {}
        self.frame_candidates = []
        self.img_frames = {}
        self.grid.clear()

        intrin_list = np.load(path / 'intrinsics.npy')
        extrin_list = np.load(path / 'extrinsics_refined.npy')
        for i in Config.CAMS_TO_ANNO:
            cam_path = path / "cam{}".format(i)
            if cam_path.is_dir():
                # cam_info = json.load(open(cam_path / 'cam_info.json'))
                # intrinsics = np.array(cam_info['intrinsics'])
                # extrinsics = np.array(cam_info['extrinsics'])
                cam_info = {}
                intrinsics = intrin_list[i]
                extrinsics = extrin_list[i]
                cam_info['intrinsics'] = intrinsics
                cam_info['extrinsics'] = extrinsics
                cam_info['depth_scale'] = Config.DEPTH_SCALE
                cam_info['width'] = Config.FRAME_WIDTH
                cam_info['height'] = Config.FRAME_HEIGHT
                flip_yz = np.eye(4)
                flip_yz[1:3, 1:3] *= -1
                extrinsics = flip_yz @ extrinsics @ flip_yz
                
                self.anno_scene.add_frame(intrinsics, extrinsics, cam_info['width'], cam_info['height'])
                self.cameras[i] = cam_path.stem
                self.cam_infos[i] = cam_info
                
                if len(self.frame_candidates) == 0:
                    self.frame_candidates = list(sorted([path.stem for path in (cam_path / 'rgb_marker').glob('*.png')]))
                    self.frame_candidates = [idx.split("rgb")[1] for idx in self.frame_candidates]

        for i in Config.CAMS_TO_ANNO:
            frame = self.grid.addViewBox(row=i // 2, col=i % 2)
            frame.setAspectLocked()
            self.img_frames[i] = frame
        
        self.label_scene.setText('Scene ID: ' + path.stem)
            
    def loadFrame(self, idx, update_all=False):
        self.curr_frame_idx = idx
        self.loaded.append(idx)
        for j, i in enumerate(Config.CAMS_TO_ANNO):
            img_data = cv2.imread(str(self.scene_root / self.cameras[i] / 'rgb_marker' / ('rgb' + idx + '.png')))[..., :3][::-1, :, ::-1]
            frame = self.img_frames[i]
            frame_img = pg.ImageItem(img_data)
            frame.addItem(frame_img)
            frame.img = frame_img
            frame.bg = img_data
            frame.rendered = img_data
            if j == 0:
                frame.scene().sigMouseClicked.connect(self.mouseClickedEvent)
                frame.scene().sigMouseMoved.connect(self.mouseMovedEvent)
                
            self.seg_scenes[i].load_img(img_data[::-1].copy())
            self.seg_scenes[i].load_mask_from(self.scene_root / self.cameras[i] / 'mask' / (idx + '.png'))
            
            depth_data = cv2.imread(str(self.scene_root / self.cameras[i] / 'depth' / ('depth' + idx + '.png')), cv2.IMREAD_ANYDEPTH) * self.cam_infos[i]['depth_scale']
            pc, idxs = transform.backproject(depth_data, self.cam_infos[i]['intrinsics'])
            extrinsics = np.array(self.cam_infos[i]['extrinsics'])
            pc = (pc - extrinsics[:3, -1]) @ extrinsics[:3, :3]  # opencv coord
            pc[:, 1:3] *= -1  # to opengl
            
            self.scene_pc = o3d.geometry.PointCloud()
            self.scene_pc.points = o3d.utility.Vector3dVector(pc)
            self.scene_pc.colors = o3d.utility.Vector3dVector(img_data[::-1][idxs[0], idxs[1]] / 255.)
            self.scene_vis.add_geometry(self.scene_pc)
        
        self.label_frame.setText('Frame ID: ' + idx)
        
        # load annotations
        load_path = self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'pose' / (self.curr_frame_idx + '.json')
        if not Path(load_path).exists():
            # try to load from cam0
            load_path = self.scene_root / 'cam0' / 'pose' / (self.curr_frame_idx + '.json')
        self.annotations = load_anno(load_path)
        for (start, _), obj in self.annotations.items():
            self.anno_scene.add(obj, start)
            for node in obj:
                self.scene_vis.add_geometry(node.mesh)
        self.update_anno(update_all=update_all)
        self.update_anno_treeview()
        if self.viewpoint is not None:
            self.scene_vis.get_view_control().convert_from_pinhole_camera_parameters(self.viewpoint)
        
    def loadObjDatabase(self):
        self.obj_tree.clear()
        
        def parse(path : Path):
            if path.is_dir() and len(list(path.glob('*.urdf'))) > 0:
                child = QTreeWidgetItem([path.stem, 'articulated', str(path)])
                root = parse_arti_obj_hier(path)
                def create_widget(node, widget):
                    for child in node.children:
                        item = QTreeWidgetItem([child.name, 'articulated part'])
                        widget.addChild(create_widget(child, item))
                    return widget
                grand_child = QTreeWidgetItem([root.name, 'articulated part'])
                child.addChild(grand_child)
                create_widget(root, grand_child)
                return child
            elif path.is_dir():
                node = QTreeWidgetItem([path.stem])
                for subpath in sorted(path.glob('*')):
                    node.addChild(parse(subpath))
                return node
            elif path.suffix == '.obj':
                return QTreeWidgetItem([path.stem, 'rigid', str(path)])
            
        categories = []
        for folder in tqdm(sorted(Path(Config.MODEL_ROOT).glob('*')), total=len(list(Path(Config.MODEL_ROOT).glob('*')))):
            node = parse(folder)
            categories.append(node)
        self.obj_tree.insertTopLevelItems(0, categories)
    
    def bundle_track(self):
        cam_idx, ok = QInputDialog.getInt(self, 'Camera Index', 'Camera Index:', min(Config.CAMS_TO_ANNO))
        if not ok:
            return
        
        obj_range, ok = QInputDialog.getText(self, 'Range', 'Range:', text=','.join([str(n) for n in range(self.curr_selection[0], self.curr_selection[1] + 1)]))
        if not ok:
            return
        
        selection = [int(n) for n in obj_range.split(',')]
        try:
            from BundleTrack.build import PyBundleTrack
        except:
            try:
                from BundleTrack.build.Release import PyBundleTrack
            except:
                MessageDialog('Warning!', 'BundleTrack import failed', self).exec()
                return
        # if self.curr_selection is None:
        #     MessageDialog('Warning!', 'Please select an object', self).exec()
        #     return
        
        if self.obj is not None and self.obj.depth > 0:
            MessageDialog('Warning!', 'Please select the root object', self).exec()
            return
        dialog = create_image_sel_dialog(self)
        code = dialog.exec()
        if code == QDialog.DialogCode.Accepted:
            selected_ids = dialog.selector.get_selected_ids()
            if len(selected_ids) == 0:
                return
            
            flip_yz = np.eye(4)
            flip_yz[1:3, 1:3] *= -1
            
            tracker = PyBundleTrack.Bundler('BundleTrack/config_nocs.yml')
            curr_idx = self.frame_candidates.index(self.curr_frame_idx)
            idx_pool = [self.frame_candidates.index(img_id) for img_id in selected_ids]
            
            # cam_idx = min(Config.CAMS_TO_ANNO)
            extrinsic = self.cam_infos[cam_idx]['extrinsics']
            intrinsic = self.cam_infos[cam_idx]['intrinsics']
            
            processor, mapper, im_transform = get_tracker()
            rgb = Image.open(str(self.scene_root / self.cameras[cam_idx] / 'rgb_marker' / str('rgb' + self.curr_frame_idx + '.png')))
            mask = self.anno_scene.get_mask(Config.CAMS_TO_ANNO.index(cam_idx))[::-1].copy()
            # mask = cv2.imread(str(self.scene_root / self.cameras[cam_idx] / 'mask' / str(self.curr_frame_idx + '.png')), cv2.IMREAD_GRAYSCALE)
            # mask = ((mask >= selection[0]) & (mask <= selection[1])).astype(np.uint8)
            mask = np.isin(mask, selection).astype(np.uint8)
            img = im_transform(rgb).cuda()
            msk, labels = mapper.convert_mask(mask)
            msk = torch.Tensor(msk).cuda()
            msk = resize_mask(msk.unsqueeze(0), 360)[0]
            processor.set_all_labels(list(mapper.remappings.values()))
            processor.step(img, msk, labels, end=False)
            
            failure_ids = []
            for i in tqdm(range(curr_idx, max(idx_pool) + 1)):
                img_id = self.frame_candidates[i]
                
                img = cv2.imread(str(self.scene_root / self.cameras[cam_idx] / 'rgb_marker' / str('rgb' + img_id + '.png')))[..., ::-1]
                img = (img / 255.).astype(np.float32)
                depth = cv2.imread(str(self.scene_root / self.cameras[cam_idx] / 'depth' / str('depth' + img_id + '.png')), cv2.IMREAD_ANYDEPTH)
                depth = (depth / 1000.).astype(np.float32)

                # mask = cv2.imread(str(self.scene_root / self.cameras[cam_idx] / 'mask' / str(img_id + '.png')), cv2.IMREAD_UNCHANGED).astype(np.uint8)
                # mask = ((mask >= self.curr_selection[0]) & (mask <= self.curr_selection[1])).astype(np.uint8) * 255
                prob = processor.step(im_transform((img * 255).astype(np.uint8)).cuda(), None, None, end=(i==max(idx_pool)))
                prob = F.interpolate(prob.unsqueeze(1), np.array(rgb).shape[:2], mode='bilinear', align_corners=False)[:,0]
                out_mask = torch.argmax(prob, dim=0)
                mask = (out_mask.detach().cpu().numpy()).astype(np.uint8) * 255
                rel_pose = flip_yz @ np.linalg.inv(extrinsic) @ tracker.track(img, depth, mask, intrinsic) @ extrinsic @ flip_yz # convert to opengl coord
                
                if i != curr_idx and i in idx_pool:
                    annotations = load_anno(self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'pose' / (img_id + '.json'), read_mesh=False)
                    for sel_id in selection:
                        node = self.find_node(sel_id)
                        obj_idx = self.get_offset(node)
                        while node.parent is not None:
                            node = node.parent
                            obj_idx = self.get_offset(node)
                        annotations[obj_idx] = deepcopy(self.annotations[obj_idx])
                        annotations[obj_idx].transform(rel_pose)
                        
                    save_anno(self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'pose' / str(img_id + '.json'), annotations)
                self.statusbar.showMessage('{}/{}'.format(i - curr_idx, len(range(curr_idx, max(idx_pool) + 1))))
            del tracker
            if min(idx_pool) < curr_idx:
                processor, mapper, im_transform = get_tracker()
                rgb = Image.open(str(self.scene_root / self.cameras[cam_idx] / 'rgb_marker' / str('rgb' + self.curr_frame_idx + '.png')))
                mask = cv2.imread(str(self.scene_root / self.cameras[cam_idx] / 'mask' / str(self.curr_frame_idx + '.png')), cv2.IMREAD_GRAYSCALE)
                # mask = ((mask >= selection[0]) & (mask <= selection[1])).astype(np.uint8)
                mask = np.isin(mask, selection).astype(np.uint8)
                img = im_transform(rgb).cuda()
                msk, labels = mapper.convert_mask(mask)
                msk = torch.Tensor(msk).cuda()
                msk = resize_mask(msk.unsqueeze(0), 360)[0]
                processor.set_all_labels(list(mapper.remappings.values()))
                processor.step(img, msk, labels, end=False)
                tracker = PyBundleTrack.Bundler()
                
                for i in tqdm(range(curr_idx, min(idx_pool) - 1, -1)):
                    img_id = self.frame_candidates[i]
                    
                    img = cv2.imread(str(self.scene_root / self.cameras[cam_idx] / 'rgb_marker' / str('rgb' + img_id + '.png')))[..., ::-1]
                    img = (img / 255.).astype(np.float32)
                    depth = cv2.imread(str(self.scene_root / self.cameras[cam_idx] / 'depth' / str('depth' + img_id + '.png')), cv2.IMREAD_ANYDEPTH)
                    depth = (depth / 1000.).astype(np.float32)

                    # mask = cv2.imread(str(self.scene_root / self.cameras[cam_idx] / 'mask' / str(img_id + '.png')), cv2.IMREAD_UNCHANGED).astype(np.uint8)
                    # mask = ((mask >= self.curr_selection[0]) & (mask <= self.curr_selection[1])).astype(np.uint8) * 255
                    prob = processor.step(im_transform((img * 255).astype(np.uint8)).cuda(), None, None, end=(i==min(idx_pool)))
                    prob = F.interpolate(prob.unsqueeze(1), np.array(rgb).shape[:2], mode='bilinear', align_corners=False)[:,0]
                    out_mask = torch.argmax(prob, dim=0)
                    mask = (out_mask.detach().cpu().numpy()).astype(np.uint8) * 255
                    rel_pose = flip_yz @ np.linalg.inv(extrinsic) @ tracker.track(img, depth, mask, intrinsic) @ extrinsic @ flip_yz # convert to opengl coord
                    
                    if i != curr_idx and i in idx_pool:
                        annotations = load_anno(self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'pose' / (img_id + '.json'), read_mesh=False)
                        # annotations[self.curr_selection] = deepcopy(self.annotations[self.curr_selection])
                        # annotations[self.curr_selection].transform(rel_pose)
                        for sel_id in selection:
                            node = self.find_node(sel_id)
                            obj_idx = self.get_offset(node)
                            while node.parent is not None:
                                node = node.parent
                                obj_idx = self.get_offset(node)
                            annotations[obj_idx] = deepcopy(self.annotations[obj_idx])
                            annotations[obj_idx].transform(rel_pose)
                        
                        save_anno(self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'pose' / str(img_id + '.json'), annotations)
                    self.statusbar.showMessage('{}/{}'.format(curr_idx - i, len(range(curr_idx, min(idx_pool) - 1, -1))))
            
            if len(failure_ids) > 0:
                MessageDialog('Warning!', 'Some frames have not been propagated: {}'.format(','.join(failure_ids)), self).exec()
            self.statusbar.showMessage('Label successfully propagated!')

    def extrapolate(self):
        dialog = create_image_sel_dialog(self)
        code = dialog.exec()
        
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
        arucoParams = aruco.DetectorParameters_create()
        arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        arucoParams.cornerRefinementWinSize = 5
        Ks = {i: np.array(self.cam_infos[i]['intrinsics']) for i in Config.CAMS_TO_ANNO}
        extrinsics = {i: np.array(self.cam_infos[i]['extrinsics']) for i in Config.CAMS_TO_ANNO}
        extrinsics_qt = {}
        for i in Config.CAMS_TO_ANNO:
            ex = extrinsics[i]
            q = R.from_matrix(ex[:3, :3]).as_quat()[[3, 0, 1, 2]]
            t = ex[:3, -1]
            extrinsics_qt[i] = np.concatenate([q, t])
        obj_pts = np.array([[-Config.MARKER_SIZE / 2000., Config.MARKER_SIZE / 2000., 0],
                                   [Config.MARKER_SIZE / 2000., Config.MARKER_SIZE / 2000., 0],
                                   [Config.MARKER_SIZE / 2000., -Config.MARKER_SIZE / 2000., 0],
                                   [-Config.MARKER_SIZE / 2000., -Config.MARKER_SIZE / 2000., 0]])
            
        def estimate_pose_multi(frame_idx):
            for i in Config.CAMS_TO_ANNO:
                pose = get_extrinsic(Ks[i], self.scene_root / self.cameras[i] / 'rgb_marker' / str('rgb' + frame_idx + '.png'), marker_length=Config.MARKER_SIZE)
                if pose is None:
                    continue
                pose = np.linalg.inv(extrinsics[i]) @ pose
                pose = np.concatenate([R.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]], pose[:3, -1]])
                break
            
            if pose is None:
                return pose
            
            try:
                import pyceres
                prob = pyceres.Problem()
                loss = pyceres.TrivialLoss()
                cameras = {i: np.array([Ks[i][[0, 1, 0, 1], [0, 1, 2, 2]]]) for i in Config.CAMS_TO_ANNO}
                for i in Config.CAMS_TO_ANNO:
                    img_gray = cv2.cvtColor(cv2.imread(str(self.scene_root / self.cameras[i] / 'rgb_marker' / str('rgb' + frame_idx + '.png'))), cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
                    if ids is None:
                        continue
                    corners = corners[0][0]
                    projs_homo = np.concatenate([corners, np.ones((*corners.shape[:-1], 1))], -1)
                    
                    ex = extrinsics_qt[i]
                    for proj, obj_pt in zip(projs_homo, obj_pts):
                        cost = pyceres.factors.BundleAdjustmentRigCost(1, proj[:2], ex[:4], ex[4:])  
                        prob.add_residual_block(cost, loss, [pose[:4], pose[4:], obj_pt, cameras[i]])
                        prob.set_parameter_block_constant(obj_pt)
                    prob.set_manifold(pose[:4], pyceres.QuaternionManifold())
                    prob.set_parameter_block_constant(cameras[i])
                
                # print(prob.num_parameter_blocks(), prob.num_parameters(), prob.num_residual_blocks(), prob.num_residuals())
                options = pyceres.SolverOptions()
                options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
                options.minimizer_progress_to_stdout = False
                options.num_threads = -1
                summary = pyceres.SolverSummary()
                pyceres.solve(options, prob, summary)
                # print(summary.BriefReport())
            except:
                pass
            
            pose = np.concatenate([np.concatenate([R.from_quat(pose[[1, 2, 3, 0]]).as_matrix(), pose[4:][:, None]], -1), 
                                   np.array([[0, 0, 0, 1]])], 0) 
            return pose
            
            
        if code == QDialog.DialogCode.Accepted:
            # Only get the camera0's intrinsics and data
            # instrinsics = np.array(self.cam_infos[0]['intrinsics'])
            # print(self.scene_root / self.cameras[0] / 'rgb_marker' / str('rgb' + self.curr_frame_idx + '.png'))
            
            # img_ref = cv2.imread(str(self.scene_root / self.cameras[0] / 'rgb_marker' / str('rgb' + self.curr_frame_idx + '.png')))
            # # img_ref = cv2.resize(cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5)
            # img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
            # depth_ref = cv2.imread(str(self.scene_root / self.cameras[0] / 'depth' / str('depth' + self.curr_frame_idx + '.png')), cv2.IMREAD_ANYDEPTH) / 1000.
            # intrinsic0_inv = np.linalg.inv(np.array(self.cam_infos[0]['intrinsics']))
            
            curr_pose = estimate_pose_multi(self.curr_frame_idx)
            if curr_pose is None:
                MessageDialog('Warning!', 'Marker detection failed on current frame, please try another frame', self).exec()
                return
            
            selected_ids = dialog.selector.get_selected_ids()
            if len(selected_ids) == 0:
                return
            
            failure_ids = []
            cnt = 0
            for img_id in tqdm(selected_ids):
                pose_est = estimate_pose_multi(img_id)
                # TODO: try another two view if failed
                if pose_est is None:
                    failure_ids.append(img_id)
                    continue
                
                # img_tgt = cv2.imread(str(self.scene_root / self.cameras[0] / 'rgb_marker' / str('rgb' + img_id + '.png')))
                # # img_tgt = cv2.resize(cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5)
                # img_tgt = cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY)
                # depth_tgt = cv2.imread(str(self.scene_root / self.cameras[0] / 'depth' / str('depth' + img_id + '.png')), cv2.IMREAD_ANYDEPTH) / 1000.
                
                # with torch.no_grad():
                #     input_dict = {
                #         "image0": torch.from_numpy(img_ref)[None, None].float().cuda() / 255.,  # LofTR works on grayscale images only
                #         "image1": torch.from_numpy(img_tgt)[None, None].float().cuda() / 255.,
                #     }
                #     res = self.matcher(input_dict)
                #     kps_ref = res['keypoints0'].int().cpu().numpy()
                #     kps_tgt = res['keypoints1'].int().cpu().numpy()
                    
                #     mask = (depth_ref[kps_ref[:, 1], kps_ref[:, 0]] > 0) & (depth_ref[kps_ref[:, 1], kps_ref[:, 0]] < 2) & \
                #         (depth_tgt[kps_tgt[:, 1], kps_tgt[:, 0]] > 0) & (depth_tgt[kps_tgt[:, 1], kps_tgt[:, 0]] < 2)
                #     kps_ref = kps_ref[mask]
                #     kps_tgt = kps_tgt[mask]
                    
                #     pts_ref = (intrinsic0_inv @ np.concatenate([kps_ref, np.ones((kps_ref.shape[0], 1))], -1).T).T * depth_ref[kps_ref[:, 1], kps_ref[:, 0]][..., None]
                #     pts_tgt = (intrinsic0_inv @ np.concatenate([kps_tgt, np.ones((kps_tgt.shape[0], 1))], -1).T).T * depth_tgt[kps_tgt[:, 1], kps_tgt[:, 0]][..., None]

                #     _, rel_pose, _ = cv2.estimateAffine3D(pts_ref, pts_tgt)
                #     rel_pose = np.concatenate([rel_pose, np.array([[0, 0, 0, 1.]])])
                    
                rel_pose = pose_est @ np.linalg.inv(curr_pose)
                flip_yz = np.eye(4)
                flip_yz[1:3, 1:3] *= -1
                rel_pose = flip_yz @ rel_pose @ flip_yz
                
                if self.curr_selection is None:
                    annotations = deepcopy(self.annotations)
                    for key, obj in annotations.items():
                        obj.transform(rel_pose)
                else:
                    annotations = load_anno(self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'pose' / str(img_id + '.json'), read_mesh=False)
                    for key, obj in self.annotations.items():
                        if key == self.curr_selection:
                            annotations[key] = deepcopy(obj)
                            annotations[key].transform(rel_pose)
    
                # print(rel_pose)
                # for key, obj in enumerate(annotations):
                #     if (self.curr_selection is None) or ((self.curr_selection is not None) and (key == self.curr_selection)):
                #         obj.transform(rel_pose)
                    
                save_anno(self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'pose' / str(img_id + '.json'), annotations)
                cnt += 1
                self.statusbar.showMessage('{}/{}'.format(cnt, len(selected_ids)))
            
            if len(failure_ids) > 0:
                MessageDialog('Warning!', 'Some frames have not been propagated: {}'.format(','.join(failure_ids)), self).exec()
            self.statusbar.showMessage('Label successfully propagated!')

        return
    
    def track(self):
        try:
            from tracking.build.tracking import RBOTTracker
        except:
            try:
                from tracking.build.Release.tracking import RBOTTracker
            except:
                MessageDialog('Warning!', 'RBOTTracker not found', self).exec()
                return
        if self.curr_selection is None:
            MessageDialog('Warning!', 'Please select an object', self).exec()
            return
        
        if self.obj is not None and self.obj.depth > 0:
            MessageDialog('Warning!', 'Please select the root object', self).exec()
            return
        dialog = create_image_sel_dialog(self)
        code = dialog.exec()
        if code == QDialog.DialogCode.Accepted:
            selected_ids = dialog.selector.get_selected_ids()
            if len(selected_ids) == 0:
                return
            
            flip_yz = np.eye(4)
            flip_yz[1:3, 1:3] *= -1
    
            intrinsics = []
            extrinsics = []
            for i in Config.CAMS_TO_ANNO:
                cam_info = self.cam_infos[i]
                intrinsic = np.array(cam_info['intrinsics'])
                intrinsic[0, 2] = Config.FRAME_WIDTH - intrinsic[0, 2]
                extrinsic = np.array(cam_info['extrinsics'])
                
                intrinsics.append(intrinsic.astype(np.float32))
                extrinsics.append((extrinsic @ flip_yz).astype(np.float32))
                
            parent_conn, child_conn = Pipe()
            p = Process(target=bcot_track, args=(child_conn,))
            p.start()
            parent_conn.send((intrinsics, extrinsics, Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
            # tracker = RBOTTracker(intrinsics, extrinsics, 640, 480)
            
            mesh = self.canon_mesh
            # mesh = self.canon_mesh.simplify_quadric_decimation(len(self.canon_mesh.triangles) // 10)
            vertices = np.array(mesh.vertices).astype(np.float32)
            triangles = np.array(mesh.triangles).astype(np.int32)
            
            parent_conn.send(('add_model', vertices, triangles, self.obj.pose.astype(np.float32)))
            # tracker.add_model(vertices, triangles, self.obj.pose.astype(np.float32))
            
            imgs = []
            for cam_id in Config.CAMS_TO_ANNO:
                img = cv2.imread(str(self.scene_root / self.cameras[cam_id] / 'rgb_marker' / str('rgb' + self.curr_frame_idx + '.png')))[..., ::-1]
                imgs.append((img / 255.).astype(np.float32))
            
            parent_conn.send(('init', imgs))
            # tracker.init(imgs)
            
            curr_idx = self.frame_candidates.index(self.curr_frame_idx)
            idx_pool = [self.frame_candidates.index(img_id) for img_id in selected_ids]
            
            rel_pose = np.eye(4)
            for i in tqdm(range(curr_idx + 1, max(idx_pool) + 1)):
                img_id = self.frame_candidates[i]
                
                imgs = []
                for cam_id in Config.CAMS_TO_ANNO:
                    img = cv2.imread(str(self.scene_root / self.cameras[cam_id] / 'rgb_marker' / str('rgb' +img_id + '.png')))[..., ::-1]
                    imgs.append((img / 255.).astype(np.float32))
                # rel_pose = tracker.track(imgs) @ rel_pose
                parent_conn.send(('track', imgs))
                res = parent_conn.recv()
                rel_pose = res @ rel_pose
                if i != curr_idx and i in idx_pool:
                    annotations = load_anno(self.scene_root / self.cameras[0] / 'pose' / (img_id + '.json'), read_mesh=False)
                    annotations[self.curr_selection] = deepcopy(self.annotations[self.curr_selection])
                    annotations[self.curr_selection].transform(np.linalg.inv(extrinsics[0]) @ rel_pose @ extrinsics[0])
                        
                    save_anno(self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'pose' / str(img_id + '.json'), annotations)
            
            parent_conn.send(('end',))
            p.join()

            if min(idx_pool) < curr_idx:
                p = Process(target=bcot_track, args=(child_conn,))
                p.start()
                parent_conn.send((intrinsics, extrinsics, Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
                parent_conn.send(('add_model', vertices, triangles, self.obj.pose.astype(np.float32)))
                # tracker = RBOTTracker(intrinsics, extrinsics, 640, 480)
                # tracker.add_model(vertices, triangles, self.obj.pose.astype(np.float32))
                imgs = []
                for cam_id in Config.CAMS_TO_ANNO:
                    img = cv2.imread(str(self.scene_root / self.cameras[cam_id] / 'rgb_marker' / str('rgb' + self.curr_frame_idx + '.png')))[..., ::-1]
                    imgs.append((img / 255.).astype(np.float32))
                
                parent_conn.send(('init', imgs))
                # tracker.init(imgs)
                rel_pose = np.eye(4)
                
                for i in tqdm(range(curr_idx - 1, min(idx_pool) - 1, -1)):
                    img_id = self.frame_candidates[i]
                    
                    imgs = []
                    for cam_id in Config.CAMS_TO_ANNO:
                        img = cv2.imread(str(self.scene_root / self.cameras[cam_id] / 'rgb_marker' / str('rgb' + img_id + '.png')))[..., ::-1]
                        imgs.append((img / 255.).astype(np.float32))
                        
                    parent_conn.send(('track', imgs))
                    res = parent_conn.recv()
                    rel_pose = res @ rel_pose
                    print(rel_pose)
                    # rel_pose = tracker.track(imgs) @ rel_pose
                    
                    if i != curr_idx and i in idx_pool:
                        annotations = load_anno(self.scene_root / self.cameras[0] / 'pose' / (img_id + '.json'), read_mesh=False)
                        annotations[self.curr_selection] = deepcopy(self.annotations[self.curr_selection])
                        annotations[self.curr_selection].transform(np.linalg.inv(extrinsics[0]) @ rel_pose @ extrinsics[0])
                            
                        save_anno(self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'pose' / str(img_id + '.json'), annotations)

                parent_conn.send(('end',))
                p.join()
    
            self.statusbar.showMessage('Label successfully propagated!')
                          
        
    def dbSelectionChanged(self, curr, prev):
        if curr.text(1) == 'rigid':
            node = load_obj(Path(curr.text(2)))
        elif 'articulated' in curr.text(1):
            if 'part' in curr.text(1):
                
                parent = curr.parent()
                while parent.text(1) != 'articulated':
                    parent = parent.parent()
                
                obj = load_obj(Path(parent.text(2)))
                node = obj.get_node_by_name(curr.text(0))
            else:
                node = load_obj(Path(curr.text(2)))
        else:
            return
        self.reset_obj_mesh(node)
        
    def annoSelectionChanged(self, curr, prev):
        if curr is None:
            return
        keys = curr.text(0).split(',')
        key = (int(keys[0]), int(keys[-1]))
        self.curr_selection = key
        self.reset_obj_mesh(self.find_node(key))
        self.update_anno(False)
        self.grid.setFocus()
        
        for i in Config.CAMS_TO_ANNO:
            if len(keys) == 1:
                self.seg_scenes[i].curr_selection = key[0]
            else:
                self.seg_scenes[i].curr_selection = 0
            self.seg_scenes[i].update()
        
    def generate_seg(self):
        for i in Config.CAMS_TO_ANNO:
            self.seg_scenes[i].load_mask(self.anno_scene.get_mask(Config.CAMS_TO_ANNO.index(i))[::-1].copy())
            self.seg_scenes[i].save_mask(self.scene_root / self.cameras[i] / 'mask' / (self.curr_frame_idx + '.png'))
            
    def generate_seg_all(self):
        intrinsics = np.load(self.scene_root  / 'intrinsics.npy')
        extrinsics = np.load(self.scene_root  / 'extrinsics_refined.npy')
        anno_scene = AnnoScene()
        for i in Config.CAMS_TO_ANNO:
            flip_yz = np.eye(4)
            flip_yz[1:3, 1:3] *= -1
            anno_scene.add_frame(intrinsics[i], flip_yz @ extrinsics[i] @ flip_yz, Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
        
        cnt = 0
        for fn in tqdm(list((self.scene_root / 'cam{}/pose'.format(min(Config.CAMS_TO_ANNO))).glob('*.json'))):
            annotations = load_anno(fn)
            for (start, _), obj in annotations.items():
                anno_scene.add(obj, start)
            anno_scene.update()
            
            for i in Config.CAMS_TO_ANNO:
                mask = anno_scene.get_mask(Config.CAMS_TO_ANNO.index(i))[::-1]
                (fn.parent.parent.parent / f'cam{i}' / 'mask').mkdir(exist_ok=True)
                cv2.imwrite(str(fn.parent.parent.parent / f'cam{i}' / 'mask' / (fn.stem + '.png')), mask)
            anno_scene.clear()
            cnt += 1
            self.statusbar.showMessage('{}/{}'.format(cnt, len(list((self.scene_root / 'cam{}/pose'.format(min(Config.CAMS_TO_ANNO))).glob('*.json')))))
        
        self.statusbar.showMessage('Segmentation successfully generated!')
    
    def reset_obj_mesh(self, node):
        if self.obj is not None:
            self.obj_vis.clear_geometries()
        if node is None:
            self.obj = None
            return
        self.obj = node
        pose = self.obj.pose
        self.obj.transform(np.linalg.inv(pose))
        self.canon_mesh = o3d.geometry.TriangleMesh(self.obj.mesh)
        self.select_mesh = o3d.geometry.TriangleMesh(self.obj.mesh)
        self.obj.transform(pose)
        self.obj_vis.add_geometry(self.select_mesh)
        self.obj_vis.update_geometry(None)
        
    def tabChanged(self, i):
        self.update_anno()
        
    def keyPressEvent(self, event):
        if time() - self.time1 < 0.1:
            return
        self.time1 = time()
        key = event.key()
        
        if event.modifiers() & Qt.ControlModifier:
            if key == Qt.Key.Key_S:
                if len(self.cameras) > 0:
                    save_anno(self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'pose' / (self.curr_frame_idx + '.json'), self.annotations)
                for i in Config.CAMS_TO_ANNO:
                    if self.seg_scenes[i].qmask is not None:
                        self.seg_scenes[i].save_mask(self.scene_root / self.cameras[i] / 'mask' / (self.curr_frame_idx + '.png'))
                self.statusbar.showMessage('Saved!', 3000)
                return
            elif key == Qt.Key.Key_Z:
                op = self.pnp_solver.undo()
                if op == self.pnp_solver.OP2D:
                    self.update_image(self.pnp_frame)
                elif op == self.pnp_solver.OP3D:
                    self.select_mesh.clear()
                    self.select_mesh += self.canon_mesh
                    pts3D = self.pnp_solver.pts3D
                    self.pnp_solver.pts3D = []
                    for pt in pts3D:
                        self.pnp_solver.pts3D.append(pt)
                        self.add_3d_point(np.array([pt[0], -pt[1], -pt[2]]))
                    self.obj_vis.update_geometry(None)
        if key == Qt.Key.Key_R:
            if self.tab.currentIndex() == 0:
                self.tab.setCurrentIndex(1)
            elif self.tab.currentIndex() == 1:
                self.tab.setCurrentIndex(0)
            return
        elif key == Qt.Key.Key_F:
            self.viewpoint = self.scene_vis.get_view_control().convert_to_pinhole_camera_parameters()
            print(self.viewpoint)
            return
                
        if key >= Qt.Key_0 and key <= Qt.Key_9:
            # Numeric key is pressed
            number = key - Qt.Key_0  # Get the numeric value
            for anno in self.annotations:
                if number in anno:
                    sel_id = (anno[0], anno[-1])
                    node = self.find_node(sel_id)
                    self.curr_selection = sel_id
                    self.reset_obj_mesh(node)
                    self.update_anno(False)
                    return
                
        # delete annotations
        if key == Qt.Key.Key_Delete:
            if self.curr_selection is not None and self.tab.currentIndex() < 2:
                self.remove_anno(self.curr_selection)
            return
        # pos change
        if key in [Qt.Key.Key_Q, Qt.Key.Key_W, Qt.Key.Key_E, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D,
                           Qt.Key.Key_Z, Qt.Key.Key_X, Qt.Key.Key_C, Qt.Key.Key_V, Qt.Key.Key_B, Qt.Key.Key_N]:
            if self.curr_selection is None:
                return
            if event.modifiers() & Qt.AltModifier:
                ctrl_scale = 10
            else:
                ctrl_scale = 1
            trans_interval = self.cfg['trans_interval'] * ctrl_scale
            rot_interval = self.cfg['rot_interval'] / 180 * np.pi * ctrl_scale
            delta = np.eye(4)
            
            pose = self.obj.pose
            trans2origin = np.eye(4)
            trans2origin[:3, -1] = -self.obj.center
            
            # print(self.obj.pose)
            self.obj.transform(trans2origin)
            if key == Qt.Key.Key_Q:
                delta[:3, 3] += pose[:3, 0] * trans_interval
            elif key == Qt.Key.Key_E:
                delta[:3, 3] -= pose[:3, 0] * trans_interval
            elif key == Qt.Key.Key_W:
                delta[:3, 3] += pose[:3, 2] * trans_interval
            elif key == Qt.Key.Key_S:
                delta[:3, 3] -= pose[:3, 2] * trans_interval
            elif key == Qt.Key.Key_D:
                delta[:3, 3] += pose[:3, 1] * trans_interval
            elif key == Qt.Key.Key_A:
                delta[:3, 3] -= pose[:3, 1] * trans_interval
            elif key == Qt.Key.Key_Z:
                delta[:3, :3] = transform.rotation_around(pose[:3, 0], rot_interval)
            elif key == Qt.Key.Key_X:
                delta[:3, :3] = transform.rotation_around(pose[:3, 0], -rot_interval)
            elif key == Qt.Key.Key_C:
                delta[:3, :3] = transform.rotation_around(pose[:3, 1], rot_interval)
            elif key == Qt.Key.Key_V:
                delta[:3, :3] = transform.rotation_around(pose[:3, 1], -rot_interval)
            elif key == Qt.Key.Key_B:
                delta[:3, :3] = transform.rotation_around(pose[:3, 2], rot_interval)
            elif key == Qt.Key.Key_N:
                delta[:3, :3] = transform.rotation_around(pose[:3, 2], -rot_interval)
            
            self.obj.transform(delta)
            trans2origin[:3, -1] *= -1
            self.obj.transform(trans2origin)
            self.update_anno()
        
        for i in Config.CAMS_TO_ANNO:
            self.seg_scenes[i].keyPressEvent(event)
        
    def keyReleaseEvent(self, event) -> None:
        for i in Config.CAMS_TO_ANNO:
            self.seg_scenes[i].keyReleasedEvent(event)
            

    def load_ui(self):
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        uic.loadUi(path, self)

    def get_hwnd(self):
        hwnd = None
        while hwnd == None:
            proc = Popen('xwininfo -tree -root', stdin=None, stdout=PIPE, stderr=None, shell=True)
            out, err = proc.communicate()
            for window in out.decode('utf-8').split('\n'):
                if 'Open3D' in window:
                    hwnd = int(window.lstrip().split(' ')[0], 16)
                    return hwnd
                
    def setup_viewer3d(self):
        self.obj_vis = o3d.visualization.VisualizerWithVertexSelection()
        self.obj_vis.create_window(width=320, height=240)
        
        self.obj_vis.register_selection_changed_callback(self.picked_3d_points)
        
        if os.name == 'nt':
            import win32gui
            window = QWindow.fromWinId(win32gui.FindWindowEx(0, 0, None, "Open3D - free view"))
        else:
            window = QWindow.fromWinId(self.get_hwnd())
        widget = QWidget.createWindowContainer(window, self)
        widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.findChild(QObject, 'verticalLayout').insertWidget(1, widget)
        
        # self.scene_pc = o3d.geometry.PointCloud()
        self.scene_vis = o3d.visualization.VisualizerWithKeyCallback()
        self.scene_vis.create_window(width=320, height=240)
        # self.scene_vis.add_geometry(self.scene_pc)
        if os.name == 'nt':
            import win32gui
            window = QWindow.fromWinId(win32gui.FindWindowEx(0, 0, None, "Open3D"))
        else:
            window = QWindow.fromWinId(self.get_hwnd())
        widget = QWidget.createWindowContainer(window, self)
        
        def key_callback(key, vis):
            event = lambda: None
            event.modifiers = lambda: False
            event.key = lambda: key
            self.keyPressEvent(event)
            
        for key in list(range(65, 91)) + list(range(48, 57)):
            self.scene_vis.register_key_callback(key, partial(key_callback, key))
        widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.findChild(QObject, 'verticalLayoutScene').insertWidget(0, widget)
        
        timer = QTimer(self)
        timer.timeout.connect(self.update_vis)
        timer.start(10)

    def add_3d_point(self, pt):
        rad = np.array(self.canon_mesh.get_axis_aligned_bounding_box().get_extent()).max() / 40
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=rad)
        mesh_sphere.translate(pt)
        mesh_sphere.paint_uniform_color(self.pnp_palette[len(self.pnp_solver.pts3D) - 1])
        self.select_mesh += mesh_sphere
    
    def picked_3d_points(self):
        if not self.adding_model or len(self.obj_vis.get_picked_points()) != 1:
            self.obj_vis.clear_picked_points()
            return
        
        picked_point = np.array(self.obj_vis.get_picked_points()[0].coord)
        self.pnp_solver.add3D(np.array([picked_point[0], -picked_point[1], -picked_point[2]]))
        self.obj_vis.clear_picked_points()
        
        # create visualized sphere
        self.add_3d_point(picked_point)
        self.obj_vis.update_geometry(None)
    
    def update_picked_points(self):
        pass
        
    def update_vis(self):
        # self.obj_vis.update_geometry(None)
        self.obj_vis.poll_events()
        self.obj_vis.update_renderer()
        if self.tab.currentIndex() == 1:
            self.scene_vis.poll_events()
            self.scene_vis.update_renderer()

    def find_node(self, idx):
        if idx is None:
            return None
        
        if isinstance(idx, tuple):
            node = self.find_node(idx[0])
            while len(node) + idx[0] - 1 < idx[1]:
                node = node.parent
            return node
        
        for idxs, obj in self.annotations.items():
            if idx >= idxs[0] and idx <= idxs[1]:
                node = obj.get_node(idx - idxs[0])
                return node
            
    def get_offset(self, node):
        for idxs, obj in self.annotations.items():
            if obj == node:
                return idxs
            
    def mouseClickedEvent(self, event):
        if (not self.adding_model):
            self.curr_selection = self.cand_selection[0]
            self.reset_obj_mesh(self.cand_selection[1])
            self.update_anno(False)
            return
        
        pos = event.pos()
        for _, frame in self.img_frames.items():
            if frame != event.currentItem:
                continue
            if self.pnp_frame is None or self.pnp_frame == frame:
                mousePoint = frame.mapFromItemToView(frame, pos)
                x_i = round(mousePoint.x())
                y_i = round(mousePoint.y())
                if x_i > 0 and x_i < frame.bg.shape[1] and y_i > 0 and y_i < frame.bg.shape[0]:
                    # self.statusbar.showMessage('({}, {})'.format(x_i, y_i))
                    self.pnp_solver.add2D(np.array([x_i, frame.bg.shape[0] - 1 - y_i]))
                    if self.pnp_frame is None:
                        self.pnp_frame = frame
                    self.update_image(frame)
                    
    def update_image(self, frame):
        if frame is None:
            return
        img = frame.rendered.copy()
        rad = min(img.shape[0], img.shape[1]) // 60
        for i, pt in enumerate(self.pnp_solver.pts2D):
            color = (np.array(self.pnp_palette[i]) * 255).astype(int)
            cv2.circle(img, (pt[0], img.shape[0] - 1 - pt[1]), rad, (int(color[0]), int(color[1]), int(color[2])), -1)
        frame.img.setImage(img)

    def mouseMovedEvent(self, pos):
        if time() - self.time1 < 0.1:
            return
        self.time1 = time()
        if self.adding_model:
            return
        
        intersect = False
        for i, frame in self.img_frames.items():
            mousePoint = frame.mapSceneToView(pos)
            x_i = round(mousePoint.x())
            y_i = round(mousePoint.y())
            if x_i > 0 and x_i < frame.bg.shape[1] and y_i > 0 and y_i < frame.bg.shape[0]:
                mask = self.anno_scene.get_mask(Config.CAMS_TO_ANNO.index(i))
                sel_id = mask[y_i, x_i]
                if sel_id > 0:
                    intersect = True
                    node = self.find_node(sel_id)
                    if self.curr_selection is not None and sel_id >= self.curr_selection[0] and sel_id <= self.curr_selection[1]:
                        while node.depth > self.obj.depth + 1:
                            node = node.parent
                        if node == self.obj:
                            self.cand_selection = (self.curr_selection, node)
                        else:
                            offset = self.obj.get_offset(node)
                            self.cand_selection = ((self.curr_selection[0] + offset, self.curr_selection[0] + offset + len(node) - 1), node)
                    else:
                        while node.parent is not None:
                            node = node.parent
                        self.cand_selection = (self.get_offset(node), node)
                    
                    self.update_anno(False)
                    break
        if not intersect:
            self.cand_selection = (None, None)
            self.update_anno(False)
    
    def cancel_add(self):
        self.pnp_solver.clear()
        self.update_image(self.pnp_frame)
        self.pnp_frame = None
        self.adding_model = False
        self.select_mesh.clear()
        self.select_mesh += self.canon_mesh
        self.obj_vis.update_geometry(None)
        self.msg.clear()
        self.button_add.setText('Add')
        
    def add_model(self):
        if self.obj is not None and self.obj.depth > 0:
            self.msg.setText("Please select the entire articulated object")
            return
        
        if self.adding_model:
            if len(self.pnp_solver.pts2D) == len(self.pnp_solver.pts3D) and len(self.pnp_solver.pts2D) >= 4:
                frame_idx = [k for k, v in self.img_frames.items() if v == self.pnp_frame][0]
                intrinsics = np.array(self.cam_infos[frame_idx]['intrinsics'])
                extrinsics = np.array(self.cam_infos[frame_idx]['extrinsics'])
                
                pose = self.pnp_solver.align(intrinsics, opengl=False)
                pose = np.linalg.inv(extrinsics) @ pose  # opencv
                flip_yz = np.eye(4)
                flip_yz[1:3, 1:3] *= -1
                pose = flip_yz @ pose @ flip_yz
                
                if pose is None:
                    self.cancel_add()
                    self.adding_model = True
                    self.msg.setText("PnP solver failed, please re-select")
                    return

                self.add_anno(self.obj, pose)
                self.update_anno()
                self.update_anno_treeview()
                
                self.cancel_add()
            else:
                self.msg.setText("Not enough correspondences or inconsistent correspondences")
                self.cancel_add()
            
        else:
            self.adding_model = True
            self.msg.setText("Please select at least four correspondences")
            self.button_add.setText('Confirm')
    
    def add_anno(self, obj, pose):
        start = next_available_start(list(self.annotations.keys()), len(obj))
        self.annotations[(start, start + len(obj) - 1)] = obj
        self.anno_scene.add(obj, start)
        for node in obj:
            self.scene_vis.add_geometry(node.mesh)
        obj.transform(pose)
        self.curr_selection = (start, start + len(obj) - 1)
        
    def remove_anno(self, key):
        if key not in self.annotations:
            self.msg.setText('Can only delete root object')
            return
        obj = self.annotations[key]
        for node in obj:
            self.scene_vis.remove_geometry(node.mesh)
        self.anno_scene.remove(obj)
        for i in Config.CAMS_TO_ANNO:
            self.seg_scenes[i].remove_anno(key)
        del self.annotations[key]
        self.curr_selection = None
        self.cand_selection = (None, None)
        self.reset_obj_mesh(None)
        self.update_anno()
        self.update_anno_treeview()
        
    def update_anno(self, update_renderer=True, update_all=False):
        if self.anno_scene is None:
            return
        if (self.tab.currentIndex() == 0 and not update_all) or update_all:
            if update_renderer:
                self.anno_scene.update()
            for i in Config.CAMS_TO_ANNO:
                bg = self.img_frames[i].bg
                rendered = self.anno_scene.get_rgb(Config.CAMS_TO_ANNO.index(i))
                rendered_mask = self.anno_scene.get_mask(Config.CAMS_TO_ANNO.index(i))
                
                img_cu = torch.from_numpy(bg.copy()).cuda().float()
                rendered_cu = torch.from_numpy(rendered).cuda().float()
                rendered_mask_cu = torch.from_numpy(rendered_mask).cuda()
                mask = torch.any(rendered_cu > 0, -1)
                img_cu[mask] = img_cu[mask] * 0.5 + rendered_cu[mask] * 0.5
                
                elem = torch.ones((5, 5)).to(img_cu)
                if self.curr_selection is not None:
                    sel_mask = ((rendered_mask_cu >= self.curr_selection[0]) & (rendered_mask_cu <= self.curr_selection[1])).float()
                    sel_mask = dilation(sel_mask[None, None], elem)[0][0] - sel_mask
                    bool_mask = sel_mask.bool()
                    sel_mask = sel_mask[..., None] * torch.tensor([255., 0, 51.]).to(img_cu)
                    img_cu[bool_mask] = sel_mask[bool_mask]
                
                if self.cand_selection[0] is not None:
                    sel_mask = ((rendered_mask_cu >= self.cand_selection[0][0]) & (rendered_mask_cu <= self.cand_selection[0][0])).float()
                    sel_mask = dilation(sel_mask[None, None], elem)[0][0] - sel_mask
                    bool_mask = sel_mask.bool()
                    sel_mask = sel_mask[..., None] * torch.tensor([255., 204., 204.]).to(img_cu)
                    img_cu[bool_mask] = sel_mask[bool_mask]
                
                img = img_cu.cpu().numpy().astype(np.uint8)
                self.img_frames[i].img.setImage(img)
                self.img_frames[i].rendered = img

        if (self.tab.currentIndex() == 1 and not update_all) or update_all:
            for obj in self.annotations.values():
                for node in obj:
                    self.scene_vis.update_geometry(node.mesh)
    
    def update_anno_treeview(self):
        def key2str(key):
            return ','.join(np.arange(key[0], key[1] + 1).astype(str))
        self.anno_tree.clear()
        items = []
        for key, obj in self.annotations.items():
            if obj.is_leaf():
                root = QTreeWidgetItem([key2str(key), obj.name, 'rigid'])
            else:
                root = QTreeWidgetItem([key2str(key), obj.name, 'articulated'])
                base_idx = key[0]
                def create_widget(node, widget : QTreeWidgetItem):
                    if node.is_leaf():
                        offset = obj.get_offset(node) + base_idx
                        return widget, (offset, offset)
                    
                    start = 10000
                    end = 0
                    for child in node.children:
                        item = QTreeWidgetItem(['', child.name, 'articulated part'])
                        child_widget, (s, e) = create_widget(child, item)
                        item.setText(0, key2str((s, e)))
                        start = min(start, s)
                        end = max(end, e)
                        widget.addChild(child_widget)
                        widget.setText(0, key2str((start, end)))
                    return widget, (start, end)
                create_widget(obj, root)
            
            items.append(root)
        self.anno_tree.insertTopLevelItems(0, items)
        self.anno_tree.expandAll()

        
if __name__ == "__main__":
    app = QApplication([])
    app.setStyleSheet("QLabel{font-size: 14pt;}")
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec_())
