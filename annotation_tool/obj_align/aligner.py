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
import cv2
import trimesh
import win32gui
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QSizePolicy, QLabel, \
    QTreeWidgetItem, QFileDialog, QDialog, QDialogButtonBox, QVBoxLayout, QScrollArea
from PyQt5.QtCore import QFile, QThread, QObject, QTimer, Qt, QEvent, QPoint
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QWindow, QPainter, QImage, QPixmap
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
from utils.obj import ObjectNode, load_obj, trimesh2o3d
from utils.pose import bcot_track, estimate_rel_pose
from utils.seg import SegScene
import utils.transform as transform
import pymeshlab
from utils.render import AnnoScene, compose_img, render_silhouette

from utils.ui import ImageSelector, MessageDialog, RunDialog, create_image_sel_dialog
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

pg.setConfigOption('imageAxisOrder', 'row-major')
# pg.setConfigOption('leftButtonPan', False)  # if False, then dragging the left mouse button draws a rectangle


# Handle high resolution displays:
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    

class Aligner(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.fspath(Path(__file__).resolve().parent / "aligner.ui"), self)
        self.setup_viewer3d()
        self.obj = None
        self.mesh_frame = None
        self.simplified = False
        self.mesh_path = None
        self.msg = QLabel()
        self.statusBar().addWidget(self.msg)
        
        self.glctx = dr.RasterizeGLContext()
        
        self.findChild(QObject, 'actionOpen').triggered.connect(self.open)
        self.findChild(QObject, 'actionSave').triggered.connect(self.save)
        self.findChild(QObject, 'autoButton').clicked.connect(self.auto_align)
        self.findChild(QObject, 'simplifyButton').clicked.connect(self.simplify)
        self.findChild(QObject, 'swapxyButton').clicked.connect(
            lambda _: self.transform(np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]))
        )
        self.findChild(QObject, 'swapxzButton').clicked.connect(
            lambda _: self.transform(np.array([
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]))
        )
        self.findChild(QObject, 'swapyzButton').clicked.connect(
            lambda _: self.transform(np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]))
        )
        
        self.findChild(QObject, 'flipxButton').clicked.connect(
            lambda _: self.transform(np.array([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]))
        )
        self.findChild(QObject, 'flipyButton').clicked.connect(
            lambda _: self.transform(np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]))
        )
        self.findChild(QObject, 'flipzButton').clicked.connect(
            lambda _: self.transform(np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]))
        )
        
        self.cfg = {
            'trans_interval': 3e-3,
            'rot_interval': 0.5,  # in degree
        }
        self.showMaximized()
        
    
    def resizeEvent(self, event):
        self.windows = []
        for i in range(3):
            window = self.findChild(QObject, ['xy', 'xz', 'yz'][i] + 'Plane')
            self.windows.append(window)
            h, w = window.height(), window.width()
            self.h, self.w = h, w
        
    def open(self):
        path = QFileDialog.getOpenFileName(self, "Open Mesh", 
                                           "./data/models" if self.mesh_path is None else str(self.mesh_path.parent), "Mesh Files (*.obj)")[0]
        if path is None or len(path) == 0:
            return

        self.ms = pymeshlab.MeshSet()
        self.ms.load_new_mesh(str(path))
        self.mesh_path = Path(path)
        
        self.reset_obj_mesh(trimesh2o3d(trimesh.load(str(path), force='mesh', process=True)))
        self.simplified = False
        
        
    def save(self):
        if self.obj is None:
            return
        
        # print(self.mesh_path.parent.parent.parent / ('models_aligned_lowres' if self.simplified else 'models_aligned_highres'))
        save_path = (self.mesh_path.parent.parent.parent / ('models_aligned_lowres' if self.simplified else 'models_aligned_highres')).joinpath(*self.mesh_path.parts[-2:])
        Path(save_path).parent.mkdir(exist_ok=True)
        path = QFileDialog.getSaveFileName(self, "Save Mesh", 
                                           str(save_path), "OBJ (*.obj)")[0]
        if path is None or len(path) == 0:
            return
        
        self.ms.save_current_mesh(path)
        self.statusBar().showMessage('Saved!', 3000)
        
    def reset_obj_mesh(self, obj):
        if self.obj is not None:
            self.vis.remove_geometry(self.obj)
            self.vis.remove_geometry(self.mesh_frame)
            
        # if not obj.has_vertex_normals():
        #     obj.compute_vertex_normals()
        self.obj = obj
        extent = np.array(obj.get_axis_aligned_bounding_box().get_extent()).max()
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=extent * 0.7, origin=[0, 0, 0])
        self.vis.add_geometry(self.obj)
        self.vis.add_geometry(self.mesh_frame)
        self.vis.update_geometry(self.obj)
        self.vis.update_geometry(self.mesh_frame)
        self.statusBar().showMessage('{} Vertices: {}, Triangles: {}'.format(self.mesh_path.stem, len(obj.vertices), len(obj.triangles)))
        self.recenter()
        self.update_proj()
    
    def init_proj(self):
        n, f = 0.1, 100
        extent = np.array(self.obj.get_axis_aligned_bounding_box().get_extent())
        eye_dist = 0.11
        
        
        self.projs = []
        self.bboxs = []
        margin = 0.2
        for i in range(3):
            if i == 0:
                offset = -extent[2] / 2 - eye_dist
                scale = np.min([2 / extent[0], 2 / extent[1]]) / (1 + margin)
                trans = torch.from_numpy(np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, offset],
                    [0, 0, 0, 1]
                ]).astype(np.float32)).cuda()
                self.bboxs.append(np.array([(1 - scale * extent[0] / 2 * self.h / self.w) * self.w / 2, (1 - scale * extent[1] / 2) * self.h / 2, 
                                            (1 + scale * extent[0] / 2 * self.h / self.w) * self.w / 2, (1 + scale * extent[1] / 2) * self.h / 2]))
            elif i == 1:
                offset = -extent[1] / 2 - eye_dist
                scale = np.min([2 / extent[0], 2 / extent[2]]) / (1 + margin)
                trans = torch.from_numpy(np.array([
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, offset],
                    [0, 0, 0, 1]
                ]).astype(np.float32)).cuda()
                self.bboxs.append(np.array([(1 - scale * extent[0] / 2 * self.h / self.w) * self.w / 2, (1 - scale * extent[2] / 2) * self.h / 2, 
                                            (1 + scale * extent[0] / 2 * self.h / self.w) * self.w / 2, (1 + scale * extent[2] / 2) * self.h / 2]))
            else:
                offset = -extent[0] / 2 - eye_dist
                scale = np.min([2 / extent[1], 2 / extent[2]]) / (1 + margin)
                trans = torch.from_numpy(np.array([
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [-1, 0, 0, offset],
                    [0, 0, 0, 1]
                ]).astype(np.float32)).cuda()
                self.bboxs.append(np.array([(1 - scale * extent[2] / 2 * self.h / self.w) * self.w / 2, (1 - scale * extent[1] / 2) * self.h / 2, 
                                            (1 + scale * extent[2] / 2 * self.h / self.w) * self.w / 2, (1 + scale * extent[1] / 2) * self.h / 2]))
            proj = torch.from_numpy(np.array([[scale * self.h / self.w,    0,            0,              0],
                            [  0,  scale,            0,              0],
                            [  0,    0, -2/(f-n), -(f+n)/(f-n)],  # orthogonal  
                            [  0,    0,           0,              1]]).astype(np.float32)).cuda()
        
            self.projs.append(proj @ trans)
            
            
    def setup_viewer3d(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=320, height=240)
        self.vis.get_render_option().mesh_show_back_face = False
        self.vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])

       
        window = QWindow.fromWinId(win32gui.FindWindowEx(0, 0, None, "Open3D"))
        widget = QWidget.createWindowContainer(window, self)
        widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.findChild(QObject, 'verticalLayout').insertWidget(0, widget)
        
        timer = QTimer(self)
        timer.timeout.connect(self.update_vis)
        timer.start(10)
        
    def update_vis(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def auto_align(self):
        if self.obj is None:
            return
        
        bbox = self.obj.get_oriented_bounding_box()
        center = np.array(bbox.center)
        rot = np.array(bbox.R)
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, -1] = center
        self.transform(np.linalg.inv(pose))
        
    def simplify(self):
        dialog = RunDialog(self)
        if dialog.exec():
            self.vis.remove_geometry(self.obj)
            
            self.ms.meshing_decimation_quadric_edge_collapse_with_texture(targetperc=float(dialog.frac.text()), preserveboundary=True, preservenormal=True)
            self.obj = self.obj.simplify_quadric_decimation(int(len(self.obj.triangles) * float(dialog.frac.text())))
            self.vis.add_geometry(self.obj)
            self.vis.update_geometry(self.obj)
            self.statusBar().showMessage('{}, Vertices: {}, Triangles: {}'.format(self.mesh_path.stem, len(self.obj.vertices), len(self.obj.triangles)))
            # MessageDialog('Alert', 'Complete!').exec()
            self.simplified = True
            
    
    def transform(self, trans):
        if self.obj is None:
            return
        
        self.ms.set_matrix(transformmatrix=trans, compose=True)
        self.obj.transform(trans)
        self.recenter()
        
        if np.linalg.det(trans[:3, :3]) < 0:  # fix normal
            self.obj.vertex_normals = o3d.utility.Vector3dVector(np.array(self.obj.vertex_normals) * -1)
            self.obj.triangle_normals = o3d.utility.Vector3dVector(np.array(self.obj.triangle_normals) * -1)
            
        self.vis.update_geometry(self.obj)
        self.update_proj()
        
        
    def keyPressEvent(self, event):
        key = event.key()
        
        if event.modifiers() & Qt.ControlModifier:
            if key == Qt.Key.Key_Z:
                return
        
        # pos change
        if key in [Qt.Key.Key_Q, Qt.Key.Key_W, Qt.Key.Key_E, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D,
                           Qt.Key.Key_Z, Qt.Key.Key_X, Qt.Key.Key_C, Qt.Key.Key_V, Qt.Key.Key_B, Qt.Key.Key_N]:
            if self.obj is None:
                return
            trans_interval = self.cfg['trans_interval']
            rot_interval = self.cfg['rot_interval'] / 180 * np.pi
            delta = np.eye(4)
            
            if key == Qt.Key.Key_Q:
                delta[2, 3] += trans_interval
            elif key == Qt.Key.Key_E:
                delta[2, 3] -= trans_interval
            elif key == Qt.Key.Key_W:
                delta[1, 3] += trans_interval
            elif key == Qt.Key.Key_S:
                delta[1, 3] -= trans_interval
            elif key == Qt.Key.Key_A:
                delta[0, 3] -= trans_interval
            elif key == Qt.Key.Key_D:
                delta[0, 3] += trans_interval
            elif key == Qt.Key.Key_Z:
                delta[:3, :3] = R.from_rotvec(np.array([1, 0, 0]) * rot_interval).as_matrix()
            elif key == Qt.Key.Key_X:
                delta[:3, :3] = R.from_rotvec(np.array([-1, 0, 0]) * rot_interval).as_matrix()
            elif key == Qt.Key.Key_C:
                delta[:3, :3] = R.from_rotvec(np.array([0, 1, 0]) * rot_interval).as_matrix()
            elif key == Qt.Key.Key_V:
                delta[:3, :3] = R.from_rotvec(np.array([0, -1, 0]) * rot_interval).as_matrix()
            elif key == Qt.Key.Key_B:
                delta[:3, :3] = R.from_rotvec(np.array([0, 0, 1]) * rot_interval).as_matrix()
            elif key == Qt.Key.Key_N:
                delta[:3, :3] = R.from_rotvec(np.array([0, 0, -1]) * rot_interval).as_matrix()
            
            self.transform(delta)
    
    def recenter(self):
        trans = np.eye(4)
        center = np.array(self.obj.get_axis_aligned_bounding_box().get_center())
        trans[:3, -1] = -center
        self.ms.set_matrix(transformmatrix=trans, compose=True)
        self.obj.transform(trans)
        self.vis.update_geometry(self.obj)
        
    def update_proj(self):
        self.init_proj()
        pos = []
        for i in range(3):
            vertices = torch.from_numpy(np.array(self.obj.vertices)).cuda().float()
            vertices = torch.cat([vertices, torch.ones((vertices.shape[0], 1)).to(vertices)], -1)
            pos.append((self.projs[i] @ vertices.T).T.contiguous())
            
        pos = torch.stack(pos)
        pos_idx = torch.from_numpy(np.array(self.obj.triangles)).cuda().int()
        vtx_col = torch.from_numpy(np.array(self.obj.vertex_colors)).cuda().float()
        rast_out, _ = dr.rasterize(self.glctx, pos, pos_idx, resolution=[self.h, self.w])
        color   , _ = dr.interpolate(vtx_col[None, ...], rast_out, pos_idx)
        
        imgs = (color.cpu().numpy()[:, ::-1] * 255).astype(np.uint8)
        for img, window, bbox in zip(imgs, self.windows, self.bboxs):
            cv2.line(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[1])), (255, 0, 0), 2, -1)
            cv2.line(img, (int(bbox[2]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2, -1)
            cv2.line(img, (int(bbox[2]), int(bbox[3])), (int(bbox[0]), int(bbox[3])), (255, 0, 0), 2, -1)
            cv2.line(img, (int(bbox[0]), int(bbox[3])), (int(bbox[0]), int(bbox[1])), (255, 0, 0), 2, -1)
            
            cv2.line(img, (self.w // 2, 0), (self.w // 2, self.h), (0, 255, 0), 2, -1)
            cv2.line(img, (0, self.h // 2), (self.w, self.h // 2), (0, 255, 0), 2, -1)
            
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
            
            window.setPixmap(QPixmap.fromImage(qimg))
            window.update()
        
        

if __name__ == "__main__":
    app = QApplication([])
    app.setStyleSheet("QLabel{font-size: 14pt;}")
    widget = Aligner()
    widget.show()
    sys.exit(app.exec_())
