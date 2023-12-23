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
try:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
except:
    pass
import numpy as np
from utils.obj import load_obj
from subprocess import Popen, PIPE

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
import math
def fibonacci_sphere(samples):
    
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

# Handle high resolution displays:
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    

class Aligner(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.fspath(Path(__file__).resolve().parent / "annotator.ui"), self)
        self.setup_viewer3d()
        self.obj = None
        self.mesh_frame = None
        self.mesh_path = None
        self.msg = QLabel()
        self.statusBar().addWidget(self.msg)
        
        ident = np.identity(3)
        # self.glctx = dr.RasterizeGLContext()
        self.findChild(QObject, 'x180Button').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident, R.from_rotvec(np.pi * np.array([1., 0, 0])).as_matrix()])
            )
        )
        self.findChild(QObject, 'y180Button').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident, R.from_rotvec(np.pi * np.array([0, 1., 0])).as_matrix()])
            )
        )
        self.findChild(QObject, 'z180Button').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident, R.from_rotvec(np.pi * np.array([0, 0, 1.])).as_matrix()])
            )
        )
        
        self.findChild(QObject, 'x90Button').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident, *[R.from_rotvec(deg * np.array([1., 0, 0]), degrees=True).as_matrix() for deg in range(0, 360, 90)]])
            )
        )
        self.findChild(QObject, 'y90Button').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident, *[R.from_rotvec(deg * np.array([0, 1., 0]), degrees=True).as_matrix() for deg in range(0, 360, 90)]])
            )
        )
        self.findChild(QObject, 'z90Button').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident, *[R.from_rotvec(deg * np.array([0, 0, 1.]), degrees=True).as_matrix() for deg in range(0, 360, 90)]])
            )
        )
        
        self.findChild(QObject, 'xinfButton').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident, *[R.from_rotvec(deg * np.array([1., 0, 0]), degrees=True).as_matrix() for deg in range(0, 360, 10)]])
            )
        )
        self.findChild(QObject, 'yinfButton').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident, *[R.from_rotvec(deg * np.array([0, 1., 0]), degrees=True).as_matrix() for deg in range(0, 360, 10)]])
            )
        )
        self.findChild(QObject, 'zinfButton').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident, *[R.from_rotvec(deg * np.array([0, 0, 1.]), degrees=True).as_matrix() for deg in range(0, 360, 10)]])
            )
        )
        self.findChild(QObject, 'noneButton').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident,])
            )
        )
        num_samples = int(4 * np.pi / (10. / 180 * np.pi))
        sphere_pts = np.array(fibonacci_sphere(num_samples))
        self.findChild(QObject, 'allinfButton').clicked.connect(
            lambda _: self.write_mat(
                np.stack([ident, *[R.from_rotvec(deg * pt, degrees=True).as_matrix() for deg in range(0, 360, 10) for pt in sphere_pts]])
            )
        )
        
        self.showMaximized()
        mesh, self.mesh_path = self.get_next_model()
        self.reset_obj_mesh(mesh)
    
    def write_mat(self, mat):
        np.save(self.mesh_path.parent / (self.mesh_path.stem + '.sym.npy'), mat)
        mesh, self.mesh_path = self.get_next_model()
        self.reset_obj_mesh(mesh)
        
    def get_next_model(self):
        for cat in Path('data/models_aligned_highres').glob('*/'):
            for fn in cat.glob('*'):
                if (fn.is_dir() or fn.suffix == '.obj') and not (fn.parent / (fn.stem + '.sym.npy')).exists():
                    mesh = load_obj(Path(str(fn).replace('highres', 'lowres'))).mesh
                    return mesh, fn
                
    def resizeEvent(self, event):
        self.windows = []
        for i in range(3):
            window = self.findChild(QObject, ['xy', 'xz', 'yz'][i] + 'Plane')
            self.windows.append(window)
            h, w = window.height(), window.width()
            self.h, self.w = h, w
        
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
        # self.update_proj()
    
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
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=320, height=240)
        self.vis.get_render_option().mesh_show_back_face = False
        self.vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])

        if os.name == 'nt':
            import win32gui
            window = QWindow.fromWinId(win32gui.FindWindowEx(0, 0, None, "Open3D"))
        else:
            print(self.get_hwnd())
            window = QWindow.fromWinId(self.get_hwnd())
        widget = QWidget.createWindowContainer(window, self)
        widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.findChild(QObject, 'verticalLayout').insertWidget(0, widget)
        
        timer = QTimer(self)
        timer.timeout.connect(self.update_vis)
        timer.start(10)
        
    def update_vis(self):
        self.vis.poll_events()
        self.vis.update_renderer()
    
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
