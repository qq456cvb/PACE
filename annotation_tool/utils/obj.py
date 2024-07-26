from copy import deepcopy
from functools import lru_cache
import logging
from pathlib import Path
import sys
from time import time
from tqdm import tqdm
import xmltodict
import open3d as o3d
import numpy as np
import trimesh
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QSizePolicy, QLabel, QTreeWidgetItem, QFileDialog
from PyQt5.QtCore import QFile, QThread, QObject, QTimer, Qt, QEvent, QPoint
from PyQt5.QtGui import QWindow, QPainter, QImage
from trimesh.base import Trimesh
class ObjectNode:
    def __init__(self, mesh=None, name=None, path=None) -> None:
        self.children = []
        self.mesh = mesh
        self.parent = None
        self.pose = np.eye(4)
        self.name = name
        self.depth = 0
        self.path = str(Path(path).absolute().relative_to(Path(__file__).parent.parent.absolute())).replace('\\', '/') if path is not None else None
    
    def get_child(self, idx):
        if self.children is None:
            return self
        return self.children[idx]
    
    def __iter__(self):
        if self.is_leaf():
            yield self
        for child in self.children:
            for node in child:
                yield node
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        
    def transform(self, trans):
        self.pose = trans @ self.pose
        if self.mesh is not None:
            self.mesh.transform(trans)
        for child in self.children:
            child.transform(trans)

    def get_node_by_name(self, name):
        def get_rec(node):
            if node.name == name:
                return node

            for child in node.children:
                res = get_rec(child)
                if res is not None:
                    return res
            return None

        return get_rec(self)
    
    def get_offset(self, target):
        cnt = 0
        def get_rec(node):
            nonlocal cnt
            if target == node:
                return cnt
            elif node.is_leaf():
                cnt += 1
                return None

            for child in node.children:
                res = get_rec(child)
                if res is not None:
                    return res
            return None

        return get_rec(self)
    
    
    def get_node(self, idx):
        cnt = 0
        def get_rec(node):
            nonlocal cnt
            if node.is_leaf():
                if cnt == idx:
                    return node
                else:
                    cnt += 1
                    return None

            for child in node.children:
                res = get_rec(child)
                if res is not None:
                    return res
            return None

        return get_rec(self)

    def __len__(self):
        cnt = 0
        if len(self.children) == 0:
            return 1
        for child in self.children:
            cnt += len(child)
        return cnt
    
    @property
    def center(self):
        return np.array(self.mesh.vertices).mean(0)
    

def load_obj(fn : Path, read_mesh=True) -> ObjectNode:
    if fn.is_dir():
        anno_file = next(fn.glob('*.urdf'))
        with open(anno_file, 'rb') as fp:
            data_dict = xmltodict.parse(fp.read())['robot']
        links = data_dict['link']
        if isinstance(links, dict):
            logging.warning(f'links should be list type, or there is only one part in object')
        
        nodes = dict()
        for link in links:
            name = link['@name']
            try:
                path = str(link['collision']['geometry']['mesh']['@filename'])[len('package://'):]
            except:
                path = str(link['visual']['geometry']['mesh']['@filename'])[len('package://'):]
            
            if read_mesh:
                if (fn / path.replace('.obj', '.cache.ply')).exists():
                    mesh = load_mesh(fn / path.replace('.obj', '.cache.ply'), use_trimesh=False, scale=1.)
                else:
                    mesh = load_mesh(fn / path, use_trimesh=True, scale=1.)
                    o3d.io.write_triangle_mesh(str(fn / path.replace('.obj', '.cache.ply')), mesh)
            else:
                mesh = None
            node = ObjectNode(mesh, name)
            nodes[name] = node

        joints = data_dict['joint']
        if isinstance(joints, dict):
            joints = [joints]
            
        for joint in joints:
            parent_name = joint['parent']['@link']
            child_name = joint['child']['@link']
            nodes[parent_name].add_child(nodes[child_name])
        
        def traverse_obj_tree(node, depth):
            
            if not node.is_leaf():
                child_nodes = []
                child_mesh = o3d.geometry.TriangleMesh(node.mesh) if read_mesh else None
                for child in node.children:
                    child_node = traverse_obj_tree(child, depth + 1)
                    child_nodes.append(child_node)
                    if read_mesh:
                        child_mesh += child_node.mesh
            else:
                node.depth = depth
                return node
            
            parent = ObjectNode(child_mesh)
            parent.depth = depth
            parent.name = child.name + '__P'
            for child in child_nodes:
                parent.add_child(child)
            node.children = []
            node.depth = depth + 1
            parent.add_child(node)
            return parent

        for node in nodes.values():
            if node.parent is None:
                root = traverse_obj_tree(node, 0)
                root.name = fn.stem
                root.path = str(fn)
                return root
    else:
        return ObjectNode(load_mesh(fn) if read_mesh else None, fn.stem, path=str(fn))


def trimesh2o3d(mesh):
    obj = o3d.geometry.TriangleMesh()
    obj.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    obj.triangles = o3d.utility.Vector3iVector(mesh.faces)
    try:
        obj.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh.visual.to_color().vertex_colors[:, :3] / 255.))
    except:
        obj.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array([[1., 0, 0]]), (mesh.vertices.shape[0], 1)))
    return obj

mesh_cache = dict()
def load_mesh(path, use_trimesh=True, scale=1e-3):
    if path not in mesh_cache:
        if use_trimesh:
            mesh = trimesh2o3d(trimesh.load(str(path), force='mesh', process=False))
        else:
            mesh = o3d.io.read_triangle_mesh(str(path), True)
        mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices) * scale)
        mesh_cache[path] = mesh
    return deepcopy(mesh_cache[path])
    


    
if __name__ == '__main__':
    node = ObjectNode(path='E:\\annotator\\data\\models')
    print(node.path)
    # obj = load_obj(Path('data/models/all/trashcan7'))
    exit()
    annotions = dict()
    annotions[(1, 3)] = obj
    def key2str(key):
        return ','.join(np.arange(key[0], key[1] + 1).astype(str))
        
    for key, obj in annotions.items():
        if obj.is_leaf():
            root = QTreeWidgetItem([key2str(key), obj.name, 'rigid'])
        else:
            root = QTreeWidgetItem([key2str(key), obj.name, 'articulated'])
            
            def create_widget(node, widget : QTreeWidgetItem):
                if node.is_leaf():
                    print(obj.get_offset(node))
                    return widget, (obj.get_offset(node), obj.get_offset(node))
                
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
    # print(obj.pose)
    # trans = np.eye(4)
    # trans[:3, -1] = -obj.center
    # obj.transform(trans)
    # print(obj.pose)
    # rad = np.array(obj.mesh.get_axis_aligned_bounding_box().get_extent()).max() / 40
    # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=rad)
    # mesh_sphere.paint_uniform_color(np.array([1, 0, 0]))
    # obj.mesh += mesh_sphere
    # o3d.visualization.draw_geometries([obj])
    # print(obj.children)
    # print(len(obj))
    # print(obj.depth)
    # print(obj.get_node(0).depth, obj.get_node(1).depth)
    # for node in obj:
    #     o3d.visualization.draw_geometries([node.mesh])
    # print(obj.children[0].depth)
    # o3d.visualization.draw_geometries([obj.children[0].mesh])
    # print(obj.children[1].depth)
    # o3d.visualization.draw_geometries([obj.children[1].mesh])
    # print(obj.children[0].children[0].depth)
    # o3d.visualization.draw_geometries([obj.children[0].children[0].mesh])
    # print(obj.children[0].children[1].depth)
    # o3d.visualization.draw_geometries([obj.children[0].children[1].mesh])
