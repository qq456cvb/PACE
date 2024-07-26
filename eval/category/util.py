"""
Mask R-CNN
Common utility functions and classes.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import cv2
import math
import random
import numpy as np
import scipy.misc
import skimage.color
import _pickle as cPickle
from ctypes import *
import copy
# from . import ICP
import glob
import time

import torch

from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from scipy.stats import special_ortho_group
from multiprocessing import Pool

"""General 3D Bounding Box class."""

import numpy as np
from numpy.linalg import lstsq as optimizer
from scipy.spatial.transform import Rotation as rotation_util

EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)

# The vertices are ordered according to the left-hand rule, so the normal
# vector of each face will point inward the box.
FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])

UNIT_BOX = np.asarray([
    [0., 0., 0.],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
])

NUM_KEYPOINTS = 9
FRONT_FACE_ID = 4
TOP_FACE_ID = 2

import json
model_info = json.load(open('/orion/u/yangyou/COLSPA/data/bop/colspa/models/models_info.json'))
symmetry_info = {int(k): np.array(v['symmetries_discrete']).reshape(-1, 4, 4) for k, v in model_info.items() if 'symmetries_discrete' in v}

############################################################
#  Bounding Boxes
############################################################

class Box(object):
    """General 3D Oriented Bounding Box."""

    def __init__(self, vertices=None):
        if vertices is None:
            vertices = self.scaled_axis_aligned_vertices(
                np.array([1., 1., 1.]))

        self._vertices = vertices
        self._rotation = None
        self._translation = None
        self._scale = None
        self._transformation = None
        self._volume = None

    @classmethod
    def from_transformation(cls, rotation, translation, scale):
        """Constructs an oriented bounding box from transformation and scale."""
        if rotation.size != 9:
            raise ValueError('Unsupported rotation, only 3x1 euler angles or 3x3 ' +
                             'rotation matrices are supported. ' + rotation)
        scaled_identity_box = cls.scaled_axis_aligned_vertices(scale)
        vertices = np.zeros((NUM_KEYPOINTS, 3))
        for i in range(NUM_KEYPOINTS):
            vertices[i, :] = np.matmul(
                rotation, scaled_identity_box[i, :]) + translation.flatten()
        return cls(vertices=vertices)

    def __repr__(self):
        representation = 'Box: '
        for i in range(NUM_KEYPOINTS):
            representation += '[{0}: {1}, {2}, {3}]'.format(i, self.vertices[i, 0],
                                                            self.vertices[i, 1],
                                                            self.vertices[i, 2])
        return representation

    def __len__(self):
        return NUM_KEYPOINTS

    def __name__(self):
        return 'Box'

    def apply_transformation(self, transformation):
        """Applies transformation on the box.
        Group multiplication is the same as rotation concatenation. Therefore return
        new box with SE3(R * R2, T + R * T2); Where R2 and T2 are existing rotation
        and translation. Note we do not change the scale.
        Args:
          transformation: a 4x4 transformation matrix.
        Returns:
           transformed box.
        """
        if transformation.shape != (4, 4):
            raise ValueError('Transformation should be a 4x4 matrix.')

        new_rotation = np.matmul(transformation[:3, :3], self.rotation)
        new_translation = transformation[:3, 3] + (
            np.matmul(transformation[:3, :3], self.translation))
        return Box.from_transformation(new_rotation, new_translation, self.scale)

    @classmethod
    def scaled_axis_aligned_vertices(cls, scale):
        """Returns an axis-aligned set of verticies for a box of the given scale.
        Args:
          scale: A 3*1 vector, specifiying the size of the box in x-y-z dimension.
        """
        w = scale[0] / 2.
        h = scale[1] / 2.
        d = scale[2] / 2.

        # Define the local coordinate system, w.r.t. the center of the box
        aabb = np.array([[0., 0., 0.], [-w, -h, -d], [-w, -h, +d], [-w, +h, -d],
                         [-w, +h, +d], [+w, -h, -d], [+w, -h, +d], [+w, +h, -d],
                         [+w, +h, +d]])
        return aabb

    @classmethod
    def fit(cls, vertices):
        """Estimates a box 9-dof parameters from the given vertices.
        Directly computes the scale of the box, then solves for orientation and
        translation.
        Args:
          vertices: A 9*3 array of points. Points are arranged as 1 + 8 (center
            keypoint + 8 box vertices) matrix.
        Returns:
          orientation: 3*3 rotation matrix.
          translation: 3*1 translation vector.
          scale: 3*1 scale vector.
        """
        orientation = np.identity(3)
        translation = np.zeros((3, 1))
        scale = np.zeros(3)

        # The scale would remain invariant under rotation and translation.
        # We can safely estimate the scale from the oriented box.
        for axis in range(3):
            for edge_id in range(4):
                # The edges are stored in quadruples according to each axis
                begin, end = EDGES[axis * 4 + edge_id]
                scale[axis] += np.linalg.norm(vertices[begin,
                                              :] - vertices[end, :])
            scale[axis] /= 4.

        x = cls.scaled_axis_aligned_vertices(scale)
        system = np.concatenate((x, np.ones((NUM_KEYPOINTS, 1))), axis=1)
        solution, _, _, _ = optimizer(system, vertices, rcond=None)
        orientation = solution[:3, :3].T
        translation = solution[3, :3]
        return orientation, translation, scale

    def inside(self, point):
        """Tests whether a given point is inside the box.
          Brings the 3D point into the local coordinate of the box. In the local
          coordinate, the looks like an axis-aligned bounding box. Next checks if
          the box contains the point.
        Args:
          point: A 3*1 numpy vector.
        Returns:
          True if the point is inside the box, False otherwise.
        """
        inv_trans = np.linalg.inv(self.transformation)
        scale = self.scale
        point_w = np.matmul(inv_trans[:3, :3], point) + inv_trans[:3, 3]
        for i in range(3):
            if abs(point_w[i]) > scale[i] / 2.:
                return False
        return True

    def sample(self):
        """Samples a 3D point uniformly inside this box."""
        point = np.random.uniform(-0.5, 0.5, 3) * self.scale
        point = np.matmul(self.rotation, point) + self.translation
        return point

    @property
    def vertices(self):
        return self._vertices

    @property
    def rotation(self):
        if self._rotation is None:
            self._rotation, self._translation, self._scale = self.fit(
                self._vertices)
        return self._rotation

    @property
    def translation(self):
        if self._translation is None:
            self._rotation, self._translation, self._scale = self.fit(
                self._vertices)
        return self._translation

    @property
    def scale(self):
        if self._scale is None:
            self._rotation, self._translation, self._scale = self.fit(
                self._vertices)
        return self._scale

    @property
    def volume(self):
        """Compute the volume of the parallelpiped or the box.
          For the boxes, this is equivalent to np.prod(self.scale). However for
          parallelpiped, this is more involved. Viewing the box as a linear function
          we can estimate the volume using a determinant. This is equivalent to
          sp.ConvexHull(self._vertices).volume
        Returns:
          volume (float)
        """
        if self._volume is None:
            i = self._vertices[2, :] - self._vertices[1, :]
            j = self._vertices[3, :] - self._vertices[1, :]
            k = self._vertices[5, :] - self._vertices[1, :]
            sys = np.array([i, j, k])
            self._volume = abs(np.linalg.det(sys))
        return self._volume

    @property
    def transformation(self):
        if self._rotation is None:
            self._rotation, self._translation, self._scale = self.fit(
                self._vertices)
        if self._transformation is None:
            self._transformation = np.identity(4)
            self._transformation[:3, :3] = self._rotation
            self._transformation[:3, 3] = self._translation
        return self._transformation

    def get_ground_plane(self, gravity_axis=1):
        """Get ground plane under the box."""

        gravity = np.zeros(3)
        gravity[gravity_axis] = 1

        def get_face_normal(face, center):
            """Get a normal vector to the given face of the box."""
            v1 = self.vertices[face[0], :] - center
            v2 = self.vertices[face[1], :] - center
            normal = np.cross(v1, v2)
            return normal

        def get_face_center(face):
            """Get the center point of the face of the box."""
            center = np.zeros(3)
            for vertex in face:
                center += self.vertices[vertex, :]
            center /= len(face)
            return center

        ground_plane_id = 0
        ground_plane_error = 10.

        # The ground plane is defined as a plane aligned with gravity.
        # gravity is the (0, 1, 0) vector in the world coordinate system.
        for i in [0, 2, 4]:
            face = FACES[i, :]
            center = get_face_center(face)
            normal = get_face_normal(face, center)
            w = np.cross(gravity, normal)
            w_sq_norm = np.linalg.norm(w)
            if w_sq_norm < ground_plane_error:
                ground_plane_error = w_sq_norm
                ground_plane_id = i

        face = FACES[ground_plane_id, :]
        center = get_face_center(face)
        normal = get_face_normal(face, center)

        # For each face, we also have a parallel face that it's normal is also
        # aligned with gravity vector. We pick the face with lower height (y-value).
        # The parallel to face 0 is 1, face 2 is 3, and face 4 is 5.
        parallel_face_id = ground_plane_id + 1
        parallel_face = FACES[parallel_face_id]
        parallel_face_center = get_face_center(parallel_face)
        parallel_face_normal = get_face_normal(
            parallel_face, parallel_face_center)
        if parallel_face_center[gravity_axis] < center[gravity_axis]:
            center = parallel_face_center
            normal = parallel_face_normal
        return center, normal


"""The Intersection Over Union (IoU) for 3D oriented bounding boxes."""

import numpy as np
import scipy.spatial as sp
from scipy.stats import special_ortho_group

_PLANE_THICKNESS_EPSILON = 0.000001
_POINT_IN_FRONT_OF_PLANE = 1
_POINT_ON_PLANE = 0
_POINT_BEHIND_PLANE = -1


class IoU(object):
    """General Intersection Over Union cost for Oriented 3D bounding boxes."""

    def __init__(self, box1, box2):
        self._box1 = box1
        self._box2 = box2
        self._intersection_points = []

    def iou(self):
        """Computes the exact IoU using Sutherland-Hodgman algorithm."""
        self._intersection_points = []
        self._compute_intersection_points(self._box1, self._box2)
        self._compute_intersection_points(self._box2, self._box1)
        if self._intersection_points:
            intersection_volume = sp.ConvexHull(
                self._intersection_points).volume
            box1_volume = self._box1.volume
            box2_volume = self._box2.volume
            union_volume = box1_volume + box2_volume - intersection_volume
            return intersection_volume / union_volume
        else:
            return 0.

    def iou_sampling(self, num_samples=10000):
        """Computes intersection over union by sampling points.
        Generate n samples inside each box and check if those samples are inside
        the other box. Each box has a different volume, therefore the number o
        samples in box1 is estimating a different volume than box2. To address
        this issue, we normalize the iou estimation based on the ratio of the
        volume of the two boxes.
        Args:
          num_samples: Number of generated samples in each box
        Returns:
          IoU Estimate (float)
        """
        p1 = [self._box1.sample() for _ in range(num_samples)]
        p2 = [self._box2.sample() for _ in range(num_samples)]
        box1_volume = self._box1.volume
        box2_volume = self._box2.volume
        box1_intersection_estimate = 0
        box2_intersection_estimate = 0
        for point in p1:
            if self._box2.inside(point):
                box1_intersection_estimate += 1
        for point in p2:
            if self._box1.inside(point):
                box2_intersection_estimate += 1
        # We are counting the volume of intersection twice.
        intersection_volume_estimate = (
            box1_volume * box1_intersection_estimate +
            box2_volume * box2_intersection_estimate) / 2.0
        union_volume_estimate = (box1_volume * num_samples + box2_volume *
                                 num_samples) - intersection_volume_estimate
        iou_estimate = intersection_volume_estimate / union_volume_estimate
        return iou_estimate

    def _compute_intersection_points(self, box_src, box_template):
        """Computes the intersection of two boxes."""
        # Transform the source box to be axis-aligned
        inv_transform = np.linalg.inv(box_src.transformation)
        box_src_axis_aligned = box_src.apply_transformation(inv_transform)
        template_in_src_coord = box_template.apply_transformation(
            inv_transform)
        for face in range(len(FACES)):
            indices = FACES[face, :]
            poly = [template_in_src_coord.vertices[indices[i], :]
                    for i in range(4)]
            clip = self.intersect_box_poly(box_src_axis_aligned, poly)
            for point in clip:
                # Transform the intersection point back to the world coordinate
                point_w = np.matmul(
                    box_src.rotation, point) + box_src.translation
                self._intersection_points.append(point_w)

        for point_id in range(NUM_KEYPOINTS):
            v = template_in_src_coord.vertices[point_id, :]
            if box_src_axis_aligned.inside(v):
                point_w = np.matmul(box_src.rotation, v) + box_src.translation
                self._intersection_points.append(point_w)

    def intersect_box_poly(self, box, poly):
        """Clips the polygon against the faces of the axis-aligned box."""
        for axis in range(3):
            poly = self._clip_poly(poly, box.vertices[1, :], 1.0, axis)
            poly = self._clip_poly(poly, box.vertices[8, :], -1.0, axis)
        return poly

    def _clip_poly(self, poly, plane, normal, axis):
        """Clips the polygon with the plane using the Sutherland-Hodgman algorithm.
        See en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm for the overview of
        the Sutherland-Hodgman algorithm. Here we adopted a robust implementation
        from "Real-Time Collision Detection", by Christer Ericson, page 370.
        Args:
          poly: List of 3D vertices defining the polygon.
          plane: The 3D vertices of the (2D) axis-aligned plane.
          normal: normal
          axis: A tuple defining a 2D axis.
        Returns:
          List of 3D vertices of the clipped polygon.
        """
        # The vertices of the clipped polygon are stored in the result list.
        result = []
        if len(poly) <= 1:
            return result

        # polygon is fully located on clipping plane
        poly_in_plane = True

        # Test all the edges in the polygon against the clipping plane.
        for i, current_poly_point in enumerate(poly):
            prev_poly_point = poly[(i + len(poly) - 1) % len(poly)]
            d1 = self._classify_point_to_plane(
                prev_poly_point, plane, normal, axis)
            d2 = self._classify_point_to_plane(current_poly_point, plane, normal,
                                               axis)
            if d2 == _POINT_BEHIND_PLANE:
                poly_in_plane = False
                if d1 == _POINT_IN_FRONT_OF_PLANE:
                    intersection = self._intersect(plane, prev_poly_point,
                                                   current_poly_point, axis)
                    result.append(intersection)
                elif d1 == _POINT_ON_PLANE:
                    if not result or (not np.array_equal(result[-1], prev_poly_point)):
                        result.append(prev_poly_point)
            elif d2 == _POINT_IN_FRONT_OF_PLANE:
                poly_in_plane = False
                if d1 == _POINT_BEHIND_PLANE:
                    intersection = self._intersect(plane, prev_poly_point,
                                                   current_poly_point, axis)
                    result.append(intersection)
                elif d1 == _POINT_ON_PLANE:
                    if not result or (not np.array_equal(result[-1], prev_poly_point)):
                        result.append(prev_poly_point)

                result.append(current_poly_point)
            else:
                if d1 != _POINT_ON_PLANE:
                    result.append(current_poly_point)

        if poly_in_plane:
            return poly
        else:
            return result

    def _intersect(self, plane, prev_point, current_point, axis):
        """Computes the intersection of a line with an axis-aligned plane.
        Args:
          plane: Formulated as two 3D points on the plane.
          prev_point: The point on the edge of the line.
          current_point: The other end of the line.
          axis: A tuple defining a 2D axis.
        Returns:
          A 3D point intersection of the poly edge with the plane.
        """
        alpha = (current_point[axis] - plane[axis]) / (
            current_point[axis] - prev_point[axis])
        # Compute the intersecting points using linear interpolation (lerp)
        intersection_point = alpha * prev_point + (1.0 - alpha) * current_point
        return intersection_point

    def _inside(self, plane, point, axis):
        """Check whether a given point is on a 2D plane."""
        # Cross products to determine the side of the plane the point lie.
        x, y = axis
        u = plane[0] - point
        v = plane[1] - point

        a = u[x] * v[y]
        b = u[y] * v[x]
        return a >= b

    def _classify_point_to_plane(self, point, plane, normal, axis):
        """Classify position of a point w.r.t the given plane.
        See Real-Time Collision Detection, by Christer Ericson, page 364.
        Args:
          point: 3x1 vector indicating the point
          plane: 3x1 vector indicating a point on the plane
          normal: scalar (+1, or -1) indicating the normal to the vector
          axis: scalar (0, 1, or 2) indicating the xyz axis
        Returns:
          Side: which side of the plane the point is located.
        """
        signed_distance = normal * (point[axis] - plane[axis])
        if signed_distance > _PLANE_THICKNESS_EPSILON:
            return _POINT_IN_FRONT_OF_PLANE
        elif signed_distance < -_PLANE_THICKNESS_EPSILON:
            return _POINT_BEHIND_PLANE
        else:
            return _POINT_ON_PLANE

    @property
    def intersection_points(self):
        return self._intersection_points


def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]
    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


############################################################
#  Evaluation
############################################################


def compute_ap_from_matches_scores(pred_match, pred_scores, gt_match):
    # sort the scores from high to low
    # print(pred_match.shape, pred_scores.shape)
    assert pred_match.shape[0] == pred_scores.shape[0]

    score_indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[score_indices]
    pred_match  = pred_match[score_indices]

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap


def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id, obj_id, handle_visibility, synset_names):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
    '''

    ## make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    if obj_id in symmetry_info:
        sym_mats = symmetry_info[obj_id]
        sym_mats = np.concatenate([sym_mats, np.eye(4)[None]])
        # hardcode if the object is sphere
        if len(sym_mats) > 2000:
            theta = 0
        else:
            thetas = []
            for sym_mat in sym_mats:
                R = R1 @ (R2 @ sym_mat[:3, :3]).transpose()
                theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
                thetas.append(theta)
            theta = np.min(thetas)
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result


def compute_RT_overlaps(gt_class_ids, gt_obj_ids, gt_RTs, gt_handle_visibility,
                        pred_class_ids, pred_RTs, 
                        synset_names):
    """Finds overlaps between prediction and ground truth instances.
    Returns:
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # print('num of gt instances: {}, num of pred instances: {}'.format(len(gt_class_ids), len(gt_class_ids)))
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_degree_cm_symmetry(pred_RTs[i], 
                                                              gt_RTs[j], 
                                                              gt_class_ids[j],
                                                              gt_obj_ids[j],
                                                              gt_handle_visibility[j],
                                                              synset_names)
            
    return overlaps


def compute_match_from_degree_cm(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)


    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches


    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2
    

    for d, degree_thres in enumerate(degree_thres_list):                
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Remove low scores
                # low_score_idx = np.where(sum_degree_shift >= 100)[0]
                # if low_score_idx.size > 0:
                #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
                # 3. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    #print(j, len(gt_match), len(pred_class_ids), len(gt_class_ids))
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue

                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """

    pre_shape = x.shape
    assert len(x.shape) == 2, x.shape
    new_x = x[~np.all(x == 0, axis=1)]
    post_shape = new_x.shape
    assert pre_shape[0] == post_shape[0]
    assert pre_shape[1] == post_shape[1]

    return new_x


def compute_3d_iou(RT_1, RT_2, scales_1, scales_2, handle_visibility, class_name_1, class_name_2, gt_obj_id):
    '''Computes IoU overlaps between two 3d bboxes.
       bbox_3d_1, bbox_3d_1: [3, 8]
    '''
    
    def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
        try:
            RT_1[:3, :3] = RT_1[:3, :3] / (np.cbrt(np.linalg.det(RT_1[:3, :3])) + 1e-7)
            RT_2[:3, :3] = RT_2[:3, :3] / (np.cbrt(np.linalg.det(RT_2[:3, :3])) + 1e-7)
            box1 = Box.from_transformation(RT_1[:3, :3], RT_1[:3, -1], scales_1)
            box2 = Box.from_transformation(RT_2[:3, :3], RT_2[:3, -1], scales_2)
            return IoU(box1, box2).iou()
        except:
            return 0
        
    if RT_1 is None or RT_2 is None:
        return -1
    
    if gt_obj_id in symmetry_info:
        sym_mats = symmetry_info[gt_obj_id]
        sym_mats = np.concatenate([sym_mats, np.eye(4)[None]])
        if len(sym_mats) > 2000:
            RT_2 = RT_2.copy()
            RT_2[:3, :3] = RT_1[:3, :3]
            max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)
        else:
            max_iou = 0
            for sym_mat in sym_mats:
                max_iou = max(max_iou, asymmetric_3d_iou(RT_1, RT_2 @ sym_mat, scales_1, scales_2))
    else:
        max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)
    return max_iou


def compute_3d_matches(gt_class_ids, gt_obj_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                       pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales,
                       iou_3d_thresholds, score_threshold=0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)
    
    if num_pred:
        if len(pred_boxes.shape) == 2:
            pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]
        
        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()
        
    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_3d_iou(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j], gt_handle_visibility[j], synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]], gt_obj_ids[j])
    # Loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(len(pred_boxes)):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                #print('gt_match: ', gt_match[j])
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                #print('iou: ', iou)
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue

                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break

    return gt_matches, pred_matches, overlaps, indices


def work(num_iou_thres, num_degree_thres, num_shift_thres, num_classes, 
         synset_names, iou_thres_list, degree_thres_list, shift_thres_list, use_matches_for_pose,
         iou_pose_thres,
         res):
    gt_class_ids = res['gt_class_ids'].astype(np.int32)
    gt_obj_ids = res['gt_obj_ids'].astype(np.int32)
    # normalize RTs and scales
    gt_RTs = np.array(res['gt_RTs'])
    gt_scales = np.array(res['gt_scales'])
    gt_handle_visibility = res['gt_handle_visibility']
    if len(gt_RTs) > 0:
        norm_gt_scales = np.stack([np.cbrt(np.linalg.det(gt_RT[:3, :3])) for gt_RT in gt_RTs])
        gt_RTs[:, :3, :3] = gt_RTs[:, :3, :3] / (norm_gt_scales[:, None, None] + 1e-7)
        gt_scales = gt_scales * norm_gt_scales[:, None]

    pred_bboxes = np.array(res['pred_bboxes'])
    pred_class_ids = res['pred_class_ids']
    pred_scales = res['pred_scales']
    pred_scores = res['pred_scores']
    pred_RTs = np.array(res['pred_RTs'])
    pred_bboxes[...] = 1
    
    if len(pred_RTs) > 0:
        norm_pred_scales = np.stack([np.cbrt(np.linalg.det(pred_RT[:3, :3])) for pred_RT in pred_RTs])
        pred_RTs[:, :3, :3] = pred_RTs[:, :3, :3] / (norm_pred_scales[:, None, None] + 1e-7)
        pred_scales = pred_scales * norm_pred_scales[:, None]

    iou_pred_matches_worker = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_pred_scores_worker  = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_gt_matches_worker   = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    
    pose_pred_matches_worker = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
    pose_gt_matches_worker  = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
    pose_pred_scores_worker = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
    
    
    if len(gt_class_ids) != 0 or len(pred_class_ids) != 0:

        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            if len(gt_class_ids) > 0:
                gt_idx_mapping = dict([(i, j) for i, j in enumerate(np.where(gt_class_ids==cls_id)[0])])
            else:
                gt_idx_mapping = dict([(i, j) for i, j in enumerate(range(20))])
            cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_obj_ids = gt_obj_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_scales = gt_scales[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
            cls_gt_RTs = gt_RTs[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))

            if len(pred_class_ids) > 0:
                pred_idx_mapping = dict([(i, j) for i, j in enumerate(np.where(pred_class_ids==cls_id)[0])])
            else:
                pred_idx_mapping = dict([(i, j) for i, j in enumerate(range(20))])
            cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_bboxes =  pred_bboxes[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4))
            cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_scales = pred_scales[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))

            # calculate the overlap between each gt instance and pred instance
            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)

            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = compute_3d_matches(cls_gt_class_ids, cls_gt_obj_ids, cls_gt_RTs, cls_gt_scales, cls_gt_handle_visibility, synset_names,
                                                                                            cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
                                                                                            iou_thres_list)
            if len(iou_pred_indices):
                pred_idx_mapping = dict([(i, pred_idx_mapping[j]) for i, j in enumerate(iou_pred_indices)])
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
                cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]


            iou_pred_matches_worker[cls_id] = np.concatenate((iou_pred_matches_worker[cls_id], iou_cls_pred_match), axis=-1)
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            iou_pred_scores_worker[cls_id] = np.concatenate((iou_pred_scores_worker[cls_id], cls_pred_scores_tile), axis=-1)
            assert iou_pred_matches_worker[cls_id].shape[1] == iou_pred_scores_worker[cls_id].shape[1]
            iou_gt_matches_worker[cls_id] = np.concatenate((iou_gt_matches_worker[cls_id], iou_cls_gt_match), axis=-1)

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)

                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]

                if len(iou_thres_pred_match) > 0 and pred_idx_mapping is not None:
                    pred_idx_mapping = dict([(i, pred_idx_mapping[j]) for i, j in enumerate(np.where(iou_thres_pred_match > -1)[0])])
                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_bboxes = cls_pred_bboxes[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4))

                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                
                if len(iou_thres_gt_match) > 0 and gt_idx_mapping is not None:
                    gt_idx_mapping = dict([(i, gt_idx_mapping[j]) for i, j in enumerate(np.where(iou_thres_gt_match > -1)[0])])
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_obj_ids = cls_gt_obj_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)

            # if cls_id == 4:
            #     print(cls_pred_RTs, cls_gt_RTs)
            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_obj_ids, cls_gt_RTs, cls_gt_handle_visibility, 
                                                cls_pred_class_ids, cls_pred_RTs,
                                                synset_names)


            pose_cls_gt_match, pose_cls_pred_match = compute_match_from_degree_cm(RT_overlaps, 
                                                                                    cls_pred_class_ids, 
                                                                                    cls_gt_class_ids, 
                                                                                    degree_thres_list, 
                                                                                    shift_thres_list)
            # for i in range(pose_cls_pred_match.shape[2]):
            #     pose_pred_matches[:, :, 0, pred_idx_mapping[i]] = np.vectorize(lambda k: gt_idx_mapping[k] if k != -1 else -1)(pose_cls_pred_match[:, :, i])
            # for i in range(pose_cls_gt_match.shape[2]):
            #     pose_gt_matches[:, :, 0, gt_idx_mapping[i]] = np.vectorize(lambda k: pred_idx_mapping[k] if k != -1 else -1)(pose_cls_gt_match[:, :, i])
            pose_pred_matches_worker[cls_id] = np.concatenate((pose_pred_matches_worker[cls_id], pose_cls_pred_match), axis=-1)
            
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            pose_pred_scores_worker[cls_id]  = np.concatenate((pose_pred_scores_worker[cls_id], cls_pred_scores_tile), axis=-1)
            assert pose_pred_scores_worker[cls_id].shape[2] == pose_pred_matches_worker[cls_id].shape[2], '{} vs. {}'.format(pose_pred_scores_worker[cls_id].shape, pose_pred_matches_worker[cls_id].shape)
            pose_gt_matches_worker[cls_id] = np.concatenate((pose_gt_matches_worker[cls_id], pose_cls_gt_match), axis=-1)
    return (iou_pred_matches_worker, iou_pred_scores_worker, iou_gt_matches_worker, \
        pose_pred_matches_worker, pose_pred_scores_worker, pose_gt_matches_worker)
        

def compute_degree_cm_mAP(final_results, synset_names, log_dir, 
                          degree_thresholds=[360], shift_thresholds=[100], iou_3d_thresholds=[0.1], 
                          iou_pose_thres=0.1, use_matches_for_pose=False, num_proc=10):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
    
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)

    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)

    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [[] for _ in range(num_classes)]
    iou_pred_scores_all  = [[] for _ in range(num_classes)]
    iou_gt_matches_all   = [[] for _ in range(num_classes)]
    
    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [[] for _  in range(num_classes)]
    pose_gt_matches_all  = [[] for _  in range(num_classes)]
    pose_pred_scores_all = [[] for _  in range(num_classes)]

    from tqdm import tqdm
    from functools import partial
    import multiprocessing
    pool = Pool(processes=num_proc)
    for worker_res in tqdm(pool.imap_unordered(partial(work, num_iou_thres, num_degree_thres, num_shift_thres, num_classes, 
         synset_names, iou_thres_list, degree_thres_list, shift_thres_list, use_matches_for_pose,
         iou_pose_thres), final_results, chunksize=num_proc), total=len(final_results)):
        
        for cls_id in range(1, num_classes):
            
            iou_pred_matches_all[cls_id].append(worker_res[0][cls_id])
            iou_pred_scores_all[cls_id].append(worker_res[1][cls_id])
            iou_gt_matches_all[cls_id].append(worker_res[2][cls_id])
            pose_pred_matches_all[cls_id].append(worker_res[3][cls_id])
            pose_pred_scores_all[cls_id].append(worker_res[4][cls_id])
            pose_gt_matches_all[cls_id].append(worker_res[5][cls_id])
            
    for cls_id in range(1, num_classes):
        iou_pred_matches_all[cls_id] = np.concatenate(iou_pred_matches_all[cls_id], -1)
        iou_pred_scores_all[cls_id] = np.concatenate(iou_pred_scores_all[cls_id], -1)
        iou_gt_matches_all[cls_id] = np.concatenate(iou_gt_matches_all[cls_id], -1)
        pose_pred_matches_all[cls_id] = np.concatenate(pose_pred_matches_all[cls_id], -1)
        pose_pred_scores_all[cls_id] = np.concatenate(pose_pred_scores_all[cls_id], -1)
        pose_gt_matches_all[cls_id] = np.concatenate(pose_gt_matches_all[cls_id], -1)
            
    if log_dir is not None:
        fig_iou = plt.figure()
        ax_iou = plt.subplot(111)
        plt.ylabel('AP')
        plt.ylim((0, 1))
        plt.xlabel('3D IoU thresholds')
        
        iou_output_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.png'.format(iou_thres_list[0], iou_thres_list[-1]))
        iou_dict_pkl_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.pkl'.format(iou_thres_list[0], iou_thres_list[-1]))

    iou_dict = {}
    iou_dict['thres_list'] = iou_thres_list
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        # print(class_name)
        for s, iou_thres in enumerate(iou_thres_list):
            iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
                                                                iou_pred_scores_all[cls_id][s, :],
                                                                iou_gt_matches_all[cls_id][s, :])
        if log_dir is not None:
            ax_iou.plot(iou_thres_list, iou_3d_aps[cls_id, :], label=class_name)
        
    iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-1, :], axis=0)
    
    if log_dir is not None:
        ax_iou.plot(iou_thres_list, iou_3d_aps[-1, :], label='mean')
        ax_iou.legend()
        fig_iou.savefig(iou_output_path)
        plt.close(fig_iou)

    iou_dict['aps'] = iou_3d_aps
    
    if log_dir is not None:
        with open(iou_dict_pkl_path, 'wb') as f:
            cPickle.dump(iou_dict, f)

    # draw pose AP vs. thresholds
    if use_matches_for_pose:
        prefix='Pose_Only_'
    else:
        prefix='Pose_Detection_'

    if log_dir is not None:
        pose_dict_pkl_path = os.path.join(log_dir, prefix+'AP_{}-{}degree_{}-{}cm.pkl'.format(degree_thres_list[0], degree_thres_list[-2], 
                                                                                            shift_thres_list[0], shift_thres_list[-2]))
    pose_dict = {}
    pose_dict['degree_thres'] = degree_thres_list
    pose_dict['shift_thres_list'] = shift_thres_list

    for i, degree_thres in enumerate(degree_thres_list):                
        for j, shift_thres in enumerate(shift_thres_list):
            # print(i, j)
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

                pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(cls_pose_pred_matches_all, 
                                                                        cls_pose_pred_scores_all, 
                                                                        cls_pose_gt_matches_all)

            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])
    
    pose_dict['aps'] = pose_aps
    
    if log_dir is not None:
        with open(pose_dict_pkl_path, 'wb') as f:
            cPickle.dump(pose_dict, f)


    if log_dir is not None:
        for cls_id in range(1, num_classes):
            class_name = synset_names[cls_id]
            fig_iou = plt.figure()
            ax_iou = plt.subplot(111)
            plt.ylabel('Rotation thresholds/degree')
            plt.xlabel('translation/cm')
            plt.imshow(pose_aps[cls_id, :-1, :-1][::-1], cmap='jet', interpolation='bilinear', extent=[shift_thres_list[0], shift_thres_list[-2], degree_thres_list[0], degree_thres_list[-2]])

            output_path = os.path.join(log_dir, prefix+'AP_{}_{}-{}degree_{}-{}cm.png'.format(class_name, 
                                                                                    degree_thres_list[0], degree_thres_list[-2], 
                                                                                    shift_thres_list[0], shift_thres_list[-2]))
            plt.colorbar()
            plt.savefig(output_path)
            plt.close(fig_iou)
    
        fig_pose = plt.figure()
        ax_pose = plt.subplot(111)
        plt.ylabel('Rotation thresholds/degree')
        plt.xlabel('translation/cm')
        plt.imshow(pose_aps[-1, :-1, :-1][::-1], cmap='jet', interpolation='bilinear', extent=[shift_thres_list[0], shift_thres_list[-2], degree_thres_list[0], degree_thres_list[-2]])
        output_path = os.path.join(log_dir, prefix+'mAP_{}-{}degree_{}-{}cm.png'.format(degree_thres_list[0], degree_thres_list[-2], 
                                                                                shift_thres_list[0], shift_thres_list[-2]))
        plt.colorbar()
        plt.savefig(output_path)
        plt.close(fig_pose)

        
        fig_rot = plt.figure()
        ax_rot = plt.subplot(111)
        plt.ylabel('AP')
        plt.ylim((0, 1.05))
        plt.xlabel('translation/cm')
        for cls_id in range(1, num_classes):
            class_name = synset_names[cls_id]
            ax_rot.plot(shift_thres_list[:-1], pose_aps[cls_id, -1, :-1], label=class_name)
        
        ax_rot.plot(shift_thres_list[:-1], pose_aps[-1, -1, :-1], label='mean')
        output_path = os.path.join(log_dir, prefix+'mAP_{}-{}cm.png'.format(shift_thres_list[0], shift_thres_list[-2]))
        ax_rot.legend()
        fig_rot.savefig(output_path)
        plt.close(fig_rot)

        fig_trans = plt.figure()
        ax_trans = plt.subplot(111)
        plt.ylabel('AP')
        plt.ylim((0, 1.05))

        plt.xlabel('Rotation/degree')
        for cls_id in range(1, num_classes):
            class_name = synset_names[cls_id]
            ax_trans.plot(degree_thres_list[:-1], pose_aps[cls_id, :-1, -1], label=class_name)

        ax_trans.plot(degree_thres_list[:-1], pose_aps[-1, :-1, -1], label='mean')
        output_path = os.path.join(log_dir, prefix+'mAP_{}-{}degree.png'.format(degree_thres_list[0], degree_thres_list[-2]))
        
        ax_trans.legend()
        fig_trans.savefig(output_path)
        plt.close(fig_trans)

    return iou_3d_aps, pose_aps
        
from PIL import Image
from torchvision.transforms import functional
def resize_crop(img, padding=0.2, out_size=224, bbox=None):
    # return np.array(img), np.eye(3)
    img = Image.fromarray(img)
    if bbox is None:
        bbox = img.getbbox()
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    size = max(height, width) * (1 + padding)
    center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    bbox_enlarged = center[0] - size / 2, center[1] - size / 2, \
        center[0] + size / 2, center[1] + size / 2
    img = functional.resize(functional.crop(img, bbox_enlarged[1], bbox_enlarged[0], size, size), (out_size, out_size))
    transform = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1.]])  \
        @ np.array([[size / out_size, 0, 0], [0, size / out_size, 0], [0, 0, 1]]) \
        @ np.array([[1, 0, -out_size / 2], [0, 1, -out_size / 2], [0, 0, 1.]])
    return np.array(img), transform


def draw(img, imgpts, axes, color, size):
    imgpts = np.int32(imgpts).reshape(-1, 2)


    # draw ground layer in darker color
    color_ground = (int(color[0]), int(color[1]), int(color[2]))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)


    # draw pillars in blue color
    color_pillar = (int(color[0]), int(color[1]), int(color[2]))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    
    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)


    # draw axes
    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), size)  # z
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), size)  # x
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), size) ## y last


    return img

def draw_pose(img, intrinsics, rot, center, scale, color=(255, 0, 0), out_size=256, bbox=None):
    mat = np.eye(4)
    scale_norm = np.linalg.norm(scale)
    mat[:3, :3] = rot * scale_norm
    mat[:3, -1] = center
    
    xyz_axis = 0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    transformed_axes = transform_coordinates_3d(xyz_axis, mat)
    projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

    bbox_3d = get_3d_bbox(scale / scale_norm, 0)
    transformed_bbox_3d = transform_coordinates_3d(bbox_3d, mat)
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
    if bbox is None:
        tl_corner = projected_bbox.min(0)
        br_corner = projected_bbox.max(0)
    else:
        tl_corner = np.array([bbox[0], bbox[1]])
        br_corner = np.array([bbox[2], bbox[3]])
    
    draw_image_bbox = draw(img, projected_bbox, projected_axes, color, size=(br_corner - tl_corner).max() // 30)
    if bbox is None:
        bbox = (tl_corner[0], tl_corner[1], br_corner[0], br_corner[1])
    draw_image_bbox = resize_crop(draw_image_bbox, padding=1.0, out_size=out_size, bbox=bbox)[0]
    return draw_image_bbox, bbox