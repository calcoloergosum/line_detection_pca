"""
This code is written by Han; A.I. System Research, Japan.
This code reproduces the results of
"A straight line detection using principal component analysis (2006)" by Yun-Seok Lee, Han-Suh Koo, Chang-Sung Jeong
(https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.1043&rep=rep1&type=pdf)
"""
import cv2
import numpy as np
from typing import Tuple, Sequence, List, Iterable, Iterator, Set, Optional
from unionfind.unionfind import UnionFind
from itertools import chain

# Point in pixels
Point2i = Tuple[int, int]

# Line segment in pixels; start point and end point
Segment2i = Tuple[Point2i, Point2i]


class PointCollection:
    def __init__(self, pts: List[Point2i]):
        self.pts = np.array(pts, dtype=np.int)
        self.scatter_matrix = self._scatter_matrix(self.pts)
        eig_max, eig_min, angle = self._eig_angle_2x2(self.scatter_matrix)
        self.eig_max = eig_max
        self.eig_min = eig_min
        self.angle = angle
        self._len = None  # type: Optional[float]

    @staticmethod
    def _scatter_matrix(pts: np.ndarray) -> np.ndarray:
        assert len(pts) >= 2, "Undefined for just a point"
        pt_mean = pts.mean(axis=0)
        devs = pts - pt_mean
        s11 = (devs[:, 0] ** 2).mean()
        s21 = s12 = (devs[:, 0] * devs[:, 1]).mean()
        s22 = (devs[:, 1] ** 2).mean()
        mat = np.array([
            [s11, s12],
            [s21, s22],
        ], dtype=np.float)
        return mat

    @staticmethod
    def _eig_angle_2x2(mat: np.ndarray) -> Tuple[float, float, float]:
        assert mat.shape == (2, 2)
        s11, s12, s21, s22 = mat.flatten()
        a = s11 + s22
        b = np.sqrt((s11 - s22) ** 2 + 4 * s12 * s21)
        eig_max = 0.5 * (a + b)
        eig_min = 0.5 * (a - b)
        angle = np.arctan2(eig_max - s11, s12)
        return eig_max, eig_min, angle

    def length(self) -> float:
        if self._len is None:
            xmin = self.pts[:, 0].min()
            xmax = self.pts[:, 0].max()
            ymin = self.pts[:, 1].min()
            ymax = self.pts[:, 1].max()
            self._len = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
        return self._len

    def is_line(self, thresh_absolute: float) -> bool:
        # thres = thresh_absolute * (self.eig_max / len(self)) ** 2
        thres = thresh_absolute * (self.length() / 30) ** 2
        if self.eig_min > thres:
            return False
        return True

    def as_point_pair(self) -> Segment2i:
        # length as bbox

        center = self.pts.mean(axis=0)
        cos = np.cos(self.angle)
        sin = np.sin(self.angle)
        affine = np.array([
            [cos, -sin, center[0]],
            [sin, cos, center[1]],
        ], dtype=np.float)
        ends = np.array([
            [-self.length() / 2, 0, 1],
            [self.length() / 2, 0, 1],
        ], dtype=np.float)
        (x1, y1), (x2, y2) = (affine @ ends.T).T
        return ((int(x1), int(y1)), (int(x2), int(y2)))


def detect_lines(im_bgr: np.ndarray, thresh_absolute: float = 0.3, min_pt_num: int = 10) -> List[PointCollection]:
    assert im_bgr.ndim == 3, "Only supports BGR image"
    assert im_bgr.dtype == np.uint8,  "Only supports uint8 image"
    h, w = im_bgr.shape[:2]
    assert h > 0 and w > 0, "Image size invalid"

    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    im_canny = cv2.Canny(im_gray, 150, 200)
    cv2.imwrite('canny.png', 255 - im_canny)

    # initial labelling
    is_row, is_col, is_single, is_cross = label_edge_points(im_canny)

    cv2.imwrite('is_row.png', 255 - np.uint8(is_row) * 255)
    cv2.imwrite('is_col.png', 255 - np.uint8(is_col) * 255)
    cv2.imwrite('is_single.png', 255 - np.uint8(is_single) * 255)
    cv2.imwrite('is_cross.png', 255 - np.uint8(is_cross) * 255)

    # start labelling

    # Row labelling by connectivity
    is_columnless = (im_canny > 0) * (~is_col) + is_cross  # exclude column, include cross
    cv2.imwrite('is_columnless.png', 255 - np.uint8(is_columnless) * 255)
    row_pts_list = cluster_by_connectivity(is_columnless)

    # Column labelling by connectivity
    is_rowless = (im_canny > 0) * (~is_row) + is_cross  # exclude row, include cross
    cv2.imwrite('is_rowless.png', 255 - np.uint8(is_rowless) * 255)
    col_pts_list = cluster_by_connectivity(is_rowless)

    # TODO: Remove single only cluster from col_pts_list

    combined_pts_list = chain(col_pts_list, row_pts_list)
    filtered_pt_collections = filter_pts_list(combined_pts_list, thresh_absolute, min_pt_num)
    return list(filtered_pt_collections)


def filter_pts_list(pts_list: Iterable[List[Point2i]], thresh_absolute: float, min_pt_num: int) -> Iterator[PointCollection]:
    for pts in pts_list:
        if len(pts) < min_pt_num:
            continue
        pt_collection = PointCollection(pts)
        if pt_collection.is_line(thresh_absolute):
            yield pt_collection


def cluster_by_connectivity(im: np.ndarray) -> List[List[Point2i]]:
    assert im.ndim == 2
    adj_dxdys = [(0, 1), (1, -1), (1, 0), (1, 1)]
    # we skip [(-1, -1), (-1, 0), (-1, 1), (0, -1)], as it is symmetric relationship
    ys, xs = np.nonzero(im)
    pts = list(zip(xs, ys))

    disjoint_set = UnionFind(range(len(pts)))
    index_dict = {
        pt: i
        for i, pt in enumerate(pts)
    }
    # iterate through all points
    h, w = im.shape
    for x, y in pts:
        # add adjacency
        for dx, dy in adj_dxdys:
            if not(0 <= x + dx < w):
                continue
            if not (0 <= y + dy < h):
                continue
            if im[y + dy, x + dx]:
                disjoint_set.union(index_dict[(x, y)], index_dict[(x + dx, y + dy)])
    index_sets = disjoint_set.components()
    points_list = [
        [
            pts[ind]
            for ind in index_set
        ]
        for index_set in index_sets
    ]
    return points_list


def label_edge_points(im_canny: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = im_canny.shape
    im_xshift = np.zeros((h, w))
    im_yshift = np.zeros((h, w))
    im_xshift[:, 1:] = im_canny[:, :-1]
    im_yshift[1:, :] = im_canny[:-1, :]

    # prepare sets
    is_edge = im_canny > 0
    is_row_pixel = (im_xshift > 0) * is_edge
    is_col_pixel = (im_yshift > 0) * is_edge
    is_cross_pixel = is_col_pixel * is_row_pixel
    is_single_pixel = ~is_row_pixel * ~is_col_pixel * is_edge
    assert (is_col_pixel * is_cross_pixel == is_cross_pixel).all()
    assert (is_row_pixel * is_cross_pixel == is_cross_pixel).all()
    assert not np.any(is_cross_pixel * is_single_pixel)
    assert ((is_row_pixel + is_col_pixel + is_cross_pixel) == is_edge).all() == False, "No single pixels"
    assert ((is_row_pixel + is_col_pixel + is_cross_pixel + is_single_pixel) == is_edge).all()
    return is_row_pixel, is_col_pixel, is_single_pixel, is_cross_pixel
