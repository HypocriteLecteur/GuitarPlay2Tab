import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from itertools import combinations
from utils.utils import angle_diff, line_line_intersection, ColorThresholder, prepare_draw_fretboard, prepare_draw_strings, find_features, cost_matrix_for_features, homogeneous_coords_from_houghlines, ransac_find_vanishing_point, project_points_to_line, find_projective_rectification
import matplotlib.pyplot as plt
from template import Template, transform_houghlines, transform_points
from itertools import combinations
from utils.utils import get_screensize, draw_line, cornerpoints_from_line_pair, crop_from_oriented_bounding_box, npprint, find_homography_from_matched_fretlines, ransac_vanishing_point_estimation
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import RANSACRegressor, LinearRegression
# from skimage.measure import ransac
from skimage.morphology import skeletonize
from itertools import product
import time
from sklearn import datasets, svm
from utils.utils_debug import VideoBuffer
from skimage.transform import SimilarityTransform
from skimage.measure import ransac
from imutils.object_detection import non_max_suppression
import math
from skimage.draw import line as skimage_line

from scipy.signal import find_peaks
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

def kb_from_lineP(lineP):
    return (lineP[1] - lineP[3])/(lineP[0] - lineP[2]), (lineP[0]*lineP[3] - lineP[2]*lineP[1])/(lineP[0] - lineP[2])

def init_group_points(bundler, pnts, linesP, dist_thres=3):
    labels = -1 * np.ones_like(pnts[:, 0])
    for i, pnt in enumerate(pnts):
        for j, lineP in enumerate(linesP):
            if bundler.distance_point_to_line(pnt, lineP) < dist_thres:
                labels[i] = j
                break
    return labels.astype(int)

def refine_group_points(labels, peaks, linesP, dist_thres=3):
    unique, counts = np.unique(labels, return_counts=True)
    counts = counts[unique > -1]
    unique = unique[unique > -1]

    while unique.size > 0:
        idx_ = np.argmax(counts)
        label_ = unique[idx_]
        counts = np.delete(counts, idx_)
        unique = np.delete(unique, idx_)

        not_labelled = np.where(labels == -1)[0]
        # k, b = kb_from_lineP(linesP[label_])
        k, b = np.linalg.lstsq(np.vstack((peaks[labels == label_, 0], np.ones_like(peaks[labels == label_, 0]))).T, peaks[labels == label_, 1])[0]
        inliers = np.abs(k * peaks[not_labelled, 0] + b - peaks[not_labelled, 1]) < dist_thres
        while np.sum(inliers) > 0:
            labels[not_labelled[inliers]] = label_
            not_labelled = np.where(labels == -1)[0]
            k, b = np.linalg.lstsq(np.vstack((peaks[labels == label_, 0], np.ones_like(peaks[labels == label_, 0]))).T, peaks[labels == label_, 1])[0]
            inliers = np.abs(k * peaks[not_labelled, 0] + b - peaks[not_labelled, 1]) < dist_thres
    return labels

def merge_group_points(labels, peaks, linesP, merge_percentage=0.5, dist_thres=3):
    unique, counts = np.unique(labels, return_counts=True)
    counts = counts[unique > -1]
    unique = unique[unique > -1]
    
    while unique.size > 0:
        idx_ = np.argmax(counts)
        label_ = unique[idx_]
        counts = np.delete(counts, idx_)
        unique = np.delete(unique, idx_)

        current_pnts = labels == label_
        k, b = np.linalg.lstsq(np.vstack((peaks[current_pnts, 0], np.ones_like(peaks[current_pnts, 0]))).T, peaks[current_pnts, 1])[0]
        inliers = np.abs(k * peaks[current_pnts == False, 0] + b - peaks[current_pnts == False, 1]) < dist_thres
        if np.sum(inliers) > 0:
            other_pnts_inliers = np.arange(labels.size)[current_pnts == False][inliers]
            unique_other, counts_other = np.unique(labels[other_pnts_inliers], return_counts=True)
            merge_labels = []
            for unique_other_, counts_other_ in zip(unique_other, counts_other):
                if counts_other_ >= counts[unique == unique_other_] * merge_percentage:
                    merge_labels.append(unique_other_)
            if len(merge_labels) > 0:
                for merge_label in merge_labels:
                    labels[labels == merge_label] = label_
                    counts = np.delete(counts, np.where(unique == merge_label)[0][0])
                    unique = np.delete(unique, np.where(unique == merge_label)[0][0])
    return labels

class HoughBundler:     
    def __init__(self,min_distance=5,min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle
    
    def get_orientation(self, line):
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])
      
        if(len(lines) == 1):
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0],points[-1]]])

    def process_lines(self, lines):
        lines_horizontal  = []
        lines_vertical  = []
  
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical  = sorted(lines_vertical , key=lambda line: line[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)
                    
        return np.asarray(merged_lines_all)


def line_segments_intersection(ls1, ls2):
    '''
    Two line segments represented as p + r and q + s
    if we can find t and u that are in [0, 1] such that
    p + tr = q + us, there is an intersection.
    We assume lines are not parallel or collinear
    '''

    p = ls1[:2]
    r = ls1[2:] - p
    q = ls2[:2]
    s = ls2[2:] - q

    t = np.cross(q - p, s) / np.cross(r, s)
    u = np.cross(q - p, r) / np.cross(r, s)
    if t >= 0 and t <= 1 and u >= 0 and u <= 1:
        return p + t * r
    else:
        return None

def line_segments_intersection_batch(ls1, batch_ls2):
    '''
    Two line segments represented as p + r and q + s
    if we can find t and u that are in [0, 1] such that
    p + tr = q + us, there is an intersection.
    We assume lines are not parallel or collinear
    '''
    p = ls1[:2]
    r = ls1[2:] - p
    q = batch_ls2[:, :2]
    s = batch_ls2[:, 2:] - q

    t = np.cross(q - p, s) / np.cross(r, s)
    u = np.cross(q - p, r) / np.cross(r, s)

    idx = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    if np.sum(idx) > 0:
        return p.reshape((1, 2)) + t[idx].reshape((-1, 1)) * r.reshape((1, 2))
    else:
        return None

from copy import copy
from numpy.random import default_rng
rng = default_rng()
class RANSAC:
    def __init__(self, n=3, k=200, t=10, d=7, model=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.best_fit = None
        self.best_error = np.inf
        self.best_inlier = None

    def fit(self, X, y, linesP):
        for _ in range(self.k):
            ids = rng.permutation(X.shape[0])

            maybe_inliers = ids[: self.n]
            maybe_model = self.model.minial_fit(X[maybe_inliers], y[maybe_inliers])

            thresholded = self.model.loss(maybe_model, X[ids][self.n :], y[ids][self.n :], linesP[ids][self.n :]) < self.t

            inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()]

            if inlier_ids.size > self.d:
                inlier_points = np.hstack([maybe_inliers, inlier_ids])
                better_model = self.model.ls_fit(X[inlier_points], y[inlier_points])

                this_error = np.sum(self.model.loss(better_model, X[inlier_points], y[inlier_points], linesP[inlier_points]))
                
                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = better_model
                    tmp = np.zeros((X.shape[0],)).astype(bool)
                    tmp[inlier_points] = True
                    self.best_inlier = tmp
        return self

class VanishingLineEstimator:
    def minial_fit(self, L, N):
        '''
        L = numpy (3, 3)
        N = numpy (3, 1) 
        '''
        vanishing_line = estimate_vanishing_line_from_three_lines(L[0], N[0], L[1], N[1], L[2], N[2])
        return (vanishing_line, L[0], N[0])
    
    def ls_fit(self, L, N):
        return (vanishing_line_from_parallel_lines(L, N), L[0], N[0])
    
    def loss(self, model, L, N, linesP):
        vanishing_line, Li, Ni = model

        # L_guess = Li.reshape((1, 3)) + (N - Ni).reshape((-1, 1)) * vanishing_line.reshape((1, 3))
        err = np.zeros((L.shape[0],1))
        for i in range(L.shape[0]):
            L0 = Li + (N[i] - Ni) * vanishing_line
            err[i] = (np.abs(np.dot(L0, [linesP[i][0], linesP[i][1], 1])) + np.abs(np.dot(L0, [linesP[i][2], linesP[i][3], 1]))) / np.linalg.norm(L0[:2])
        return err


def estimate_vanishing_line_from_three_lines(li, ni, lj, nj, lk, nk):
    alpha = - 1 / np.linalg.norm(np.cross(lj, lk))**2 / (nk - ni) * np.dot(np.cross(li, lk), np.cross(lj, lk))
    beta = - 1 / np.linalg.norm(np.cross(lj, lk))**2 / (nj - ni) * np.dot(np.cross(li, lj), np.cross(lk, lj))
    return alpha * lj + beta * lk

def houghline_from_kb(slope, intercept):
    return np.array([-intercept/np.sqrt(slope**2 + 1), np.arctan2(-1/np.sqrt(slope**2 + 1), slope/np.sqrt(slope**2 + 1))])

def construct_KeyLine(image, start_point, end_point):
    L1 = cv2.line_descriptor.KeyLine()
    L1.startPointX = start_point[0]
    L1.startPointY = start_point[1]
    L1.endPointX = end_point[0]
    L1.endPointY = end_point[1]
    L1.sPointInOctaveX = L1.startPointX;
    L1.sPointInOctaveY = L1.startPointY;
    L1.ePointInOctaveX = L1.endPointX;
    L1.ePointInOctaveY = L1.endPointY;
    L1.lineLength = np.linalg.norm(end_point - start_point)
    L1.angle = np.arctan2((L1.endPointY - L1.startPointY), (L1.endPointX - L1.startPointX))
    L1.size = (L1.endPointX - L1.startPointX) * (L1.endPointY - L1.startPointY)
    L1.response = L1.lineLength / max(image.shape)
    L1.pt = [(L1.endPointX + L1.startPointX) / 2, (L1.endPointY + L1.startPointY) / 2]
    L1.numOfPixels = createLineIterator(start_point.astype(int), end_point.astype(int), image).shape[0]
    return L1

def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer

def vanishing_line_from_parallel_lines(lines_coords, spacing):
    A = np.zeros((2*lines_coords.shape[0], 6))
    for i in range(lines_coords.shape[0]):
        A[2*i, 2] = -lines_coords[i, 2] * spacing[i]
        A[2*i, 3] = -lines_coords[i, 2]
        A[2*i, 4] = lines_coords[i, 1] * spacing[i]
        A[2*i, 5] = lines_coords[i, 1]
        A[2*i+1, 0] = lines_coords[i, 2] * spacing[i]
        A[2*i+1, 1] = lines_coords[i, 2]
        A[2*i+1, 4] = -lines_coords[i, 0] * spacing[i]
        A[2*i+1, 5] = -lines_coords[i, 0]
    a = np.linalg.svd(A)[2][-1]
    vanishing_line = a[[0, 2, 4]]
    vanishing_line = vanishing_line / vanishing_line[2]
    return vanishing_line

def hough_line_from_two_points(point1, point2):
    angle = np.arctan2(point2[0] - point1[0], point1[1] - point2[1])
    dist = point2[0] * np.cos(angle) + point2[1] * np.sin(angle)
    return np.array([dist, angle])

def linesP_to_houghlines(linesP, sort=True):
    angles = np.arctan2(linesP[:, 0, 2] - linesP[:, 0, 0], linesP[:, 0, 1] - linesP[:, 0, 3])
    dists = linesP[:, 0, 2] * np.cos(angles) + linesP[:, 0, 3] * np.sin(angles)
    angles[dists < 0] = angles[dists < 0] - np.sign(angles[dists < 0]) * np.pi
    dists[dists < 0] = -dists[dists < 0]

    if sort:
        angles, dists = (list(item) for item in zip(*sorted(zip(angles, dists), key=lambda x: x[1], reverse=False)))
    angles = np.array(angles)
    dists = np.array(dists)
    return np.concatenate((dists[:, np.newaxis], angles[:, np.newaxis]), axis=1)

def get_longest_consecutive_condition_subset(input_list: list, condition) -> list:

    max_subset = []
    current_max_subset = []
    current_index = -1
    flag = False

    for i, number in enumerate(input_list):
        if condition(number):
            if not flag:
                current_index = i
                flag = True
            current_max_subset.append(number)
        else:
            flag = False
            if len(current_max_subset) > len(max_subset):
                max_subset = current_max_subset
            current_max_subset = []
    if flag and len(current_max_subset) > len(max_subset):
        max_subset = current_max_subset
    
    if current_index == -1:
        return 0, []
    return current_index, max_subset

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def show_image_cube(img):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = img.reshape((-1, 3))
    # mask = data[:, 0] != 0
    ax.scatter(data[0:-1:50, 0],data[0:-1:50, 1],data[0:-1:50, 2], c=data[0:-1:50, :] / 255, marker='.')
    # ax.scatter(data[mask[0:-1:50], 0],data[mask[0:-1:50], 1],data[mask[0:-1:50], 2], c=data[mask[0:-1:50], :] / 255, marker='.')
    ax.xlim = [0, 255]
    ax.ylim = [0, 255]
    ax.zlim = [0, 255]
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

def show_multiple_images_cube(*args):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for image in args:
        data = image.reshape((-1, 3))
        ax.scatter(data[0:-1:50, 0],data[0:-1:50, 1],data[0:-1:50, 2], marker='.')
    ax.xlim = [0, 255]
    ax.ylim = [0, 255]
    ax.zlim = [0, 255]
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

def show_multiple_images_2d(model, *args):
    fig, ax = plt.subplots()
    for data in args:
        ax.scatter(data[:, 0],data[:, 1], marker='.')
    ax.xlim = [0, 255]
    ax.ylim = [0, 255]
    abline(-model.coef_[0, 0] / model.coef_[0, 1], -model.intercept_ / model.coef_[0, 1])
    ax.set_xlabel('R')
    ax.set_ylabel('G')

def kmeans_color_quantization(image, clusters=8, rounds=1):
    samples = image.reshape((-1, 3)).astype(np.float32)

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape)), labels.flatten().reshape((image.shape[:2])), centers

class FretboardDetector():
    def __init__(self) -> None:
        self.counter = 1

        self.screensize = get_screensize()
        # self.colorthresholder = ColorThresholder(self.screensize)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        self.is_initialized = False

        self.template = Template()

        self.tracked_marker_positions = None
        self.tracked_fretlines = None
        self.fretline_to_marker_dist = None

        self.oriented_bounding_box = None

        self.marker_areas = None

        self.videoplayback = VideoBuffer()
        
        self.string_template = cv2.cvtColor(cv2.imread('test\\string_template.png'), cv2.COLOR_BGR2GRAY)
        self.string_template2 = cv2.cvtColor(cv2.imread('test\\string_template_2.png'), cv2.COLOR_BGR2GRAY)

        self.backSub = cv2.createBackgroundSubtractorMOG2()

        # BaseOptions = mp.tasks.BaseOptions
        # GestureRecognizer = mp.tasks.vision.GestureRecognizer
        # GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        # VisionRunningMode = mp.tasks.vision.RunningMode

        # options = GestureRecognizerOptions(
        # base_options=BaseOptions(model_asset_path='play2tab/video_processing/utils/gesture_recognizer.task'),
        # running_mode=VisionRunningMode.IMAGE)

        # self.recognizer = GestureRecognizer.create_from_options(options)
    
    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = self.clahe.apply(gray)
        return gray
    
    def fingerboard_lines_detection(self, edges, theta_tol=9, rho_tol=50, is_visualize=False):
        '''
        fingerboard_lines = [[r1, t1, h1], [r2, t2, h2]]
        '''
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 280, None, 0, 0)

        if lines is None:
            return None, None
        
        max_dist = 0
        fingerboard_lines = None
        for [[ri, ti], [rj, tj]] in combinations(zip(lines[:, 0, 0], lines[:, 0, 1]), 2):
            if angle_diff(ti, tj) < theta_tol / 180 * np.pi and abs(ri - rj) > rho_tol and abs(ri - rj) > max_dist:
                fingerboard_lines = np.array([[ri, ti], [rj, tj]])
                max_dist = abs(ri - rj)

        if is_visualize and fingerboard_lines is not None:
            cdst = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
            for line in fingerboard_lines:
                draw_line(cdst, line[0], line[1])
            cv2.imshow("Detected Lines", cdst)
            return fingerboard_lines, cdst
        return fingerboard_lines, None
    
    def rotate_fingerboard(self, frame, template):
        cornerpoints = cornerpoints_from_line_pair(template.fingerboard_lines, frame.shape)

        rect = cv2.minAreaRect(cornerpoints) # ((x-coordinate, y-coordinate),(width, height), rotation)
        if rect[1][1] > rect[1][0]:
            angle = -(90-rect[2])
            padding = 0.2*rect[1][0]
            padded_rect = (rect[0], (rect[1][0]+2*padding, rect[1][1]), rect[2])
        else:
            angle = rect[2]
            padding = 0.2*rect[1][1]
            padded_rect = (rect[0], (rect[1][0], rect[1][1]+2*padding), rect[2])
        rotated, rot_mat = crop_from_oriented_bounding_box(frame, padded_rect)
        # rotated, rot_mat = rotate_image(frame, rect[0], angle)
        transform_houghlines(template.fingerboard_lines, np.linalg.inv(rot_mat))
        fingerboard_lines = transform_houghlines(template.fingerboard_lines, np.linalg.inv(rot_mat))

        template.register_rotation(rect[0], angle, rot_mat)
        template.register_padded_rect(padded_rect)

        return rotated, fingerboard_lines
    
    def find_undistort_transform(self, rotated_edges, theta_range = 10/180*np.pi, is_visualize=False):
        tested_angles = np.linspace(-theta_range, theta_range, 10, endpoint=False)
        h, theta, d = hough_line(rotated_edges, theta=tested_angles)
        hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10)

        # sort by dist
        # hspace, angles, dists = (list(item) for item in zip(*sorted(zip(hspace, angles, dists), key=lambda x: x[1])))
        
        src_pts = np.empty((0, 2))
        dst_pts = np.empty((0, 2))
        for angle, dist in zip(angles, dists):
            tmp_src_pts = np.array([
                [dist / np.cos(angle), 0],
                [(dist - 200*np.sin(angle)) / np.cos(angle), 200]
            ])
            tmp_dst_pts = np.array([
                [dist / np.cos(angle), 0],
                [dist / np.cos(angle), 200]
            ])
            src_pts = np.concatenate((src_pts, tmp_src_pts), axis=0)
            dst_pts = np.concatenate((dst_pts, tmp_dst_pts), axis=0)
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        lines = np.concatenate((dists[:, np.newaxis], angles[:, np.newaxis]), axis=1)
        if is_visualize and angles is not None:
            cdst = cv2.cvtColor(rotated_edges.copy(), cv2.COLOR_GRAY2BGR)
            for angle, dist in zip(angles, dists):
                draw_line(cdst, dist, angle)
            cv2.imshow('Detected Fret Lines', cdst)
            return homography, cdst, lines
        return homography, None, lines
    
    def locate_fretboard(self, rotated_frame, template):
        '''
        Locating Fretboard Pipeline:
        1. Detect vertical lines
        2. Merge closely spaced lines
        3. Delete fretline outliers by calculating second order difference
        '''
        cdst = cv2.cvtColor(rotated_frame.copy(), cv2.COLOR_GRAY2BGR)

        thresh = cv2.adaptiveThreshold(rotated_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 11, 0)
        edges = skeletonize(thresh).astype("uint8") * 255

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, None, 30, 20)

        # filter by angle (> 45 deg)
        linesP = linesP[2 * np.abs(linesP[:, 0, 0] - linesP[:, 0, 2]) <= np.abs(linesP[:, 0, 1] - linesP[:, 0, 3]), ...]

        # houghlinesp to hough line coordinates
        lines = linesP_to_houghlines(linesP)
        dists = (lines[:, 0] - rotated_frame.shape[0]/2*np.sin(lines[:, 1])) / np.cos(lines[:, 1])
        
        # second stage filter (merge closely spaced lines)
        idxes = np.argwhere(np.diff(dists) < 10).squeeze()
        dists[idxes] = (dists[idxes] + dists[idxes+1]) / 2
        lines[idxes, :] = (lines[idxes, :] + lines[idxes+1, :]) / 2
        dists = np.delete(dists, idxes+1)
        lines = np.delete(lines, idxes+1, axis=0)

        # third stage filter (delete outliers)
        # fret_dists = np.diff(dists)
        # fret_second_order_difference_normalized = np.diff(np.diff(fret_dists)) / fret_dists[1:-1]
        # outliers_idx = np.argwhere(fret_second_order_difference_normalized > 2)
        # print(outliers_idx)
        # if outliers_idx.size > 0:
        #     outliers_idx = list(outliers_idx.reshape((-1,)))
        #     for i in range(len(outliers_idx)):
        #         outlier = outliers_idx[i]
        #         if fret_ratio(dists[outlier-1], dists[outlier], dists[outlier+2]) < \
        #         fret_ratio(dists[outlier-1], dists[outlier+1], dists[outlier+2]):
        #             outliers_idx[i] = outliers_idx[i]+1
        #             # draw_line(cdst, dists[outlier+1], 0, color=(0, 0, 255), thickness=3)
        #         else:
        #             # draw_line(cdst, dists[outlier], 0, color=(0, 0, 255), thickness=3)
        #             pass
        #         # cv2.rectangle(cdst, (int(dists[outlier]), 0), (int(dists[outlier+1]), 20), (0, 255, 0), -1) 
        #     dists = np.delete(dists, outliers_idx)
        #     lines = np.delete(lines, outliers_idx, axis=0)
        #     # hspace = np.delete(hspace, outliers_idx)
        
        for dist, angle in lines:
            draw_line(cdst, dist, angle)
        # cv2.imshow('cdst', cdst)

        cv2.namedWindow("locate_fretboard_debug")
        def mouse_callback(event, x, y, flags, param):
            idx = np.argmin(np.abs(np.array(dists) - x))
            cdst2 = cdst.copy()
            if abs(dists[idx] - x ) < 5:
                draw_line(cdst2, dists[idx], 0, thickness=3)
                cv2.putText(cdst2, f'{dists[idx]}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            cv2.imshow('locate_fretboard_debug', cdst2)
        cv2.setMouseCallback("locate_fretboard_debug", mouse_callback)

        fret_dists = np.diff(dists)
        fret_second_order_difference_normalized = np.diff(fret_dists) / fret_dists[0:-1]
        current_index, max_subset = get_longest_consecutive_condition_subset(fret_second_order_difference_normalized, lambda x: abs(x) < 0.2)

        # vailidation
        is_located = False
        if len(max_subset) + 1 >= 20:
            is_located = True
            fret_idx = np.arange(current_index, current_index + 21)
            for idx in fret_idx:
                cv2.line(cdst, (int(dists[idx]), 0), (int(dists[idx]), 20), (0, 0, 255), 1)
        cv2.imshow('locate_fretboard_debug', cdst)

        if is_located:
            template.register_final_shift([int(dists[fret_idx[0]-1]), 0])
            template.register_fretlines(dists[fret_idx] - int(dists[fret_idx[0]-1]))
            template.register_template_image(rotated_frame[:, int(dists[fret_idx[0]-1]):])

            fretboard_region = rotated_frame[:, int(dists[fret_idx[0]-1]):int(dists[fret_idx[-1]])]
            res = cv2.matchTemplate(fretboard_region, self.string_template2, cv2.TM_CCOEFF_NORMED)
            threshold = 0.4
            res_thres = res
            res_thres = (res > threshold).astype(np.uint8)*255
            res_threscdst = cv2.cvtColor(res_thres.copy(), cv2.COLOR_GRAY2BGR)
            w, h = self.string_template.shape[::-1]

            # lsd=cv2.createLineSegmentDetector(0)

            # lsd_res = res.copy()
            # lsd_res[lsd_res < 0] = 0
            # (lsd_res / np.max(lsd_res) * 255).astype(np.uint8)

            # lines_std=lsd.detect(res_thres)
            # drawn_img = lsd.drawSegments(res_cdst, lines_std[0])
            # cv2.imshow('drawn_img', drawn_img)

            # res = ((res - np.min(res)) / (np.max(res)-np.min(res)) * 255).astype(np.uint8)
            # cv2.imshow('res', res)

            linesP_res = cv2.HoughLinesP(res_thres, 1, np.pi / 180, 30, None, 10, 20)
            hb = HoughBundler()
            linesP_res = hb.process_lines(linesP_res).squeeze()

            # vanishing_point, vp_inliers = ransac_vanishing_point_estimation(linesP_res, num_ransac_iter=1000)
            # homography = np.array([
            #     [1, 0, 0], 
            #     [0, 1, 0], 
            #     [-1/vanishing_point[0], 0, 1]
            # ])
            # affine = np.array([
            #     [1, 0, 0],
            #     [-vanishing_point[1] / vanishing_point[0], 1, 0],
            #     [0, 0, 1]
            # ])
            # homography = affine @ homography

            # res_rect = cv2.warpPerspective(res, homography, res.shape[1::-1], flags=cv2.INTER_LINEAR)
            # cv2.imshow('res_rect', res_rect)

            angles = np.array([hb.get_orientation(lineP) for lineP in linesP_res])
            lengths = np.sqrt((linesP_res[:, 2] - linesP_res[:, 0])**2 + (linesP_res[:, 3] - linesP_res[:, 1])**2)
            bins = np.linspace(-10, 10, 10)
            max_bin_idx = np.argmax(np.histogram(angles, bins, weights=lengths)[0])
            angle_range = bins[[max_bin_idx,max_bin_idx+1]]

            fine_angle_bins = np.linspace(angle_range[0], angle_range[1], 7)
            test_angles = fine_angle_bins[:-1] + 0.5 * (fine_angle_bins[1] - fine_angle_bins[0])
            variance = []
            for test_angle in test_angles:
                m = -np.tan(test_angle/180*np.pi)
                affine_rect = np.array([
                    [1, 0],
                    [m, 1]
                    ])
                start_linesP = (affine_rect @ linesP_res[:, :2].T).T
                end_linesP = (affine_rect @ linesP_res[:, 2:].T).T

                variance.append(np.sum(np.abs(start_linesP[:, 1] - end_linesP[:, 1]) * lengths))

                # affine_rect_h = np.array([
                #     [1, 0, 0],
                #     [m, 1, 0],
                # ])
                # cv2.imshow(f'{test_angle}', cv2.warpAffine(res, affine_rect_h, res.shape[1::-1], flags=cv2.INTER_LINEAR))
                # cv2.waitKey()
            refine_angle = test_angles[np.argmin(variance)]
            m = -np.tan(refine_angle/180*np.pi)
            affine_rect_h = np.array([
                [1, 0, 0],
                [m, 1, 0],
                [0, 0, 1]
            ])
            res_rect = cv2.warpAffine(res, affine_rect_h[:2, :], res.shape[1::-1], flags=cv2.INTER_LINEAR)
            cv2.imshow(f'{refine_angle}', res_rect)

            y_projection = np.sum(res_rect, axis=1)
            autocorrelation = autocorr(y_projection)
            peak_idx = find_peaks(autocorrelation)[0]

            peak_spacing = peak_idx[0]

            harmonic_series_idxes = np.arange(6) * peak_spacing
            harmonic_summation = [np.sum(y_projection[harmonic_series_idxes + i]) for i in np.arange(y_projection.shape[0] - 5*peak_spacing)]
            string_begin_y = np.argmax(harmonic_summation)

            strings_lines = np.vstack((string_begin_y + harmonic_series_idxes + (h/2), np.pi/2*np.ones((6,)))).T
            strings_lines = transform_houghlines(strings_lines, affine_rect_h)
            self.tracked_strings = transform_houghlines(strings_lines, self.template.total_mat)
            
            for strings_line in strings_lines:
                draw_line(cdst, strings_line[0], strings_line[1])

            # cdst_strings = cv2.cvtColor(rotated_frame.copy(), cv2.COLOR_GRAY2BGR)
            # lines2 = cv2.HoughLines(edges, 1, np.pi / 180, 200, None, 0, 0).squeeze()
            # lines2 = lines2[np.argsort(lines2[:, 0]), :]
            # # linesP2 = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, None, 15, 40)
            # cdst2 = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
            # for line in lines2:
            #     draw_line(cdst2, line[0], line[1])
            # # if linesP2 is not None:
            #     # for i in range(0, len(linesP2)):
            #         # l = linesP2[i][0]
            #         # cv2.line(cdst2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
            # # cv2.imshow('cdst2', cdst2)

            # # lines2 = linesP_to_houghlines(linesP2)
            # left_features = (lines2[:, 0] - dists[fret_idx[0]]*np.cos(lines2[:, 1])) / np.sin(lines2[:, 1])
            # mid_features = (lines2[:, 0] - dists[fret_idx[12]]*np.cos(lines2[:, 1])) / np.sin(lines2[:, 1])
            # right_features = (lines2[:, 0] - dists[fret_idx[-1]]*np.cos(lines2[:, 1])) / np.sin(lines2[:, 1])
            # # merge similar lines
            # index = 0
            # while index <= left_features.shape[0] - 2:
            #     rep_index = np.argwhere(np.abs(mid_features[index] - mid_features) < 5).reshape((-1,))
            #     # rep_index = np.argwhere(np.bitwise_and(np.abs(left_features[index] - left_features) < 3, np.abs(right_features[index] - right_features) < 3)).reshape((-1,))
            #     if rep_index.shape[0] == 1:
            #         index = index + 1
            #     else:
            #         left_features[rep_index[0]] = np.mean(left_features[rep_index])
            #         mid_features[rep_index[0]] = np.mean(mid_features[rep_index])
            #         right_features[rep_index[0]] = np.mean(right_features[rep_index])
            #         lines2[rep_index[0]] = hough_line_from_two_points([dists[fret_idx[0]], left_features[rep_index[0]]], [dists[fret_idx[-1]], right_features[rep_index[0]]])
            #         left_features = np.delete(left_features, rep_index[1:])
            #         mid_features = np.delete(mid_features, rep_index[1:])
            #         right_features = np.delete(right_features, rep_index[1:])
            #         lines2 = np.delete(lines2, rep_index[1:], axis=0)
            #         index = 0
            # for line in lines2:
            #     draw_line(cdst_strings, line[0], line[1])
            # # cv2.imshow('cdst_strings', cdst_strings)
            # if mid_features.shape[0] >= 6:
            #     valid_start_index = np.arange(mid_features.shape[0] - 6 + 1)
            #     mean_right_feature = np.array([np.mean(np.diff(right_features[valid_start_index[i_]:valid_start_index[i_]+6])) for i_ in valid_start_index])
            #     final_index = valid_start_index[np.argmin(mean_right_feature)]
            #     self.tracked_strings = transform_houghlines(lines2[final_index:final_index+6], self.template.homography @ self.template.rot_mat)

            # keyLines = []
            self.tracked_marker_positions = []
            for idx in fret_idx:
                start_point = line_line_intersection(strings_lines[0], [dists[idx], 0])
                end_point = line_line_intersection(strings_lines[5], [dists[idx], 0])
                self.tracked_marker_positions.append((start_point + end_point) / 2)
            self.tracked_marker_positions = np.array(self.tracked_marker_positions)
            self.tracked_marker_positions = transform_points(self.tracked_marker_positions, np.linalg.inv(self.template.homography @ self.template.rot_mat))

            #     keyLines.append(construct_KeyLine(rotated_frame, start_point, end_point))
            # bd = cv2.line_descriptor.BinaryDescriptor.createBinaryDescriptor()
            # _, self.tracked_features = bd.compute(rotated_frame, keyLines)

            # autocorrelation[0:autocorrelation.shape[0] - 5*peak_spacing-1]
            # plt.plot(np.arange(len(harmonic_summation)), harmonic_summation)
            # plt.show()

            for i in range(0, linesP_res.shape[0]):
                l = linesP_res[i]
                cv2.line(res_threscdst, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0,0,255), 1, cv2.LINE_AA)
            cv2.imshow('res_threscdst', res_threscdst)

            start_point = line_line_intersection(strings_lines[0], [dists[fret_idx[10]], 0])
            end_point = line_line_intersection(strings_lines[5], [dists[fret_idx[10]], 0])
            self.fret_template = rotated_frame[int(start_point[1]):int(end_point[1]), int(start_point[0])-10:int(start_point[0])+10]

            start_point = line_line_intersection(strings_lines[0], [dists[fret_idx[-1]], 0])
            end_point = line_line_intersection(strings_lines[5], [dists[fret_idx[-1]], 0])
            self.neck_template = rotated_frame[int(start_point[1])-10:int(end_point[1])+10, int(start_point[0])-10:int(start_point[0])+10]

            start_point = line_line_intersection(strings_lines[2], [dists[fret_idx[15]], 0])
            end_point = line_line_intersection(strings_lines[3], [dists[fret_idx[15]], 0])
            self.string_template3 = rotated_frame[int(start_point[1]):int(end_point[1]), int(start_point[0])-15:int(start_point[0])+15]

            for line in strings_lines:
                draw_line(cdst, line[0], line[1])
            cv2.imshow('locate_fretboard_debug', cdst)
            cv2.waitKey(1)
            return True
        else:
            return False
    
    def init_detect(self, frame):
        '''
        Image Processing Pipeline:
        1. Preprocess: convert to gray -> denoising (median filter) -> enhance contrast (clahe)
        2. Edge Detection: canny
        3. Fingerboard detection: Hough Transform -> parallel lines detection
        4. Rotated Bounding-box
        5. Mask Within fingerboard lines
        6. Enhance Contrast
        7. Projective Rectification
        8. Locate the fretboard
        '''
        gray = self.preprocess(frame)
        self.template.register_image(gray)

        edges = cv2.Canny(gray, 30, 100, 5)

        fingerboard_lines, _ = self.fingerboard_lines_detection(edges, theta_tol=5, rho_tol=50, is_visualize=True)
        self.template.register_fingerboard_lines(fingerboard_lines)

        if fingerboard_lines is None:
            return

        rotated_frame, fingerboard_lines = self.rotate_fingerboard(gray, self.template)

        cornerpoints = cornerpoints_from_line_pair(fingerboard_lines, rotated_frame.shape)
        mask = np.zeros_like(rotated_frame)
        cv2.fillPoly(mask, pts=[cornerpoints], color=(255, 255, 255))

        # rotated_frame = cv2.bitwise_and(rotated_frame, mask)

        rotated_edges = cv2.Canny(rotated_frame, 30, 150, 5)
        # cv2.imshow('rotated_edges', rotated_edges)

        homography, _, fret_lines = self.find_undistort_transform(rotated_edges, is_visualize=False)
        self.template.register_homography(homography)

        rotated_frame = cv2.warpPerspective(rotated_frame, homography, rotated_frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        cv2.imshow('rotated_frame', rotated_frame)

        flag = self.locate_fretboard(rotated_frame, self.template)

        if flag:
            rotated_frame_rgb, rot_mat = crop_from_oriented_bounding_box(frame, self.template.padded_rect)
            # ycrcb = cv2.cvtColor(rotated_frame_rgb, cv2.COLOR_BGR2YCrCb)
            # cv2.imshow('ycrcb_y', ycrcb[:, :, 0])
            # cv2.imshow('ycrcb_cr', ycrcb[:, :, 1])
            # cv2.imshow('ycrcb_cb', ycrcb[:, :, 2])

            # window_size = 5
            # # standard deviation
            # ycrcb = ycrcb.astype("float32") # convert to float
            # mu = cv2.blur(ycrcb, (window_size, window_size))
            # musquare = cv2.blur(ycrcb**2, (window_size, window_size))
            # std = np.sqrt(musquare - mu**2)
            # std = std / np.max(np.max(std, axis=0), axis=0)
            # # discontinuity
            # grad = np.sqrt(cv2.Sobel(ycrcb, cv2.CV_32F, 0, 1, ksize=window_size)**2 + \
            #                cv2.Sobel(ycrcb, cv2.CV_32F, 1, 0, ksize=window_size)**2)
            # grad = grad / np.max(np.max(grad, axis=0), axis=0)
            # # local homogeneity
            # color_features = 1 - grad * std

            # # cv2.imshow('texture', (grad[:, :, 0]*255).astype(np.uint8))
            # # cv2.imshow('color_features_y', (color_features[:, :, 0]*255).astype(np.uint8))
            # # cv2.imshow('color_features_cr', (color_features[:, :, 1]*255).astype(np.uint8))
            # # cv2.imshow('color_features_cb', (color_features[:, :, 2]*255).astype(np.uint8))
            # # cv2.waitKey()


            # # gabor filter texture for y channel
            # ksize = 11
            # sigma = 5 # sigma for Gaussian envelope
            # gamma = 0.1
            # theta = 0
            # theta2 = np.pi/2
            # lambd = 18 # frequency
            # lambd2 = 1
            # # theta = 0, np.pi/4, np.pi/2, np.pi*3/4
            # # lambd = 20, 40
            # kernel1 = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F)
            # npprint(kernel1)
            # result1 = cv2.filter2D(rotated_frame, cv2.CV_32F, kernel1)
            # kernel2 = cv2.getGaborKernel((ksize, ksize), sigma, theta2, lambd2, gamma, psi=0, ktype=cv2.CV_32F)
            # result2 = cv2.filter2D(rotated_frame, cv2.CV_32F, kernel2)

            # cv2.imshow('kernel', kernel2)
            # plt.imshow(result2)
            # plt.show()

            # total_features = np.concatenate((ycrcb[:, :, 1, np.newaxis], ycrcb[:, :, 2, np.newaxis], result1[..., np.newaxis], result2[..., np.newaxis]), axis=2)

            # cornerpoints = np.vstack((
            #     line_line_intersection(self.tracked_strings[0], self.template.image_fretlines[0]),
            #     line_line_intersection(self.tracked_strings[0], self.template.image_fretlines[-1]),
            #     line_line_intersection(self.tracked_strings[-1], self.template.image_fretlines[-1]),
            #     line_line_intersection(self.tracked_strings[-1], self.template.image_fretlines[0]),
            # ))
            # cornerpoints = transform_points(cornerpoints, rot_mat).astype(int)
            # mask = np.zeros_like(rotated_frame_rgb)
            # cv2.fillPoly(mask, pts=[cornerpoints], color=(255, 255, 255))

            # data1 = total_features[mask[:, :, 0] != 0, :]
            # data2 = total_features[mask[:, :, 0] == 0, :]

            # data = np.concatenate((data1, data2), axis=0)
            # label = np.concatenate((np.ones((data1.shape[0], 1)), np.zeros((data2.shape[0], 1))), axis=0).reshape((-1,))

            # show_multiple_images_cube(data1[:, 1:], data2[:, 1:])
            # plt.show()

            # model = svm.LinearSVC(C=1, max_iter=1000)
            # model.fit(data, label)
            # model.predict(data1)

            self.is_initialized = True

            self.tracked_fretlines = self.template.image_fretlines

            cdst = frame.copy()

            # self.template.draw_fretboard(cdst, np.linalg.inv(self.template.total_mat))
            # for line in self.template.fingerboard_lines:
            #     draw_line(cdst, line[0], line[1])
            # for line in self.template.image_fretlines:
            #     draw_line(cdst, line[0], line[1])

            cornerpoints = np.array([
                line_line_intersection(self.template.fingerboard_lines[0], self.template.image_fretlines[0]),
                line_line_intersection(self.template.fingerboard_lines[1], self.template.image_fretlines[0]),
                line_line_intersection(self.template.fingerboard_lines[1], self.template.image_fretlines[-1]),
                line_line_intersection(self.template.fingerboard_lines[0], self.template.image_fretlines[-1]),
            ])
            rect = cv2.minAreaRect(cornerpoints.astype(int))
            rect = (rect[0], (rect[1][0] + 50, rect[1][1] + 50), rect[2])

            self.oriented_bounding_box = rect
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(cdst, [box], 0, (0,0,255), 2)

            _, crop_transform = crop_from_oriented_bounding_box(frame, self.oriented_bounding_box)

            pts1, pts2 = prepare_draw_fretboard(rect[1][0], self.tracked_fretlines, crop_transform)
            for pt1, pt2 in zip(pts1, pts2):
                cv2.line(cdst, pt1, pt2, color=(0, 255, 0), thickness=2)
            pts1, pts2 = prepare_draw_strings(rect[1][1], self.tracked_strings, crop_transform)
            for pt1, pt2 in zip(pts1, pts2):
                cv2.line(cdst, pt1, pt2, color=(0, 255, 0), thickness=2)

            cv2.imshow('final_result', cdst)

            cornerpoints = np.array([
                line_line_intersection(self.tracked_strings[0], self.template.image_fretlines[0]),
                line_line_intersection(self.tracked_strings[-1], self.template.image_fretlines[0]),
                line_line_intersection(self.tracked_strings[-1], self.template.image_fretlines[-1]),
                line_line_intersection(self.tracked_strings[0], self.template.image_fretlines[-1]),
            ])
            mask = (np.ones_like(frame[:, :])*255).astype(np.uint8)
            cv2.fillPoly(mask, pts=[cornerpoints.astype(int)], color=(0, 0, 0))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13, 13))
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
            frame = cv2.bitwise_and(frame, mask)
            # cv2.imshow('bg', frame)
            # cv2.waitKey()

            self.backSub.apply(frame)
    
    def refine_fretboard(self, frame, final_cdst=None):
        '''
        need to update:

        self.oriented_bounding_box
        self.tracked_fretlines
        '''
        if final_cdst is None:
            final_cdst = frame.copy()
        
        fgmask = self.backSub.apply(frame, None, learningRate=0)
        cv2.imshow('fgmask', fgmask)
        
        rotated_cropped_frame, crop_transform = crop_from_oriented_bounding_box(frame, self.oriented_bounding_box)
        rotated_cropped_mask, _ = crop_from_oriented_bounding_box(fgmask, self.oriented_bounding_box)
        rotated_cropped_frame = cv2.bitwise_and(rotated_cropped_frame, rotated_cropped_frame, mask=rotated_cropped_mask)

        # self.colorthresholder.update(rotated_cropped_frame)
        # rotated_cropped_frame = self.colorthresholder.output

        gray = self.preprocess(rotated_cropped_frame)
        cdst = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        
        # canny = cv2.Canny(gray, threshold1=40, threshold2=150, apertureSize=5, L2gradient=True)
        # canny_cdst = cv2.cvtColor(canny.copy(), cv2.COLOR_GRAY2BGR)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 31, 0)
        edges = skeletonize(thresh).astype("uint8") * 255
        # cv2.imshow('edges', edges)

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, None, 10, 20)

        # filter by angle (> 45 deg)
        # horiz_lines = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0).squeeze()

        # horizontal_linesP = linesP[np.abs(linesP[:, 0, 0] - linesP[:, 0, 2]) > np.abs(linesP[:, 0, 1] - linesP[:, 0, 3]), ...]
        # horizontal_linesP = horizontal_linesP.squeeze()
        # horizontal_linesP = horizontal_linesP[np.sqrt((horizontal_linesP[:, 2] - horizontal_linesP[:, 0])**2 + (horizontal_linesP[:, 3] - horizontal_linesP[:, 1])**2)>=100, :]

        linesP = linesP[np.abs(linesP[:, 0, 0] - linesP[:, 0, 2]) <= np.abs(linesP[:, 0, 1] - linesP[:, 0, 3]), ...]
        lines = linesP_to_houghlines(linesP, sort=False)

        for line in lines:
            draw_line(cdst, line[0], line[1], color=(0, 0, 255))

        # filter by vanishing point
        vanishing_point, vp_inliers = ransac_vanishing_point_estimation(linesP)
        if vanishing_point is not None:
            linesP = linesP[vp_inliers, ...]
            lines = lines[vp_inliers, :]
        # cv2.imshow('cdst', cdst)
        # cv2.waitKey()
        # vanishing_point, vp_inliers = ransac_find_vanishing_point(lines)

        for line in lines:
            draw_line(cdst, line[0], line[1])
        cv2.imshow('cdst', cdst)

        homography = find_projective_rectification(lines)
        gray_rect = cv2.warpPerspective(gray, homography, gray.shape[1::-1], flags=cv2.INTER_LINEAR)
        rect_lines = transform_houghlines(lines, np.linalg.inv(homography))
        fret_dists = rect_lines[:, 0] / np.cos(rect_lines[:, 1])

        gray_rect_cdst = cv2.cvtColor(gray_rect.copy(), cv2.COLOR_GRAY2BGR)

        cv2.imshow('self.fret_template', self.fret_template)
        res = cv2.matchTemplate(gray_rect, self.fret_template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where(res >= threshold)
        loc = np.vstack((loc[1], loc[0])).T

        w, h = self.fret_template.shape[::-1]
        # for pt in loc:
            # cv2.rectangle(gray_rect_cdst, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
            # cv2.circle(gray_rect_cdst, (pt[0] + int(w/2), pt[1] + int(h/2)), 2, (0,0,255))

        loc_center = loc + [w/2, h/2]
        label = np.argmin(np.abs(loc_center[:, 0].reshape((-1, 1)) - fret_dists.reshape((1, -1))), axis=1)

        unique_label = np.unique(label)
        loc_max = []
        for unique_label_ in unique_label:
            class_mask = label == unique_label_
            maxind = np.argmax(res[loc[class_mask, 1], loc[class_mask, 0]])
            loc_max.append(loc[np.argwhere(class_mask).reshape((-1,))[maxind], :])
        loc_max = np.array(loc_max)

        ransac_lm = RANSACRegressor(estimator=LinearRegression())
        ransac_lm.fit(loc_center[:, 0].reshape((-1, 1)), loc_center[:, 1].reshape((-1, 1)))
        
        mid_line = houghline_from_kb(ransac_lm.estimator_.coef_[0, 0], ransac_lm.estimator_.intercept_[0])
        draw_line(gray_rect_cdst, mid_line[0], mid_line[1])

        for pt in loc_max:
            cv2.rectangle(gray_rect_cdst, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            cv2.circle(gray_rect_cdst, (pt[0] + int(w/2), pt[1] + int(h/2)), 3, (0,0,255))
        # for pt in prev_markers:
        #     cv2.circle(gray_rect_cdst, (int(pt[0]), int(pt[1])), 3, (255,0,0))

        # cost_matrix = np.abs(prev_markers[:, 0].reshape((-1, 1)) - (loc_max[:, 0] + w/2).reshape((1, -1)))
        # cost_matrix = np.concatenate((cost_matrix, np.ones((cost_matrix.shape[0], cost_matrix.shape[0])) * 50), axis=1)
        # prev_ind, now_ind = linear_sum_assignment(cost_matrix)
        # matched_prev_ind = prev_ind[now_ind < min(loc_max.shape[0], prev_markers.shape[0])]
        # matched_now_ind = now_ind[now_ind < min(loc_max.shape[0], prev_markers.shape[0])]

        # cv2.waitKey()

        # if linesP is not None:
        #     for i in range(0, len(linesP)):
        #         l = linesP[i][0]
        #         cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
        #     for i in range(0, len(horizontal_linesP)):
        #         l = horizontal_linesP[i]
        #         cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (255,0,0), 1, cv2.LINE_AA)    

        # # Center Horizontal Line Estimation
        # linesP = linesP.squeeze()

        # # cdst_fs = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
        # fret_string_intersections = [line_segments_intersection_batch(lineP, horizontal_linesP) for lineP in linesP]
        # for fret_string_intersections_ in fret_string_intersections:
        #     for pnt in fret_string_intersections_:
        #         cv2.circle(cdst, (int(pnt[0]), int(pnt[1])), 3, (0, 255, 0))
        # cv2.imshow('fret_string_intersections', cdst)


        # mid_points = (linesP[:, :2] + linesP[:, 2:]) / 2
        # ransac_lm = RANSACRegressor(estimator=LinearRegression())
        # ransac_lm.fit(mid_points[:, 0].reshape((-1, 1)), mid_points[:, 1].reshape((-1, 1)))

        # mid_line = houghline_from_kb(ransac_lm.estimator_.coef_[0, 0], ransac_lm.estimator_.intercept_[0])
        # draw_line(cdst, mid_line[0], mid_line[1])
        # cv2.imshow('mid line', cdst)
        # cv2.waitKey()

        # intersection_points = np.array([line_line_intersection(riti, mid_line) for riti in lines])

        prev_lines = transform_houghlines(self.tracked_fretlines, np.linalg.inv(homography @ crop_transform))
        prev_markers = (prev_lines[:, 0] - (gray_rect.shape[0])/2 * np.sin(prev_lines[:, 1])) / np.cos(prev_lines[:, 1])

        cost_matrix = np.abs(prev_markers.reshape((-1, 1)) - fret_dists.reshape((1, -1)))
        cost_matrix = np.concatenate((cost_matrix, np.ones((cost_matrix.shape[0], prev_markers.shape[0])) * 50), axis=1)

        prev_ind, now_ind = linear_sum_assignment(cost_matrix)
        matched_prev_ind = prev_ind[now_ind < min(self.tracked_fretlines.shape[0], lines.shape[0])]
        matched_now_ind = now_ind[now_ind < min(self.tracked_fretlines.shape[0], lines.shape[0])]

        # model = VanishingLineEstimator()
        # fit_res = model.ls_fit(homogeneous_coords_from_houghlines(lines[matched_now_ind, :]), self.tracked_N[matched_prev_ind])

        # # projective rectification
        # homography = np.eye(3)
        # homography[2, :] = fit_res[0]
        # print(fit_res[0])

        # gray_rect = cv2.warpPerspective(gray, homography, gray.shape[1::-1], flags=cv2.INTER_LINEAR)
        # gray_rect_cdst = cv2.cvtColor(gray_rect.copy(), cv2.COLOR_GRAY2BGR)
        # rect_matched_lines = transform_houghlines(lines[matched_now_ind, :], np.linalg.inv(homography))
        # for line in rect_matched_lines:
        #     draw_line(gray_rect_cdst, line[0], line[1])
        # cv2.imshow('rectified_gray', gray_rect_cdst)
        # cv2.waitKey()

        # inliers are consider matched
        now_tracked_frets = np.zeros_like(self.tracked_fretlines)
        now_tracked_frets[matched_prev_ind, :] = transform_houghlines(lines[matched_now_ind, :], crop_transform)

        # estimate 1d similarity transform
        model_robust, inliers = ransac(
        (np.vstack((prev_markers[matched_prev_ind], np.zeros((matched_prev_ind.shape[0],)))).T, 
         np.vstack((fret_dists[matched_now_ind], np.zeros((matched_now_ind.shape[0],)))).T), SimilarityTransform, min_samples=2, residual_threshold=5, max_trials=100)
        s = model_robust.params[0, 0]
        t = model_robust.params[0, 2]

        outliers = inliers == False
        if np.sum(outliers) > 0:
            now_tracked_frets[matched_prev_ind[outliers], :] = transform_houghlines(np.vstack((s * prev_markers[matched_prev_ind[outliers]] + t, np.zeros((matched_prev_ind[outliers].shape[0])))).T, homography @ crop_transform)

        notmatched_prev_ind = np.setdiff1d(np.arange(prev_markers.shape[0]), matched_prev_ind)
        if notmatched_prev_ind.shape[0] > 0:
            now_tracked_frets[notmatched_prev_ind, :] = transform_houghlines(np.vstack((s * prev_markers[notmatched_prev_ind] + t, np.zeros((notmatched_prev_ind.shape[0])))).T, homography @ crop_transform)
        
        # vertical localization
        start_end_fretlines = transform_houghlines(now_tracked_frets[[0, -1], :], np.linalg.inv(homography @ crop_transform))
        total_span = (start_end_fretlines[:, 0] - (gray_rect.shape[0])/2 * np.sin(start_end_fretlines[:, 1])) / np.cos(start_end_fretlines[:, 1])

        gray_rect_cdst = cv2.cvtColor(gray_rect.copy(), cv2.COLOR_GRAY2BGR)
        res = cv2.matchTemplate(gray_rect, self.string_template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.5
        res[res < threshold] = 0
        # res_thres = res
        res_thres = (res > threshold).astype(np.uint8)*255
        w, h = self.string_template.shape[::-1]

        cv2.imshow('res', res)

        # being start and end two points (x1,y1), (x2,y2)
        fretboard_upper = []
        fretboard_lower = []
        peak_spacings = []
        peak_loc = []
        peak_loc_valid = []
        for fret in transform_houghlines(now_tracked_frets, np.linalg.inv(homography @ crop_transform)):
            start = [int(fret[0] / math.cos(fret[1])), 0]
            peak_loc.append(start[0])
            end = [int((fret[0] - gray_rect.shape[0]*math.sin(fret[1])) / math.cos(fret[1])), gray_rect.shape[0]]

            # shift because the response map has an offset
            start = start - np.array([int(w/2), int(h/2)])
            end = end - np.array([int(w/2), int(h/2)])

            cols, rows = skimage_line(*start, *end)
            valid_index = (rows >= 0 ) & (rows < res.shape[0]) & (cols >= 0 ) & (cols < res.shape[1])
            cols = cols[valid_index]
            rows = rows[valid_index]

            # hot_spots = np.where(res[rows, cols] >= threshold)[0]
            # if hot_spots.size > 0:
            hot_spots = find_peaks(res[rows, cols])[0]
            if hot_spots.size > 1 and hot_spots[-1] - hot_spots[0] > 30:
                fretboard_upper.append([cols[hot_spots[0]] + int(w/2), rows[hot_spots[0]] + int(h/2)])
                fretboard_lower.append([cols[hot_spots[-1]] + int(w/2), rows[hot_spots[-1]] + int(h/2)])
            
            if res[rows, cols].size > 0:
                autocorrelation = autocorr(res[rows, cols])
                peak_idx = find_peaks(autocorrelation)[0]
                if peak_idx.size > 0:
                    peak_loc_valid.append(start[0])
                    if peak_idx[0] < 30:
                        peak_spacings.append(peak_idx[0])
                    else:
                        fft_w = np.fft.rfft(res[rows, cols])
                        w_peaks = find_peaks(np.abs(fft_w))[0]
                        # freqs = np.fft.fftfreq(len(fft_w))
                        peak_spacings.append(rows.size / w_peaks[np.argmax(np.abs(fft_w)[w_peaks])])
                        # if rows.size / w_peaks[np.argmax(np.abs(fft_w)[w_peaks])] > 30:
                        #     plt.plot(autocorrelation)
                        #     plt.title('autocorrelation')
                        #     plt.figure()
                        #     plt.plot(res[rows, cols], '.')
                        #     plt.title('')
                        #     plt.figure()
                        #     plt.plot(rows.size / np.arange(fft_w.size), np.abs(fft_w))
                        #     plt.title('fft')
                        #     plt.show()

        npprint(f'{self.counter}: {peak_spacings}')
        if self.counter == 16:
            npprint(peak_loc_valid)
            npprint(peak_spacings)
        ransac__lm = RANSACRegressor(estimator=LinearRegression())
        ransac__lm.fit(np.array(peak_loc_valid).reshape((-1, 1)), np.array(peak_spacings).reshape((-1, 1)))
        peak_spacings_fit = np.array(peak_loc)*ransac__lm.estimator_.coef_[0, 0] + ransac__lm.estimator_.intercept_[0]
        if np.sum(peak_spacings_fit < 0) > 0:
            print('')

        strings_start = []
        strings_start_idx = []
        for i, fret in enumerate(transform_houghlines(now_tracked_frets, np.linalg.inv(homography @ crop_transform))):
            start = [int(fret[0] / math.cos(fret[1])), 0]
            end = [int((fret[0] - gray_rect.shape[0]*math.sin(fret[1])) / math.cos(fret[1])), gray_rect.shape[0]]

            # shift because the response map has an offset
            start = start - np.array([int(w/2), int(h/2)])
            end = end - np.array([int(w/2), int(h/2)])

            cols, rows = skimage_line(*start, *end)
            valid_index = (rows >= 0 ) & (rows < res.shape[0]) & (cols >= 0 ) & (cols < res.shape[1])
            cols = cols[valid_index]
            rows = rows[valid_index]

            if res[rows, cols].size > 0:
                max_idx = np.argmax(np.correlate(res[rows, cols], np.ones((int(peak_spacings_fit[i]*5),)), mode='vdlid'))
                strings_start.append([cols[max_idx] + int(w/2), rows[max_idx] + int(h/2)])
                strings_start_idx.append(i)

        # strings_start = np.array(strings_start)
        # fretboard_upper = strings_start
        # fretboard_lower = fretboard_upper.copy()
        # fretboard_lower[:, 1] = fretboard_lower[:, 1] + peak_spacings_fit[strings_start_idx]*5
        fretboard_upper = np.array(fretboard_upper)
        fretboard_lower = np.array(fretboard_lower)

        for pnt in fretboard_upper:
            cv2.circle(gray_rect_cdst, (pnt[0], pnt[1]), 5, (0,0,255), -1)
        for pnt in fretboard_lower:
            cv2.circle(gray_rect_cdst, (pnt[0], pnt[1]), 5, (0,255,255), -1)
        # for pnt in strings_start:
        #     cv2.circle(gray_rect_cdst, (pnt[0], pnt[1]), 5, (255,255,255), -1)
        # for i, pnt in enumerate(strings_start):
        #     cv2.circle(gray_rect_cdst, (pnt[0], pnt[1] + int(peak_spacings_fit[i]*5)), 5, (255,255,255), -1)
        
        ransac__lm = RANSACRegressor(estimator=LinearRegression())
        ransac__lm.fit(fretboard_upper[:, 0].reshape((-1, 1)), fretboard_upper[:, 1].reshape((-1, 1)))
        upper_line = houghline_from_kb(ransac__lm.estimator_.coef_[0, 0], ransac__lm.estimator_.intercept_[0])
        ransac__lm.fit(fretboard_lower[:, 0].reshape((-1, 1)), fretboard_lower[:, 1].reshape((-1, 1)))
        lower_line = houghline_from_kb(ransac__lm.estimator_.coef_[0, 0], ransac__lm.estimator_.intercept_[0])

        draw_line(gray_rect_cdst, upper_line[0], upper_line[1])
        draw_line(gray_rect_cdst, lower_line[0], lower_line[1])

        # dst = cv2.cornerHarris(gray_rect, 4, 3, 0.04)
        # normalized_image = cv2.convertScaleAbs(cv2.normalize(dst, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
        # cv2.imshow('harris', normalized_image)

        # loc = np.where(res >= threshold)
        # prob = res[loc]
        # loc = np.vstack((loc[1], loc[0])).T
        # w, h = self.string_template.shape[::-1]
        # pick = non_max_suppression(np.hstack((loc[:, [1, 0]], loc[:, [1, 0]] + [w, h])), prob, 0.5)

        # for (startY, startX) in pick[:, :2]:
            # cv2.circle(gray_rect_cdst, (startX + int(w/2), startY + int(h/2)), 3, (0,0,255), -1)
            # cv2.rectangle(gray_rect_cdst, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        # w2, h2 = self.string_template2.shape[::-1]
        # res2 = cv2.matchTemplate(gray_rect, self.string_template2, cv2.TM_CCOEFF_NORMED)
        # threshold = 0.6
        # loc2 = np.where(res2 >= threshold)
        # loc2 = np.vstack((loc2[1], loc2[0])).T
        # res2_thres = (res2 >= threshold).astype(np.uint8)*255
        
        # cv2.imshow('res2_thres', res2_thres)HoughBundler
        # cv2.imshow('local_region', gray_rect[int(pick[0, 1]) + int(h/2)-7:int(pick[0, 1]) + int(h/2)+7, int(pick[0, 0]) + int(w/2)-20:int(pick[0, 0]) + int(w/2)+20])

        # strings_lines = None
        # if np.sum(res2_thresholded > 0) > 2000:
        #     linesP = cv2.HoughLinesP(res2_thresholded, 1, np.pi / 180, 20, None, 15, 30)
        #     bundler = HoughBundler(min_distance=3,min_angle=2)
        #     linesP = bundler.process_lines(linesP)

        #     peaks = pick[:, :2] + [w/2, h/2]
        #     linesP = linesP.squeeze() + [w2/2, h2/2, w2/2, h2/2]

        #     cdst2 = cv2.cvtColor(np.zeros_like(res2_thresholded), cv2.COLOR_GRAY2BGR)
        #     # if linesP is not None:
        #     #     for i in range(0, len(linesP)):
        #     #         l = linesP[i].astype(int)
        #     #         cv2.line(cdst2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
            
        #     for (startX, startY) in peaks.astype(int):
        #         cv2.circle(cdst2, (startX, startY), 3, (0,0,255), -1)

        #     labels = init_group_points(bundler, peaks, linesP)
        #     labels = refine_group_points(labels, peaks, linesP)
        #     labels = merge_group_points(labels, peaks, linesP)
        #     colors = np.random.choice(range(256), size=(linesP.shape[0], 3))

        #     for (startX, startY), label_ in zip(peaks[labels > -1].astype(int), labels[labels > -1]):
        #         # cv2.line(cdst2, start_point, end_point, (int(colors[label_][0]),int(colors[label_][1]),int(colors[label_][2])), thickness=2)
        #         cv2.circle(cdst2, (startX, startY), 3, (int(colors[label_][0]),int(colors[label_][1]),int(colors[label_][2])), -1)

        #     strings_lines = []
        #     for label_ in labels[labels > -1]:
        #         if np.sum(labels == label_) > 10:
        #             current_pnts = labels == label_
        #             k, b = np.linalg.lstsq(np.vstack((peaks[current_pnts, 0], np.ones_like(peaks[current_pnts, 0]))).T, peaks[current_pnts, 1])[0]
        #             start_point = (0, int(b))
        #             end_point = (gray_rect.shape[1],  int(k*gray_rect.shape[1]+b))
        #             strings_lines.append([0, int(b), gray_rect.shape[1],  int(k*gray_rect.shape[1]+b)])
        #             cv2.line(cdst2, start_point, end_point, (int(colors[label_][0]),int(colors[label_][1]),int(colors[label_][2])), thickness=1)
        #             # strings_lines.append(houghline_from_kb(k, b))
        #     strings_lines = np.array(strings_lines)
        #     cv2.imshow('labeled intersections', cdst2)
        #     # def mouse_callback(event, x, y, flags, param):
        #     #     idx = np.argmin(np.sum(np.abs(peaks - [x, y]), axis=1))
        #     #     if np.sum(np.abs(peaks[idx] - [x, y])) - x < 5:
        #     #         print(labels[idx])
        #     # cv2.setMouseCallback("labeled intersections", mouse_callback)
        # else:
        #     main_direction = np.array([1, 0])
        
        # if strings_lines is not None:
        #     for line in strings_lines:
        #         draw_line(gray_rect_cdst, line[0], line[1])

        # update
        self.tracked_fretlines = now_tracked_frets

        center = self.oriented_bounding_box[0]
        center = transform_points(np.array(center).reshape((1, 2)), homography @ crop_transform).squeeze()
        center[0] = np.mean(total_span)
        center[1] = ransac_lm.estimator_.coef_[0, 0] * center[0] + ransac_lm.estimator_.intercept_[0] # correct y coord
        center = transform_points(center.reshape((1, 2)), np.linalg.inv(crop_transform) @ np.linalg.inv(homography)).squeeze()

        old_oriented_bounding_box = self.oriented_bounding_box
        self.oriented_bounding_box = ((center[0], center[1]), (self.oriented_bounding_box[1][0]*s, (total_span[1] - total_span[0]) + 50), self.oriented_bounding_box[2] + np.arctan(ransac_lm.estimator_.coef_[0, 0])*180/np.pi)

        # theta_range = 20/180*np.pi
        # tested_angles = np.linspace(-theta_range, theta_range, 20, endpoint=False)
        # h, theta, d = hough_line(edges, theta=tested_angles)
        # hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=5, min_angle=2, threshold=20)

        # cdst = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)

        # # houghlinesp to hough line coordinates
        # lines = linesP_to_houghlines(linesP)
        # # for line in lines:
        # #     draw_line(cdst, line[0], line[1])
        
        # # match lines
        # prev_crop_tracked_fretlines = transform_houghlines(self.tracked_fretlines, np.linalg.inv(crop_transform))

        # lines3 = find_homography_from_matched_fretlines(prev_crop_tracked_fretlines, lines)
        # lines_coords = homogeneous_coords_from_houghlines(lines)
        # spacing = self.template.template_fretlines[:, 0] - self.template.template_fretlines[0, 0]
        # vanishing_line = vanishing_line_from_parallel_lines(lines_coords, spacing)

        # npprint(lines3)
        # for line in lines3:
        #     draw_line(cdst, line[0], line[1], color=(0, 0, 255))
        # cv2.imshow('lines_debug', cdst)
        # cv2.waitKey()
        # outliers = inliers == False

        # # update tracked lines
        # # self.tracked_fretlines[matched_index[inliers], :] = transform_houghlines(prev_crop_tracked_fretlines[matched_index[inliers], :], transform_mat @ crop_transform)
        # tracked_fretlines = self.tracked_fretlines.copy()
        # tracked_fretlines[prev_ind[inliers], :] = transform_houghlines(lines[now_ind[inliers], :], crop_transform)

        # if np.sum(outliers) > 0:
        #     # self.tracked_fretlines[outliers, :] = transform_houghlines(lines[now_ind[outliers], :], crop_transform)
        #     tracked_fretlines[prev_ind[outliers], :] = transform_houghlines(prev_crop_tracked_fretlines[prev_ind[outliers], :], np.linalg.inv(transform) @ crop_transform)
        
        # for line in transform_houghlines(tracked_fretlines, np.linalg.inv(crop_transform)):
        #     draw_line(cdst, line[0], line[1], color=(0, 0, 255))
        # for line in transform_houghlines(tracked_fretlines[inliers, :], np.linalg.inv(crop_transform)):
        #     draw_line(cdst, line[0], line[1], color=(0, 0, 255), thickness=2)
        # cv2.imshow('cdst', cdst)

        # # tmp = self.tracked_fretlines[:, 0] < 0
        # # self.tracked_fretlines[tmp, 1] = self.tracked_fretlines[tmp, 1] - np.sign(self.tracked_fretlines[tmp, 1]) * np.pi
        # # self.tracked_fretlines[tmp, 0] = -self.tracked_fretlines[tmp, 0]

        tracked_upper = transform_houghlines(upper_line.reshape((1, -1)), homography @ crop_transform)[0]
        tracked_lower = transform_houghlines(lower_line.reshape((1, -1)), homography @ crop_transform)[0]
        cornerpoints = np.array([
            line_line_intersection(tracked_upper, self.tracked_fretlines[0]),
            line_line_intersection(tracked_lower, self.tracked_fretlines[0]),
            line_line_intersection(tracked_lower, self.tracked_fretlines[-1]),
            line_line_intersection(tracked_upper, self.tracked_fretlines[-1]),
        ])
        mask = (np.ones_like(frame[:, :])*255).astype(np.uint8)
        cv2.fillPoly(mask, pts=[cornerpoints.astype(int)], color=(0, 0, 0))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13, 13))
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        frame = cv2.bitwise_and(frame, mask)
        cv2.imshow('frame', frame)

        self.backSub.apply(frame, None, learningRate=-1)
        cv2.imshow('bg_model', self.backSub.getBackgroundImage())

        pts1, pts2 = prepare_draw_fretboard(self.oriented_bounding_box[1][0], self.tracked_fretlines, crop_transform)

        for line in prev_lines:
            draw_line(gray_rect_cdst, line[0], line[1], color=(255, 0, 0))
        for line in rect_lines:
            draw_line(gray_rect_cdst, line[0], line[1], color=(0, 0, 255))
        for i in range(matched_prev_ind.shape[0]):
            cv2.line(gray_rect_cdst, (int(prev_markers[matched_prev_ind[i]]), i*5+1), (int(fret_dists[matched_now_ind[i]]), i * 5+1), (255, 0, 255), thickness=2)
        for i in range(matched_prev_ind[inliers].shape[0]):
            cv2.line(gray_rect_cdst, (int(prev_markers[matched_prev_ind[i]]), i*5+1), (int(fret_dists[matched_now_ind[i]]), i * 5+1), (0, 255, 0), thickness=2)

        box = cv2.boxPoints(old_oriented_bounding_box)
        box = np.intp(box)
        cv2.drawContours(final_cdst, [box], 0, (0, 255, 0), 2)

        # box = cv2.boxPoints(self.oriented_bounding_box)
        # box = np.intp(box)
        # cv2.drawContours(final_cdst, [box], 0, (255, 255, 0), 2)

        # for line in self.tracked_fretlines:
        #     draw_line(final_cdst, line[0], line[1], color=(0, 255, 0))
        # for line in self.tracked_fretlines[notmatched_prev_ind]:
        #     draw_line(final_cdst, line[0], line[1], color=(0, 0, 255))
        # if strings_lines is not None:
        #     pts1a = transform_points(strings_lines[:, :2], np.linalg.inv(homography @ crop_transform))
        #     pts2a = transform_points(strings_lines[:, 2:], np.linalg.inv(homography @ crop_transform))
        #     for pt1, pt2 in zip(pts1a, pts2a):
        #         cv2.line(final_cdst, pt1.astype(int), pt2.astype(int), color=(0, 255, 0), thickness=2)

        for line in transform_houghlines(upper_line.reshape((1, -1)), homography @ crop_transform):
            draw_line(final_cdst, line[0], line[1])
        for line in transform_houghlines(lower_line.reshape((1, -1)), homography @ crop_transform):
            draw_line(final_cdst, line[0], line[1])

        for pt1, pt2 in zip(pts1, pts2):
            cv2.line(final_cdst, pt1, pt2, color=(0, 255, 0), thickness=2)
        for pt1, pt2 in zip(pts1[notmatched_prev_ind], pts2[notmatched_prev_ind]):
            cv2.line(final_cdst, pt1, pt2, color=(0, 255, 0), thickness=2)
        cv2.putText(final_cdst, f'{self.counter}', (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            
        cv2.imshow('cdst', cdst)
        cv2.imshow('final_cdst', final_cdst)
        cv2.imshow('gray_rect', gray_rect_cdst)
        cv2.imshow('template', self.string_template)
        cv2.waitKey()
        # final_cdst[:gray_rect_cdst.shape[0], :gray_rect_cdst.shape[1]] = gray_rect_cdst
        cv2.putText(final_cdst, f'{self.counter}', (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        self.video.write(final_cdst)
    
    def apply(self, frame):
        if not self.is_initialized:
            self.init_detect(frame)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            self.colorthresholder = ColorThresholder(self.screensize)
            self.video = cv2.VideoWriter('video.mp4',  
                    cv2.VideoWriter_fourcc(*'H264'), 
                    25, (frame.shape[1], frame.shape[0]))
        else:
            final_cdst = self.refine_fretboard(frame)
            # self.videoplayback.put(final_cdst)
            # self.videoplayback.imshow()
        self.counter = self.counter + 1

if __name__ == '__main__':
    cap = cv2.VideoCapture('test\\guitar3.mp4')

    # img1 = cv2.imread('test\\hand.png')
    # img2 = cv2.imread('test\\fretboard.png')
    # cbcr1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
    # cbcr1 = cbcr1[cbcr1[:, :, 0] != 0]
    # # cbcr1 = cbcr1[:, :, 1:]
    # cbcr2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
    # cbcr2 = cbcr2[cbcr2[:, :, 0] != 0]
    # # cbcr2 = cbcr2[:, :, 1:]

    # data1 = cbcr1[:, 1:]
    # data2 = cbcr2[:, 1:]

    # model = svm.LinearSVC(C=1, max_iter=1000)

    # data = np.concatenate((data1, data2), axis=0)
    # label = np.concatenate((np.ones((data1.shape[0], 1)), -np.ones((data2.shape[0], 1))), axis=0)

    # model.fit(data, label)
    # show_multiple_images_2d(model, data1, data2)
    # plt.show()

# sift = cv2.SIFT_create()
# kp = sift.detect(rotated_frame,None)

    fd = FretboardDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # ret, frame = cap.read()
        fd.apply(frame)
        cv2.waitKey(1)
    fd.video.release()
