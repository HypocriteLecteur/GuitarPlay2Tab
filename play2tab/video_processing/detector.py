from abc import ABC, abstractmethod

import numpy as np
import cv2
from itertools import combinations
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import skeletonize

import utils.utils_math as utils
from utils.visualize import draw_houghline_batch, draw_fretboard

from fretboard import Fretboard
from typing import Union, Tuple
from numpy.typing import NDArray
from cv2.typing import MatLike

class DetectorInterface(ABC):
    @abstractmethod
    def detect(self, frame: MatLike) -> Tuple[bool, Union[Fretboard, None]]:
        pass

class Detector(DetectorInterface):
    def __init__(self) -> None:
        super().__init__()

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        self.string_template = cv2.cvtColor(cv2.imread('play2tab\\video_processing\\template\\string_template_2.png'), cv2.COLOR_BGR2GRAY)

        self.fretboard = None

    def preprocess(self, frame: MatLike, k_size=3) -> MatLike:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, k_size)
        gray = self.clahe.apply(gray)
        return gray
    
    def fretboard_boundary_detection(self, 
                                     edges: MatLike, 
                                     houghline_thres=280,
                                     theta_tol=9, 
                                     rho_tol=50, 
                                     is_visualize=False) \
            -> Tuple[Union[NDArray, None], Union[MatLike, None]]:
        '''
        Detect near-parallel line pair that is the farthest apart.
        '''
        lines = cv2.HoughLines(edges, 1, np.pi / 180, houghline_thres, None, 0, 0)

        if lines is None:
            return None, None
        
        max_dist = 0
        fretboard_boundary = None
        for [[ri, ti], [rj, tj]] in combinations(zip(lines[:, 0, 0], lines[:, 0, 1]), 2):
            if abs(ri - rj) > rho_tol and abs(ri - rj) > max_dist and utils.angle_diff(ti, tj) < theta_tol / 180 * np.pi:
                fretboard_boundary = np.array([[ri, ti], [rj, tj]])
                max_dist = abs(ri - rj)

        if is_visualize and fretboard_boundary is not None:
            cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            draw_houghline_batch(cdst, fretboard_boundary)
            cv2.imshow("Detected Lines", cdst)
            return fretboard_boundary, cdst
        return fretboard_boundary, None
    
    def crop_from_fretboard_boundary(self, 
                                     gray: MatLike,
                                     edges: MatLike,
                                     fretboard_boundaries: NDArray, 
                                     rect: Tuple,
                                     padding_percent=0.2) \
            -> Tuple[MatLike, MatLike, NDArray, NDArray]:
        if rect[1][1] > rect[1][0]:
            padding = padding_percent*rect[1][0]
            padded_rect = (rect[0], (rect[1][0]+2*padding, rect[1][1]), rect[2])
        else:
            padding = padding_percent*rect[1][1]
            padded_rect = (rect[0], (rect[1][0], rect[1][1]+2*padding), rect[2])

        cropped_gray, crop_mat = utils.crop_from_oriented_bb(gray, padded_rect)
        cropped_edges, _ = utils.crop_from_oriented_bb(edges, padded_rect)
        cropped_fretboard_boundaries = utils.transform_houghlines(fretboard_boundaries, np.linalg.inv(crop_mat))
        return cropped_gray, cropped_edges, cropped_fretboard_boundaries, crop_mat
    
    def find_rectification(self, cropped_edges: MatLike, theta_range=10/180*np.pi, is_visualize=False) \
            -> Tuple[NDArray, NDArray, Union[MatLike, None]]:
        '''
        Detect vertical frets and rectify.
        '''
        tested_angles = np.linspace(-theta_range, theta_range, 10, endpoint=False)
        h, theta, d = hough_line(cropped_edges, theta=tested_angles)
        hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10)

        frets = np.concatenate((dists[:, np.newaxis], angles[:, np.newaxis]), axis=1)

        # sort by dist
        # hspace, angles, dists = (list(item) for item in zip(*sorted(zip(hspace, angles, dists), key=lambda x: x[1])))
        
        homography, inliers = utils.find_affine_rectification(frets, cropped_edges.shape)
        if homography is None:
            return None, None, None

        if is_visualize and dists is not None:
            cdst = cv2.cvtColor(cropped_edges.copy(), cv2.COLOR_GRAY2BGR)
            inliers_ = (inliers.squeeze()[::2]) & (inliers.squeeze()[1::2])
            draw_houghline_batch(cdst, frets[inliers_ == True, :], color=(0, 255, 0))
            draw_houghline_batch(cdst, frets[inliers_ == False, :], color=(0, 0, 255))
            cv2.imshow('Detected Frets', cdst)
            return homography, frets, cdst
        return homography, frets, None
    
    def locate_frets(self, 
                         rect_gray: MatLike, 
                         frets_num=20, 
                         merge_thres=10, 
                         validation_thres=0.2, 
                         is_visualize=False) \
            -> Tuple[bool, Union[NDArray, None]]:
        '''
        1. Detect vertical lines
        2. Merge closely spaced lines
        3. Delete fretline outliers by calculating second order difference

        Note: One can also use the detection results from find_rectification, but for locating thick frets, skeletonization
        works better than canny.
        '''
        thresh = cv2.adaptiveThreshold(rect_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 11, 0)
        edges = skeletonize(thresh).astype("uint8") * 255

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, None, 30, 20)

        # filter by angle
        linesP = linesP[2*np.abs(linesP[:, 0, 0] - linesP[:, 0, 2]) <= np.abs(linesP[:, 0, 1] - linesP[:, 0, 3]), ...]

        # houghlinesp to hough line coordinates
        lines = utils.linesP_to_houghlines(linesP)
        dists = utils.houghlines_x_from_y(lines, rect_gray.shape[0]/2)
        
        # second stage filter (merge closely spaced lines)
        idxes = np.argwhere(np.diff(dists) < merge_thres).squeeze()
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

        if is_visualize:
            cdst = cv2.cvtColor(rect_gray, cv2.COLOR_GRAY2BGR)
            draw_houghline_batch(cdst, lines)

            # cv2.namedWindow("locate_fretboard_debug")
            # def mouse_callback(event, x, y, flags, param):
            #     idx = np.argmin(np.abs(np.array(dists) - x))
            #     cdst2 = cdst.copy()
            #     if abs(dists[idx] - x ) < 5:
            #         draw_houghline(cdst2, [dists[idx], 0], thickness=3)
            #         cv2.putText(cdst2, f'{dists[idx]}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            #     cv2.imshow('locate_fretboard_debug', cdst2)
            # cv2.setMouseCallback("locate_fretboard_debug", mouse_callback)
        
        fret_dists = np.diff(dists)
        fret_second_order_difference_normalized = np.diff(fret_dists) / fret_dists[0:-1]
        
        valid_condition = lambda x: abs(x) < validation_thres
        subset_begin_index, max_subset = utils.find_longest_consecutive_subset(fret_second_order_difference_normalized, 
                                                                         valid_condition)
        
        is_located = False
        if len(max_subset) + 1 >= frets_num:
            is_located = True
            frets_idx = np.arange(subset_begin_index, subset_begin_index + len(max_subset) + 2)
            if is_visualize:
                for idx in frets_idx:
                    cv2.line(cdst, (int(dists[idx]), 0), (int(dists[idx]), 20), (0, 0, 255), 1)
                cv2.imshow('locate_fretboard_debug', cdst)
        if is_located:
            return True, np.vstack((dists[frets_idx], np.zeros_like(frets_idx))).T
        else:
            return False, None
    
    def locate_strings(self, 
                       rect_gray: MatLike, 
                       frets: NDArray, 
                       strings_num=6, 
                       dist_percent_tol=0.5, 
                       is_visualize=True) \
            -> Tuple[bool, Union[NDArray, None]]:
        '''
        1. Template matching
        2. Edge detection for matching response map (remember theres an offset)
        3. HoughlinesP and merge similar lines
        4. Pick the longest {strings_num} lines
        5. Validation
        6. Compensate offset
        '''
        roi_gray = rect_gray[:, int(frets[0, 0]):int(frets[-1, 0])]

        w, h = self.string_template.shape[::-1]
        template_res = cv2.matchTemplate(roi_gray, self.string_template, cv2.TM_CCOEFF_NORMED)

        template_res[template_res < 0] = 0
        template_res = (template_res / np.max(template_res) * 255).astype(np.uint8)

        thresh = cv2.adaptiveThreshold(template_res, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 5, 0)
        edges = skeletonize(thresh).astype("uint8") * 255

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, 30, 50)
        hb = utils.HoughBundler()
        linesP = hb.process_lines(linesP)

        if linesP is None or linesP.shape[0] < strings_num:
            return False, None
                
        lengths_idx = np.argsort((linesP[:, 0, 2] - linesP[:, 0, 0])**2 + (linesP[:, 0, 3] - linesP[:, 0, 1])**2)
        linesP = linesP[lengths_idx[-1:-1-6:-1]]
        lines = utils.linesP_to_houghlines(linesP)

        left_y = utils.houghlines_y_from_x(lines, 0)
        right_y = utils.houghlines_y_from_x(lines, rect_gray.shape[1])

        # no intersection allowed
        if np.sum(np.diff(left_y) < 0) & np.sum(np.diff(right_y) < 0) > 0:
            return False, None
        
        # approximately equal distance between strings
        middle_y_diff = (np.diff(left_y) + np.diff(right_y)) / 2
        if np.sum((middle_y_diff[1:] - middle_y_diff[0]) / middle_y_diff[0] > dist_percent_tol) > 0:
            return False, None
        
        offset_mat = np.array([
            [1, 0, int(frets[0, 0]) + w/2],
            [0, 1, h/2],
            [0, 0, 1]
        ])
        if is_visualize:
            cdst = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
            draw_houghline_batch(cdst, utils.transform_houghlines(lines, np.linalg.inv(offset_mat)))
            cv2.imshow('cdst', cdst)
        return True, utils.transform_houghlines(lines, np.linalg.inv(offset_mat))

    def detect(self, frame: MatLike, is_visualize=True) -> Tuple[bool, Union[Fretboard, None]]:
        '''
        Image Processing Pipeline:
        1. Preprocess: convert to gray -> denoising (median filter) -> enhance contrast (clahe)
        2. Edge Detection: canny
        3. Fretboard boundary detection: Hough Transform -> parallel lines detection
        4. Crop image with padded rotated bounding-box
        5. Mask out the background and enhance contrast (Optional)
        6. Affine Rectification, recover vertical parallelism
        7. Locate frets
        8. Locate strings
        9. Prepare data for tracker

        TODO:
        Currently initialization is complete only when frets and strings are both located.
        It's possible that strings are not visible under bad lighting condition.
        When string detection is not practical, it would be more desirable to track feature points on the fretboard instead.
        If possible, the algorithm should adapt to different lighting conditions and automaticallly decide whether to
        detect frets+strings or frets+feature points.
        Same goes for the tracker.
        '''
        gray = self.preprocess(frame)

        edges = cv2.Canny(gray, 30, 100, 5)

        fretboard_boundaries, _ = self.fretboard_boundary_detection(edges, theta_tol=5, rho_tol=50, is_visualize=False)
        if fretboard_boundaries is None:
            return False, None
        
        rect = utils.oriented_bb_from_line_pair(fretboard_boundaries, frame.shape)
        cropped_gray, cropped_edges, cropped_fretboard_boundaries, crop_mat = self.crop_from_fretboard_boundary(gray, edges, fretboard_boundaries, rect)

        # Optional, mask out the background
        # cornerpoints = cornerpoints_from_line_pair(cropped_fretboard_boundaries, cropped_gray.shape)
        # mask = np.zeros_like(cropped_gray)
        # cv2.fillPoly(mask, pts=[cornerpoints], color=(255, 255, 255))
        # cropped_gray = cv2.bitwise_and(cropped_gray, mask)
        # cropped_gray = self.clahe.apply(cropped_gray)

        homography, _, _ = self.find_rectification(cropped_edges, is_visualize=False)
        if homography is None:
            return False, None
        
        rect_gray = cv2.warpPerspective(cropped_gray, homography, cropped_edges.shape[1::-1], flags=cv2.INTER_LINEAR)

        is_frets_located, rect_frets= self.locate_frets(rect_gray, is_visualize=False)
        if is_frets_located:
            is_strings_located, rect_strings = self.locate_strings(rect_gray, rect_frets, is_visualize=False)
            if is_strings_located:
                total_mat = homography @ crop_mat

                frets = utils.transform_houghlines(rect_frets, total_mat)
                strings = utils.transform_houghlines(rect_strings, total_mat)

                rect = utils.oriented_bb_from_frets_strings(frets, strings)
                rect = (rect[0], (rect[1][0] + 50, rect[1][1] + 100), rect[2])

                fretboard = Fretboard(frets, strings, rect)

                if is_visualize:
                    cdst = frame.copy()
                    draw_fretboard(cdst, fretboard)
                    cv2.imshow('cdst', cdst)
                return True, fretboard
        return False, None