from abc import ABC, abstractmethod

import numpy as np
import cv2
from itertools import combinations
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import skeletonize
from pathlib import Path

from .utils import utils_math as utils
from .utils.visualize import draw_houghline_batch, draw_fretboard, draw_houghlineP_batch

from .fretboard import Fretboard
from typing import Union, Tuple
from numpy.typing import NDArray
from cv2.typing import MatLike

class DetectorInterface(ABC):
    @abstractmethod
    def detect(self, frame: MatLike) -> Tuple[bool, Union[Fretboard, None]]:
        pass

from enum import Enum, auto
class DetectorStatus(Enum):
    boundaries_not_detected = auto()
    vanishing_point_not_detected = auto()
    homography_not_detected = auto()
    frets_localization_failed = auto()
    strings_localization_failed = auto()

    success = auto()
    error = auto()
    

class Detector(DetectorInterface):
    def __init__(self) -> None:
        super().__init__()

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        imagepath = str(Path('.').absolute() / 'play2tab' / 'video_processing' / 'template' /'string_template_2.png')
        self.string_template = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2GRAY)

        self.resize_width = 1920

    def preprocess(self, frame: MatLike, k_size=3) -> MatLike:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, k_size)
        gray = self.clahe.apply(gray)
        return gray
    
    def fretboard_boundary_detection(self, 
                                     edges: MatLike, 
                                     houghline_thres=280,
                                     theta_tol=5, 
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
                                     padding_percent=0.2) \
            -> Tuple[MatLike, MatLike, NDArray, NDArray]:
        rect = utils.oriented_bb_from_line_pair(fretboard_boundaries, gray.shape)
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
    
    def find_frets_vanishing_point_with_lsd(self, gray: MatLike):
        from .utils.rectification import edgelets_from_linesP, ransac_vanishing_point, reestimate_model

        lsd = cv2.ximgproc.createFastLineDetector()
        lsd_linesP = lsd.detect(gray)

        hb = utils.HoughBundler()
        lsd_linesP = hb.process_lines(lsd_linesP)

        delta_x = np.abs(lsd_linesP[:, 0, 0] - lsd_linesP[:, 0, 2])
        delta_y = np.abs(lsd_linesP[:, 0, 1] - lsd_linesP[:, 0, 3])
        linesP_vertical = lsd_linesP[delta_x <= delta_y]

        edgelets = edgelets_from_linesP(linesP_vertical)
        vp_frets, _ = ransac_vanishing_point(edgelets, 500, threshold_inlier=5)
        vp_frets, vp_inliers = reestimate_model(vp_frets, edgelets, 5)

        # line_on_image = lsd.drawSegments(gray, linesP_vertical)
        # cv2.imshow('total lsd lines', line_on_image)
        # vis_model(gray, edgelets1, vp_frets)
        return vp_frets, linesP_vertical[vp_inliers]
    
    def find_frets_vanishing_point_with_houghlines(self, edges: MatLike, theta_range=20/180*np.pi):
        tested_angles = np.linspace(-theta_range, theta_range, 20, endpoint=False)
        h, theta, d = hough_line(edges, theta=tested_angles)
        hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10)

        frets = np.concatenate((dists[:, np.newaxis], angles[:, np.newaxis]), axis=1)

        vp_frets, vp_inliers = utils.ransac_vanishing_point_estimation_lines(frets)
        return vp_frets, frets[vp_inliers]

    def find_frets_vanishing_point(self, gray: MatLike, edges: MatLike):
        vp_frets, frets_inlier = self.find_frets_vanishing_point_with_lsd(gray)
        if vp_frets is None:
            vp_frets, frets_inlier = self.find_frets_vanishing_point_with_houghlines(edges)
        return vp_frets, frets_inlier
    
    def merge_closely_spaced_lines(self, lines_length, dists, lines):
        idxes = np.argwhere(-np.diff(dists) < 10).reshape(-1,)
        if len(idxes) > 0:
            merge_groups = utils.find_consecutive_groups(idxes)
            for group in merge_groups:
                group.append(max(group)+1)
                weights = lines_length[group] / np.sum(lines_length[group])
                
                dists[group[0]] = np.sum(dists[group] * weights)
                lines[group[0]] = (weights.reshape((1,-1)) @ lines[group]).reshape(-1,)
            dists = np.delete(dists, idxes+1)
            lines = np.delete(lines, idxes+1, axis=0)
        return dists, lines
    
    def locate_frets(self, gray, frets_inlier, is_visualize=False):
        # convert frets to houghlines and store its length as weights
        if frets_inlier.shape[1] > 2:
            lines_length = np.sqrt((frets_inlier[:, 0] - frets_inlier[:, 2])**2 + (frets_inlier[:, 1] - frets_inlier[:, 3])**2)
            lines = utils.linesP_to_houghlines(frets_inlier[:, None, :], sort=False)
        else:
            lines_length = np.ones((frets_inlier.shape[0]))
            lines = frets_inlier
        dists = utils.houghlines_x_from_y(lines, gray.shape[0]/2)

        # sort lines from right to left
        sort_ind = np.argsort(dists)[::-1]
        lines = lines[sort_ind]
        lines_length = lines_length[sort_ind]
        dists = dists[sort_ind]

        dists, lines = self.merge_closely_spaced_lines(lines_length, dists, lines)

        if is_visualize:
            cdst = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # draw_houghline_batch(cdst, np.hstack((dists.reshape((-1, 1)), np.zeros((dists.shape[0], 1)))))
            draw_houghline_batch(cdst, lines)

        c = 2**(-1/12) # theoretical fret spacing ratio
        ratio_tol_low = -0.2
        ratio_tol_high = 0.2

        lines_spacing = -np.diff(dists)
        spacing_ratio = lines_spacing[1:] / lines_spacing[:-1]

        valid_condition = lambda x: x >= c + ratio_tol_low and x <= c + ratio_tol_high
        begin_ind, max_subset = utils.find_longest_consecutive_subset(spacing_ratio, valid_condition)
        valid_ind = np.arange(begin_ind, begin_ind + len(max_subset) + 2)

        grow_start_flag = True
        while grow_start_flag:
            start_index = valid_ind[0]
            next_index = valid_ind[1]
            high_bound = dists[start_index] + 1/(c+ratio_tol_low) * (dists[start_index] - dists[next_index])
            low_bound = dists[start_index] + 1/(c+ratio_tol_high) * (dists[start_index] - dists[next_index])
            theoretical = dists[start_index] + 1/c * (dists[start_index] - dists[next_index])
            
            inliers = (dists >= low_bound) & (dists <= high_bound)
            if np.sum(inliers) > 0:
                if np.sum(inliers) == 1:
                    valid_ind = np.hstack(([np.where(inliers)[0][0]], valid_ind))
                else:
                    # pick the one closest to the theoretical ratio
                    valid_ind = np.hstack(([np.where(inliers)[0][np.argmin(np.abs(dists[inliers] - theoretical))]], valid_ind))
            else:
                grow_start_flag = False
        if is_visualize:
            cv2.line(cdst, (int(high_bound), 50), (int(high_bound), 100), (0, 255, 255), 1)
            cv2.line(cdst, (int(low_bound), 50), (int(low_bound), 100), (0, 255, 255), 1)
        
        grow_end_flag = True
        while grow_end_flag:
            end_index = valid_ind[-1]
            prev_index = valid_ind[-2]

            if dists[prev_index] - dists[end_index] < 40:
                adaptive_ratio_tol_high = ratio_tol_high + 0.1
                adaptive_ratio_tol_low = ratio_tol_low - 0.1
            else:
                adaptive_ratio_tol_high = ratio_tol_low
                adaptive_ratio_tol_low = ratio_tol_high
            high_bound = dists[end_index] - (c+adaptive_ratio_tol_low)*(dists[prev_index] - dists[end_index])
            low_bound = dists[end_index] - (c+adaptive_ratio_tol_high)*(dists[prev_index] - dists[end_index])
            theoretical = dists[end_index] - c*(dists[prev_index] - dists[end_index])

            inliers = (dists >= low_bound) & (dists <= high_bound)
            if np.sum(inliers) > 0:
                if np.sum(inliers) == 1:
                    valid_ind = np.hstack((valid_ind, [np.where(inliers)[0][0]]))
                else:
                    valid_ind = np.hstack((valid_ind, [np.where(inliers)[0][np.argmin(np.abs(dists[inliers] - theoretical))]]))
            else:
                grow_end_flag = False
        if is_visualize:
            cv2.line(cdst, (int(high_bound), 50), (int(high_bound), 100), (0, 255, 255), 1)
            cv2.line(cdst, (int(low_bound), 50), (int(low_bound), 100), (0, 255, 255), 1)
            for idx in valid_ind:
                cv2.line(cdst, (int(dists[idx]), 0), (int(dists[idx]), 20), (0, 0, 255), 1)
            cv2.imshow('frets localization debug', cdst)
        
        if len(valid_ind) < 12:
            return None
        return np.vstack((dists[valid_ind], np.zeros_like(valid_ind))).T
    
    def locate_frets_thick_frets(self, gray:MatLike, frets_inlier, is_visualize=False):
        from .utils.rectification import edgelets_from_linesP, ransac_vanishing_point, reestimate_model

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 15, 0)
        edges = skeletonize(thresh).astype("uint8") * 255

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, None, 30, 20)
        if linesP is None:
            return None

        # filter by angle
        linesP = linesP[np.abs(linesP[:, 0, 0] - linesP[:, 0, 2]) <= np.abs(linesP[:, 0, 1] - linesP[:, 0, 3]), ...]
        if linesP.shape[0] == 0:
            return None
        
        edgelets = edgelets_from_linesP(linesP)
        vp_frets, _ = ransac_vanishing_point(edgelets, 500, threshold_inlier=5)
        vp_frets, vp_inliers = reestimate_model(vp_frets, edgelets, 5)
        if vp_frets is None:
            return None

        homography = utils.find_one_point_rectification(vp_frets)

        second_rect_gray = cv2.warpPerspective(gray, homography, gray.shape[1::-1], flags=cv2.INTER_LINEAR)
        frets_inlier[:, :2] = utils.transform_points(frets_inlier[:, :2], homography)
        frets_inlier[:, 2:] = utils.transform_points(frets_inlier[:, 2:], homography)
        linesP = frets_inlier[:, None, :]

        # houghlinesp to hough line coordinates
        lines = utils.linesP_to_houghlines(linesP, sort=True)
        dists = utils.houghlines_x_from_y(lines, gray.shape[0]/2)
        lines_length = np.ones_like(dists)

        if is_visualize:
            cdst = cv2.cvtColor(second_rect_gray, cv2.COLOR_GRAY2BGR)
            draw_houghlineP_batch(cdst, linesP)
            cv2.imshow('detected lines thick', cdst)

        # sort lines from right to left
        sort_ind = np.argsort(dists)[::-1]
        lines = lines[sort_ind]
        lines_length = lines_length[sort_ind]
        dists = dists[sort_ind]

        dists, lines = self.merge_closely_spaced_lines(lines_length, dists, lines)

        c = 2**(-1/12) # theoretical fret spacing ratio
        ratio_tol_low = -0.15
        ratio_tol_high = 0.15

        lines_spacing = -np.diff(dists)
        spacing_ratio = lines_spacing[1:] / lines_spacing[:-1]

        valid_condition = lambda x: x >= c + ratio_tol_low and x <= c + ratio_tol_high
        begin_ind, max_subset = utils.find_longest_consecutive_subset(spacing_ratio, valid_condition)
        valid_ind = np.arange(begin_ind, begin_ind + len(max_subset) + 2)

        grow_start_flag = True
        while grow_start_flag:
            start_index = valid_ind[0]
            next_index = valid_ind[1]
            high_bound = dists[start_index] + 1/(c+ratio_tol_low) * (dists[start_index] - dists[next_index])
            low_bound = dists[start_index] + 1/(c+ratio_tol_high) * (dists[start_index] - dists[next_index])
            theoretical = dists[start_index] + 1/c * (dists[start_index] - dists[next_index])
            
            inliers = (dists >= low_bound) & (dists <= high_bound)
            if np.sum(inliers) > 0:
                if np.sum(inliers) == 1:
                    valid_ind = np.hstack(([np.where(inliers)[0][0]], valid_ind))
                else:
                    # pick the one closest to the theoretical ratio
                    valid_ind = np.hstack(([np.where(inliers)[0][np.argmin(np.abs(dists[inliers] - theoretical))]], valid_ind))
            else:
                grow_start_flag = False
        if is_visualize:
            cv2.line(cdst, (int(high_bound), 50), (int(high_bound), 100), (0, 255, 255), 1)
            cv2.line(cdst, (int(low_bound), 50), (int(low_bound), 100), (0, 255, 255), 1)
        
        grow_end_flag = True
        while grow_end_flag:
            end_index = valid_ind[-1]
            prev_index = valid_ind[-2]

            if dists[prev_index] - dists[end_index] < 40:
                adaptive_ratio_tol_high = ratio_tol_high + 0.15
                adaptive_ratio_tol_low = ratio_tol_low - 0.15
            else:
                adaptive_ratio_tol_high = ratio_tol_low
                adaptive_ratio_tol_low = ratio_tol_high
            high_bound = dists[end_index] - (c+adaptive_ratio_tol_low)*(dists[prev_index] - dists[end_index])
            low_bound = dists[end_index] - (c+adaptive_ratio_tol_high)*(dists[prev_index] - dists[end_index])
            theoretical = dists[end_index] - c*(dists[prev_index] - dists[end_index])

            inliers = (dists >= low_bound) & (dists <= high_bound)
            if np.sum(inliers) > 0:
                if np.sum(inliers) == 1:
                    valid_ind = np.hstack((valid_ind, [np.where(inliers)[0][0]]))
                else:
                    valid_ind = np.hstack((valid_ind, [np.where(inliers)[0][np.argmin(np.abs(dists[inliers] - theoretical))]]))
            else:
                grow_end_flag = False
        if is_visualize:
            cv2.line(cdst, (int(high_bound), 50), (int(high_bound), 100), (0, 255, 255), 1)
            cv2.line(cdst, (int(low_bound), 50), (int(low_bound), 100), (0, 255, 255), 1)
            for idx in valid_ind:
                cv2.line(cdst, (int(dists[idx]), 0), (int(dists[idx]), 20), (0, 0, 255), 1)
            cv2.imshow('frets localization debug thick', cdst)
        if len(valid_ind) < 12:
            return None
        return utils.transform_houghlines(np.vstack((dists[valid_ind], np.zeros_like(valid_ind))).T, homography)
    
    def locate_strings(self, gray: MatLike, frets: NDArray, 
                       strings_num=6, dist_percent_tol=0.5, is_visualize=False):
        roi_gray = gray[:, int(frets[-1, 0]):int(frets[0, 0])]

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
            return None
        
        lengths_idx = np.argsort((linesP[:, 0, 2] - linesP[:, 0, 0])**2 + (linesP[:, 0, 3] - linesP[:, 0, 1])**2)
        linesP = linesP[lengths_idx[-1:-1-6:-1]]
        lines = utils.linesP_to_houghlines(linesP)

        left_y = utils.houghlines_y_from_x(lines, 0)
        right_y = utils.houghlines_y_from_x(lines, edges.shape[1])

        # no intersection allowed
        if np.sum(np.diff(left_y) < 0) & np.sum(np.diff(right_y) < 0) > 0:
            return None
        
        # approximately equal distance between strings
        middle_y_diff = (np.diff(left_y) + np.diff(right_y)) / 2
        if np.sum((middle_y_diff[1:] - middle_y_diff[0]) / middle_y_diff[0] > dist_percent_tol) > 0:
            return None
        
        offset_mat = np.array([
            [1, 0, int(frets[0, 0]) + w/2],
            [0, 1, h/2],
            [0, 0, 1]
        ])
        return utils.transform_houghlines(lines, np.linalg.inv(offset_mat))

    
    def detect_(self, frame: MatLike, 
               bb=None, fretboard_boundaries=None, 
               is_resize=True, is_visualize=False):
        if fretboard_boundaries is not None:
            fretboard_boundaries = np.array(fretboard_boundaries)

        if is_resize:
            factor = self.resize_width / frame.shape[1]
            frame = cv2.resize(frame, (0, 0), fx=factor, fy=factor)
            if bb is not None:
                bb = tuple([int(factor*x) for x in bb])
            if fretboard_boundaries is not None:
                fretboard_boundaries = factor*fretboard_boundaries

        if bb is not None:
            frame = frame[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], :]
            if fretboard_boundaries is not None:
                fretboard_boundaries = fretboard_boundaries - [bb[0], bb[1]]
        
        gray = self.preprocess(frame)

        edges_canny = cv2.Canny(gray, 30, 100, 5)
        
        if fretboard_boundaries is None:
            fretboard_boundaries, _ = self.fretboard_boundary_detection(edges_canny)
        else:
            fretboard_boundaries = utils.linesP_to_houghlines(np.expand_dims(fretboard_boundaries, axis=1))
        if fretboard_boundaries is None:
            return DetectorStatus.boundaries_not_detected, None
        
        cropped_gray, cropped_edges, \
            cropped_fretboard_boundaries, crop_mat = self.crop_from_fretboard_boundary(gray, edges_canny, fretboard_boundaries)

        # Optional, mask out the background
        # cornerpoints = cornerpoints_from_line_pair(cropped_fretboard_boundaries, cropped_gray.shape)
        # mask = np.zeros_like(cropped_gray)
        # cv2.fillPoly(mask, pts=[cornerpoints], color=(255, 255, 255))
        # cropped_gray = cv2.bitwise_and(cropped_gray, mask)
        # cropped_gray = self.clahe.apply(cropped_gray)

        vp_frets, frets_inlier = self.find_frets_vanishing_point(cropped_gray, cropped_edges)
        if vp_frets is None:
            return DetectorStatus.vanishing_point_not_detected, None
        
        vp_strings = utils.line_line_intersection(cropped_fretboard_boundaries[0], cropped_fretboard_boundaries[1])
        homography = utils.compute_homography(cropped_gray, np.hstack((vp_strings, 1)), vp_frets, clip=True)
        if homography is None:
            return DetectorStatus.homography_not_detected, None
        
        rect_gray = cv2.warpPerspective(cropped_gray, homography, cropped_edges.shape[1::-1], flags=cv2.INTER_LINEAR)
        # frets_inlier can be either lsd_lines or houghlines
        if frets_inlier.ndim > 2: # lsd_lines
            frets_inlier = frets_inlier.squeeze()
            frets_inlier[:, :2] = utils.transform_points(frets_inlier[:, :2], homography)
            frets_inlier[:, 2:] = utils.transform_points(frets_inlier[:, 2:], homography)
        else: # houghlines
            frets_inlier = utils.transform_houghlines(frets_inlier, np.linalg.inv(homography))
        
        rect_frets= self.locate_frets(rect_gray, frets_inlier, is_visualize=True)
        if rect_frets is None:
            rect_frets= self.locate_frets_thick_frets(rect_gray, frets_inlier, is_visualize=True)
        if rect_frets is None:
            return DetectorStatus.frets_localization_failed, None
        
        # rect_strings = self.locate_strings(rect_gray, rect_frets, is_visualize=False)
        rect_strings = None
        if rect_strings is None:
            rect_strings = utils.transform_houghlines(cropped_fretboard_boundaries, np.linalg.inv(homography))
        
        total_mat = homography @ crop_mat
        frets = utils.transform_houghlines(rect_frets, total_mat)
        strings = utils.transform_houghlines(rect_strings, total_mat)
        oriented_bb = utils.oriented_bb_from_frets_strings(frets, strings)
        oriented_bb = (oriented_bb[0], (oriented_bb[1][0] + 50, oriented_bb[1][1] + 100), oriented_bb[2])
        fretboard = Fretboard(frets, strings, oriented_bb)
        
        if bb is not None:
            fretboard = utils.transform_fretboard(fretboard, bb)
        if is_resize:
            fretboard = fretboard.resize(factor=1/factor)
        return DetectorStatus.success, fretboard

    def detect(self, *args, **kwargs):
        try:
            status, result = self.detect_(*args, **kwargs)
            return status, result
        except Exception as e:
            print(e)
            return DetectorStatus.error, None


class HandDetector(DetectorInterface):
    def __init__(self):
        super().__init__()

        from .utils.utils_math import transform_points
        import mediapipe as mp
        # mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        self.model_hands = mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence=0.8, max_num_hands=1)
        self.transform_points = transform_points
    
    def hand_position_estimation(self, frame, fretboard):
        cornerpoints = utils.cornerpoints_from_frets_strings(fretboard.frets, fretboard.strings).astype(int)
        h = abs(cornerpoints[1, 1] - cornerpoints[0, 1])
        w = abs(cornerpoints[2, 0] - cornerpoints[0, 0])
        target_cornerpoints = np.array([
            [0, 0],
            [0, h],
            [w, h],
            [w, 0]
        ])

        M, mask = cv2.findHomography(cornerpoints, target_cornerpoints, cv2.RANSAC, 5.0)
        rect_img = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)

        # R-X color anomaly detection
        choices = np.random.randint(0, rect_img.shape[0]*rect_img.shape[0], 10000)
        color_mean = np.mean(rect_img.reshape((-1, 3))[choices], axis=0)
        color_invvar = np.linalg.inv((rect_img.reshape((-1, 3))[choices] - color_mean).T @ (rect_img.reshape((-1, 3))[choices] - color_mean))
        mahalanobis = np.sum(((rect_img.reshape((-1, 3)) - color_mean) @ color_invvar) * (rect_img.reshape((-1, 3)) - color_mean), axis=1)

        r, c = np.where((mahalanobis >= 0.08**2).reshape(h, w))

        center = np.array([np.median(c), np.median(r)]).reshape((1, 2))

        if np.isnan(center[0][0]):
            return None

        return utils.transform_points(center, np.linalg.inv(M)).reshape(-1,).astype(int)
    
    def detect(self, frame, fretboard=None, bb=None, prev_hands=None):
        if bb is not None:
            frame_bb = frame[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], :]
            cv2.imshow('hey', frame_bb)
            cv2.waitKey(1)
            frame_bb = cv2.cvtColor(frame_bb, cv2.COLOR_BGR2RGB)
            results = self.model_hands.process(frame_bb)
            return self.hand_mediapipe_to_numpy(results, frame_bb, bb)
    
        if prev_hands is not None:
            prev_hands_bb = cv2.boundingRect(prev_hands[0])
            prev_size = max([prev_hands_bb[2], prev_hands_bb[3]])
            hand_pos = np.mean(prev_hands[0], axis=0)
            
            size = int(1.5*prev_size)
            bb = [int(hand_pos[0]-0.5*size), int(hand_pos[1]-0.5*size), size, size]
            frame_bb = frame[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], :]
            cv2.imshow('hey', frame_bb)
            cv2.waitKey(1)
            frame_bb = cv2.cvtColor(frame_bb, cv2.COLOR_BGR2RGB)
            results = self.model_hands.process(frame_bb)
            return self.hand_mediapipe_to_numpy(results, frame_bb, bb)
        
        # if fretboard is None:
        cv2.imshow('hey', frame)
        cv2.waitKey(1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model_hands.process(frame)
        return self.hand_mediapipe_to_numpy(results, frame)
        
        # hand_pos = self.hand_position_estimation(frame, fretboard)
        # size = int(1.2*fretboard.oriented_bb[1][1])
        # bb = [int(hand_pos[0]-0.5*size), int(hand_pos[1]-0.5*size), size, size]
        # frame_bb = frame[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], :]
        # frame_bb = cv2.cvtColor(frame_bb, cv2.COLOR_BGR2RGB)
        # results = self.model_hands.process(frame_bb)
        # return self.hand_mediapipe_to_numpy(results, frame_bb, bb)
    
    def hand_mediapipe_to_numpy(self, results, frame, bb=None):
        if results.multi_hand_landmarks:
            h, w = frame.shape[0], frame.shape[1]
            hands = [[] for _ in range(len(results.multi_hand_landmarks))]
            for i, hand in enumerate(results.multi_hand_landmarks):
                for idx in range(21):
                    hands[i].append([int(hand.landmark[idx].x * w), int(hand.landmark[idx].y * h)])
            hands = np.array(hands)

            if bb is not None:
                shift_mat = np.eye(3)
                shift_mat[0, 2] = -bb[0]
                shift_mat[1, 2] = -bb[1]
                for i in range(len(results.multi_hand_landmarks)):
                    hands[i, ...] = self.transform_points(hands[i, ...], np.linalg.inv(shift_mat)).astype(int)
            return hands
        else:
            return None