from abc import ABC, abstractmethod

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.transform import SimilarityTransform
from skimage.measure import ransac
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor, LinearRegression
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from pathlib import Path

from .utils import utils_math as utils
from .utils.visualize import draw_houghline_batch, draw_fretboard, draw_houghlineP_batch

from .fretboard import Fretboard
from typing import Union, Tuple
from numpy.typing import NDArray
from cv2.typing import MatLike
from skimage.morphology import skeletonize

class TrackerInterface(ABC):
    @abstractmethod
    def track(self, frame: MatLike) -> Tuple[bool, Union[Fretboard, None]]:
        pass

class Tracker(TrackerInterface):
    def __init__(self) -> None:
        super().__init__()

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        imagepath = str(Path('.').absolute() / 'play2tab' / 'video_processing' / 'template' /'string_template.png')
        self.string_template = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2GRAY)

        self.fretboard = None

    def preprocess(self, frame: MatLike, k_size=3) -> MatLike:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, k_size)
        gray = self.clahe.apply(gray)
        return gray
    
    def frets_matching(self, 
                       rect_lines: NDArray, 
                       gray_rect: MatLike, 
                       frets: NDArray, 
                       transform_mat: NDArray,
                       notmatched_thres=30,
                       ransac_outlier_thres=5,
                       is_visualize=True) -> Tuple[NDArray, Tuple]:
        fret_dists = utils.houghlines_x_from_y(rect_lines, gray_rect.shape[0]/2)

        prev_lines = utils.transform_houghlines(frets, np.linalg.inv(transform_mat))
        prev_fret_dists = utils.houghlines_x_from_y(prev_lines, (gray_rect.shape[0])/2)

        cost_matrix = np.abs(prev_fret_dists.reshape((-1, 1)) - fret_dists.reshape((1, -1)))
        cost_matrix = np.concatenate((cost_matrix, np.ones((cost_matrix.shape[0], prev_fret_dists.shape[0])) * notmatched_thres), axis=1) # add dummy

        prev_ind, now_ind = linear_sum_assignment(cost_matrix)
        matched_prev_ind = prev_ind[now_ind < min(frets.shape[0], rect_lines.shape[0])]
        matched_now_ind = now_ind[now_ind < min(frets.shape[0], rect_lines.shape[0])]

        # inliers are consider matched
        now_tracked_frets = np.zeros_like(frets)
        now_tracked_frets[matched_prev_ind, :] = rect_lines[matched_now_ind, :]

        # estimate 1d similarity transform
        model_robust, inliers = ransac(
            (np.vstack((prev_fret_dists[matched_prev_ind], np.zeros((matched_prev_ind.shape[0],)))).T, 
            np.vstack((fret_dists[matched_now_ind], np.zeros((matched_now_ind.shape[0],)))).T), 
            SimilarityTransform, min_samples=2, residual_threshold=ransac_outlier_thres, max_trials=100)
        s = model_robust.params[0, 0]
        t = model_robust.params[0, 2]

        outliers = inliers == False
        if np.sum(outliers) > 0:
            now_tracked_frets[matched_prev_ind[outliers], :] = np.vstack((0.8*(s * prev_fret_dists[matched_prev_ind[outliers]] + t) + 0.2*(fret_dists[matched_now_ind][outliers]), 
                                                                          np.zeros((matched_prev_ind[outliers].shape[0])))).T

        notmatched_prev_ind = np.setdiff1d(np.arange(prev_fret_dists.shape[0]), matched_prev_ind)
        if notmatched_prev_ind.shape[0] > 0:
            now_tracked_frets[notmatched_prev_ind, :] = np.vstack((s * prev_fret_dists[notmatched_prev_ind] + t, 
                                                                   np.zeros((notmatched_prev_ind.shape[0])))).T
        if is_visualize:
            cdst = cv2.cvtColor(gray_rect, cv2.COLOR_GRAY2BGR)
            draw_houghline_batch(cdst, rect_lines)
            draw_houghline_batch(cdst, prev_lines, color=(255, 0, 0))
            draw_houghline_batch(cdst, now_tracked_frets, color=(255, 255, 0))
            # draw_houghline_batch(cdst, np.vstack((prev_fret_dists*s + t, np.zeros_like(prev_fret_dists))).T, color=(0, 255, 255))
            for i, j in zip(matched_prev_ind, matched_now_ind):
                cv2.line(cdst, (int(prev_lines[i, 0]), i*5+5), (int(rect_lines[j, 0]), i*5+5), (0, 255, 255), 3, cv2.LINE_AA)
            for i, j in zip(matched_prev_ind[outliers], matched_now_ind[outliers]):
                cv2.line(cdst, (int(prev_lines[i, 0]), i*5+5), (int(rect_lines[j, 0]), i*5+5), (255, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('frets_matching', cdst)
        return now_tracked_frets, (s, t)
    
    # def strings_locating(self, frets, gray, strings_num=6, is_visualize=False) -> Tuple[NDArray, NDArray, NDArray]:
    #     '''
    #     A simple template matching to locate top and bottom of frets, for some reason works wonderfully.
    #     '''
    #     test_template = np.zeros((70, 13), np.uint8)
    #     w, h = test_template.shape[::-1]
    #     test_template[:, 4:9] = 255

    #     res = cv2.matchTemplate(gray, test_template, cv2.TM_CCOEFF_NORMED)
    #     res[res<0.5] = 0
    #     coordinates = peak_local_max(res, min_distance=10) + [0, w/2]
    #     if coordinates.shape[0] < 10:
    #         res = cv2.matchTemplate(gray, test_template, cv2.TM_CCOEFF_NORMED)
    #         res[res<0.2] = 0
    #         coordinates = peak_local_max(res, min_distance=10) + [0, w/2]
        
    #     cdst = cv2.cvtColor((res / np.max(res) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    #     for pnt in coordinates:
    #         cv2.circle(cdst, (int(pnt[1]), int(pnt[0])), 3, (0, 0, 255), 2)
    #     cv2.imshow('strings_locating',cdst)

    #     if coordinates.shape[0] < 5:
    #         return None, None, None
    #     ransac = RANSACRegressor(estimator=LinearRegression())
    #     ransac.fit(coordinates[:, 1].reshape((-1, 1)), coordinates[:, 0].reshape((-1, 1)))
    #     upper_line = utils.houghline_from_kb(ransac.estimator_.coef_[0, 0], ransac.estimator_.intercept_[0])

    #     gray_flipped = cv2.flip(gray, 0)

    #     res = cv2.matchTemplate(gray_flipped, test_template, cv2.TM_CCOEFF_NORMED)
    #     res[res<0.5] = 0
    #     coordinates2 = peak_local_max(res, min_distance=10) + [0, w/2]
    #     if coordinates2.shape[0] < 10:
    #         res = cv2.matchTemplate(gray, test_template, cv2.TM_CCOEFF_NORMED)
    #         res[res<0.2] = 0
    #         coordinates2 = peak_local_max(res, min_distance=10) + [0, w/2]
        
    #     if coordinates2.shape[0] < 5:
    #         return None, None, None
    #     ransac = RANSACRegressor(estimator=LinearRegression())
    #     ransac.fit(coordinates2[:, 1].reshape((-1, 1)), (gray.shape[0] - coordinates2[:, 0]).reshape((-1, 1)))
    #     lower_line = utils.houghline_from_kb(ransac.estimator_.coef_[0, 0], ransac.estimator_.intercept_[0])

    #     # ransac.fit(((coordinates[:, 1] + coordinates2[:, 1])/2).reshape((-1, 1)), ((coordinates[:, 0]+gray.shape[0] - coordinates2[:, 0])/2).reshape((-1, 1)))
    #     left_point = np.array([0, (utils.houghlines_y_from_x(upper_line.reshape((1, 2)), 0)[0] + utils.houghlines_y_from_x(lower_line.reshape((1, 2)), 0)[0])/2, 1])
    #     right_point = np.array([gray.shape[1], (utils.houghlines_y_from_x(upper_line.reshape((1, 2)), gray.shape[1])[0] + utils.houghlines_y_from_x(lower_line.reshape((1, 2)), gray.shape[1])[0])/2, 1])
    #     line_cord = np.cross(left_point, right_point)

    #     t = np.arctan2(line_cord[1] / np.linalg.norm(line_cord[:2]), line_cord[0] / np.linalg.norm(line_cord[:2]))
    #     r = abs(line_cord[2]) / np.linalg.norm(line_cord[:2])
    #     mid_line = np.array([r, t])
    #     # mid_line = utils.houghline_from_kb(ransac.estimator_.coef_[0, 0], ransac.estimator_.intercept_[0])

    #     if is_visualize:
    #         cdst = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    #         # for i in range(fret_dists.size):
    #         #     cv2.circle(cdst, (int(fret_dists[i]), int(strings_start_y[i])+int(h/2)), 3, (0, 0, 255))
    #         #     cv2.circle(cdst, (int(fret_dists[i]), int(strings_end_y[i])+int(h/2)), 3, (0, 0, 255))
    #         #     cv2.circle(cdst, (int(fret_dists[i]), int((strings_start_y[i]+strings_end_y[i])/2)+int(h/2)), 3, (0, 0, 255))

    #         draw_houghline_batch(cdst, upper_line.reshape((1, 2)))
    #         draw_houghline_batch(cdst, mid_line.reshape((1, 2)))
    #         draw_houghline_batch(cdst, lower_line.reshape((1, 2)))
    #         # draw_houghline_batch(template_res_cdst, np.vstack((fret_dists.astype(int) - int(fret_dists[0]) - int(w/2), np.zeros_like(fret_dists))).T)
    #         cv2.imshow('strings cdst', cdst)
    #     return upper_line, mid_line, lower_line

    def strings_locating(self, frets, gray, strings_num=6, is_visualize=False) -> Tuple[NDArray, NDArray, NDArray]:
        '''
        Strings Localization Pipeline:
        1. fret_dists
        2. matching
        3. find spacing
        '''
        test_template = np.zeros((70, 13), np.uint8)
        test_template[:, 4:9] = 255
        res = cv2.matchTemplate(gray, test_template, cv2.TM_CCOEFF_NORMED)
        # res = (res > 0.5).astype(np.uint8) * 255
        res_padded = cv2.copyMakeBorder(res, 0, gray.shape[0]-res.shape[0], int(test_template.shape[1]/2), gray.shape[1]-res.shape[1]-int(test_template.shape[1]/2), 0)
        # plt.imshow(gray, alpha=0.5)
        # plt.imshow(res_padded, alpha=0.5)
        # plt.show()
        # cv2.imshow('res', res)
        # cv2.waitKey()

        w, h = self.string_template.shape[::-1]

        fret_dists = utils.houghlines_x_from_y(frets, gray.shape[0]/2)
        total_span = [int(fret_dists[0])-int(w/2), int(fret_dists[-1])+int(w/2)]

        if gray.shape[1] <= int(fret_dists[-1])+int(w/2):
            fret_dists = fret_dists[fret_dists.astype(int)+int(w/2) < gray.shape[1]]

        roi_gray = gray[:, total_span[0]:total_span[-1]]

        template_res = cv2.matchTemplate(roi_gray, self.string_template, cv2.TM_CCOEFF_NORMED)
        template_res_cdst = cv2.cvtColor(template_res, cv2.COLOR_GRAY2BGR)

        template_res[template_res < 0] = 0
        template_res = (template_res / np.max(template_res) * 255).astype(np.uint8)

        offset_fret_dists = fret_dists.astype(int) - int(fret_dists[0])
        res_frets = template_res[:, offset_fret_dists]

        peak_spacings = []
        valid_idx = []
        for i in range(fret_dists.size):
            peaks = find_peaks(res_frets[:, i])[0]
            if peaks.size > 1:
                peaks_dist = np.diff(peaks)
                if np.sum(np.abs((peaks_dist - np.mean(peaks_dist))) > 3*np.std(peaks_dist)) > 0:
                    print('peak finding error')
                peak_spacings.append(np.mean(peaks_dist))
                valid_idx.append(i)

        ransac = RANSACRegressor(estimator=LinearRegression())
        ransac.fit(offset_fret_dists[valid_idx].reshape((-1, 1)), np.array(peak_spacings).reshape((-1, 1)))
        peak_spacings_fit = offset_fret_dists*ransac.estimator_.coef_[0, 0] + ransac.estimator_.intercept_[0]
        if np.sum(peak_spacings_fit < 0) > 0:
            print('peak_spacings_fit error')

        # strings_start_y = []
        # strings_end_y = []
        # for i in range(fret_dists.size):
        #     peaks = find_peaks(res_frets[:, i])[0]
        #     strings_start_y.append(peaks[0])
        #     strings_end_y.append(peaks[-1])
        # strings_start_y = np.array(strings_start_y)
        # strings_end_y = np.array(strings_end_y)

        strings_start_y = []
        for i in range(fret_dists.size):
            max_y = np.argmax(np.correlate(res_frets[:, i], np.ones((int(peak_spacings_fit[i]*(strings_num-1)),)), mode='valid'))
            # plt.plot(res_frets[:, i])
            # plt.plot(np.arange(6)*peak_spacings_fit[i]+max_y, res_frets[(np.arange(6)*peak_spacings_fit[i]+max_y).astype(int), i])
            # plt.figure()
            # plt.plot(np.correlate(res_frets[:, i], np.ones((int(peak_spacings_fit[i]*(strings_num-1)),)), mode='valid'))
            # plt.show()

            peaks = find_peaks(res_frets[:, i])[0]
            if peaks.size > 1:
                max_y = peaks[np.argmin(np.abs(peaks-max_y))]
            
            strings_start_y.append(max_y)
        strings_start_y = np.array(strings_start_y)
        strings_end_y = strings_start_y + peak_spacings_fit * (strings_num-1)

        ransac = RANSACRegressor(estimator=LinearRegression())
        ransac.fit(offset_fret_dists.reshape((-1, 1)), strings_start_y.reshape((-1, 1)))
        upper_line = utils.houghline_from_kb(ransac.estimator_.coef_[0, 0], ransac.estimator_.intercept_[0])

        ransac.fit(offset_fret_dists.reshape((-1, 1)), strings_end_y.reshape((-1, 1)))
        lower_line = utils.houghline_from_kb(ransac.estimator_.coef_[0, 0], ransac.estimator_.intercept_[0])

        ransac.fit(offset_fret_dists.reshape((-1, 1)), ((strings_start_y + strings_end_y)/2).reshape((-1, 1)))
        mid_line = utils.houghline_from_kb(ransac.estimator_.coef_[0, 0], ransac.estimator_.intercept_[0])

        offset_mat = np.array([
            [1, 0, -int(fret_dists[0]) + int(w/2)],
            [0, 1, int(h/2)],
            [0, 0, 1]
        ])
        upper_line = utils.transform_houghlines(upper_line.reshape((1, -1)), np.linalg.inv(offset_mat)).squeeze()
        lower_line = utils.transform_houghlines(lower_line.reshape((1, -1)), np.linalg.inv(offset_mat)).squeeze()
        mid_line = utils.transform_houghlines(mid_line.reshape((1, -1)), np.linalg.inv(offset_mat)).squeeze()

        if is_visualize:
            cdst = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)        
            for i in range(fret_dists.size):
                cv2.circle(cdst, (int(fret_dists[i]), int(strings_start_y[i])+int(h/2)), 3, (0, 0, 255))
                cv2.circle(cdst, (int(fret_dists[i]), int(strings_end_y[i])+int(h/2)), 3, (0, 0, 255))
                cv2.circle(cdst, (int(fret_dists[i]), int((strings_start_y[i]+strings_end_y[i])/2)+int(h/2)), 3, (0, 0, 255))

            draw_houghline_batch(cdst, upper_line.reshape((1, 2)))
            draw_houghline_batch(cdst, mid_line.reshape((1, 2)))
            draw_houghline_batch(cdst, lower_line.reshape((1, 2)))
            # draw_houghline_batch(template_res_cdst, np.vstack((fret_dists.astype(int) - int(fret_dists[0]) - int(w/2), np.zeros_like(fret_dists))).T)
            # cv2.imshow('strings cdst', cdst)
        return upper_line, mid_line, lower_line

    def track(self, frame: MatLike, fgmask: MatLike, fretboard: Tuple, is_visualize=False) -> Tuple[bool, Union[Fretboard, None]]:
        '''
        Tracking Pipeline:
        1. Mask out background
        2. Preprocess: convert to gray -> denoising (median filter) -> enhance contrast (clahe)
        3. Edge Detection: threshold -> thinning
        4. HoughlinesP
        5. Partial affine rectification, find a vanishing point and recover vertical parallelism
        6. Frets Matching
        7. Strings Localization
        '''
        cropped_frame, crop_transform = utils.crop_from_oriented_bb(frame, fretboard.oriented_bb)
        if fgmask is not None:
            cropped_mask, _ = utils.crop_from_oriented_bb(fgmask, fretboard.oriented_bb)
            cropped_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=cropped_mask)

        gray = self.preprocess(cropped_frame)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 21, 0)
        edges = skeletonize(thresh).astype("uint8") * 255

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, None, 10, 20)
        linesP = linesP[np.abs(linesP[:, 0, 0] - linesP[:, 0, 2]) <= np.abs(linesP[:, 0, 1] - linesP[:, 0, 3]), ...]
        
        vanishing_point, vp_inliers = utils.ransac_vanishing_point_estimation(linesP)
        if vanishing_point is None:
            return False, None

        linesP = linesP[vp_inliers, ...]
        hb = utils.HoughBundler()
        linesP = hb.process_lines(linesP)
        
        lines = utils.linesP_to_houghlines(linesP)

        homography, _ = utils.find_rectification(lines, gray.shape)
        if homography is None:
            return False, None

        gray_rect = cv2.warpPerspective(gray, homography, gray.shape[1::-1], flags=cv2.INTER_LINEAR)

        rect_lines = utils.transform_houghlines(lines, np.linalg.inv(homography))

        now_tracked_frets, (s, t) = self.frets_matching(rect_lines, gray_rect, fretboard.frets, homography @ crop_transform)

        upper_line, mid_line, lower_line = self.strings_locating(now_tracked_frets.copy(), gray_rect)
        if upper_line is None:
            return False, None

        total_span = utils.houghlines_x_from_y(now_tracked_frets[[0, -1], :], gray_rect.shape[0]/2)

        center = fretboard.oriented_bb[0]
        center = utils.transform_points(np.array(center).reshape((1, 2)), homography @ crop_transform).squeeze()
        center[0] = np.mean(total_span)
        center[1] = utils.houghlines_y_from_x(mid_line.reshape((1, 2)), center[0])[0]
        center = utils.transform_points(center.reshape((1, 2)), np.linalg.inv(homography @ crop_transform)).squeeze()

        now_oriented_bb = ((center[0], center[1]), 
                           (fretboard.oriented_bb[1][0]*s, (total_span[1] - total_span[0]) + 100), 
                           fretboard.oriented_bb[2] + (mid_line[1]-np.pi/2*np.sign(mid_line[1]))*180/np.pi)

        now_fretboard = Fretboard(utils.transform_houghlines(now_tracked_frets, homography @ crop_transform), 
                                  utils.transform_houghlines(np.vstack((upper_line, lower_line)), homography @ crop_transform), 
                                  now_oriented_bb)
        
        if is_visualize:
            cdst = frame.copy()
            draw_fretboard(cdst, fretboard)
            cv2.imshow('cdst', cdst)
        return True, now_fretboard


import torch
class TrackerLightGlue(TrackerInterface):
    def __init__(self, resize_width=960):
        super().__init__()

        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
        torch.set_grad_enabled(False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        self.extractor = SuperPoint(max_num_keypoints=512).eval().to(self.device)  # load the extractor
        self.matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1, n_layers=7).eval().to(self.device)
        # self.matcher.compile(mode='reduce-overhead')
        self.rbd = rbd

        self.feats0 = None
        self.fretboard0 = None

        self.resize_width = resize_width
        
    def preprocess(self, frame):
        if self.resize_width is not None:
            factor = self.resize_width / frame.shape[1]
            frame = cv2.resize(frame, (0, 0), fx=factor, fy=factor)
        frame = frame[..., ::-1]
        frame = frame.transpose((2, 0, 1))  # HxWxC to CxHxW
        return torch.tensor(frame / 255.0, dtype=torch.float)
    
    def create_template(self, frame, fretboard):
        self.image0 = self.preprocess(frame)
        self.feats0 = self.extractor.extract(self.image0.to(self.device))
        self.fretboard0 = fretboard.copy()
        if self.resize_width is not None:
            factor = self.resize_width / frame.shape[1]
            self.fretboard0.resize(factor)
    
    def filter_match0(self, m_kpts0):
        sign0 = self.fretboard0.strings[0][0] - m_kpts0[:, 0]*np.cos(self.fretboard0.strings[0][1]) - m_kpts0[:, 1]*np.sin(self.fretboard0.strings[0][1])
        sign1 = self.fretboard0.strings[-1][0] - m_kpts0[:, 0]*np.cos(self.fretboard0.strings[-1][1]) - m_kpts0[:, 1]*np.sin(self.fretboard0.strings[-1][1])
        sign2 = self.fretboard0.frets[0][0] - m_kpts0[:, 0]*np.cos(self.fretboard0.frets[0][1]) - m_kpts0[:, 1]*np.sin(self.fretboard0.frets[0][1])
        sign3 = self.fretboard0.frets[-1][0] - m_kpts0[:, 0]*np.cos(self.fretboard0.frets[-1][1]) - m_kpts0[:, 1]*np.sin(self.fretboard0.frets[-1][1])

        sign0 = np.sign(sign0)
        sign1 = np.sign(sign1)
        sign2 = np.sign(sign2)
        sign3 = np.sign(sign3)

        filtered = np.logical_and(sign0 * sign1 < 0, sign2 * sign3 < 0)
        return filtered

    def track(self, frame: MatLike, is_visualize=False) -> Tuple[bool, Union[Fretboard, None]]:
        # import cProfile
        # import pstats
        # with cProfile.Profile() as profile:
        image1 = self.preprocess(frame)
        feats1 = self.extractor.extract(image1.to(self.device))

        matches01 = self.matcher({"image0": self.feats0, "image1": feats1})
        feats0_, feats1_, matches01_ = [
            self.rbd(x) for x in [self.feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0_["keypoints"], feats1_["keypoints"], matches01_["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]].cpu().numpy(), kpts1[matches[..., 1]].cpu().numpy()

        filtered = self.filter_match0(m_kpts0)

        M, mask = cv2.findHomography(m_kpts0[filtered, :], m_kpts1[filtered, :], cv2.RANSAC, 5.0)

        now_frets = utils.transform_houghlines(self.fretboard0.frets, np.linalg.inv(M))
        now_strings = utils.transform_houghlines(self.fretboard0.strings, np.linalg.inv(M))

        rect = utils.oriented_bb_from_frets_strings(now_frets, now_strings)
        rect = (rect[0], (rect[1][0] + 50, rect[1][1] + 100), rect[2])
        # results = pstats.Stats(profile)
        # results.sort_stats(pstats.SortKey.TIME)
        # results.dump_stats('results.prof')
        
        now_fretboard = Fretboard(now_frets, now_strings, rect)
        if self.resize_width is not None:
            factor = self.resize_width / frame.shape[1]
            now_fretboard.resize(1/factor)
        return True, now_fretboard