import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from itertools import combinations
from utils.utils import angle_diff, fret_ratio, line_line_intersection_fast, line_line_intersection, point_distance_to_line, ColorThresholder, angle_diff_np, prepare_draw_fretboard
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation
from template import Template, transform_houghlines, shift_center_houghlines, rotate_line_pair, transform_points
from itertools import combinations
from utils.utils import get_screensize, draw_line, cornerpoints_from_line_pair, rotate_image, crop_from_oriented_bounding_box, npprint, find_homography_from_matched_fretlines, ransac_vanishing_point_estimation
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import RANSACRegressor
from skimage.transform import SimilarityTransform
from skimage.measure import ransac
from skimage.morphology import skeletonize
from itertools import product
import time
from sklearn import datasets, svm
from utils.utils_debug import VideoBuffer

# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import munkres

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
    # from sklearn.inspection import DecisionBoundaryDisplay
    # disp = DecisionBoundaryDisplay.from_estimator(
    #     model,
    #     data,
    #     response_method="predict",
    #     cmap=plt.cm.coolwarm,
    #     alpha=0.8,
    #     ax=ax,
    #     xlabel='R',
    #     ylabel='G',
    # )
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
    
    def fingerboard_lines_detection(self, edges, theta_tol=5, rho_tol=100, is_visualize=False):
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

        if is_visualize and fingerboard_lines is not None:
            cdst = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
            for line in fingerboard_lines:
                draw_line(cdst, line[0], line[1])
            return fingerboard_lines, cdst
        return fingerboard_lines, None
    
    def rotate_fingerboard(self, frame, template):
        cornerpoints = cornerpoints_from_line_pair(template.fingerboard_lines, frame.shape[1])

        rect = cv2.minAreaRect(cornerpoints) # ((x-coordinate, y-coordinate),(width, height), rotation)
        if rect[1][1] > rect[1][0]:
            angle = -(90-rect[2])
            padding = 0.2*rect[1][0]
            padded_rect = (rect[0], (rect[1][0]+2*padding, rect[1][1]), rect[2])
        else:
            angle = rect[2]
            padding = 0.2*rect[1][1]
            padded_rect = (rect[0], (rect[1][0], rect[1][1]+2*padding), rect[2])
        rotated, rot_mat = rotate_image(frame, rect[0], angle)
        fingerboard_lines = rotate_line_pair(angle/180*np.pi, rot_mat, template.fingerboard_lines)

        template.register_rotation(rect[0], angle, rot_mat)
        template.register_padded_rect(padded_rect)

        # clip rotated image
        if padded_rect[1][1] > padded_rect[1][0]:
            delta = 0.5*padded_rect[1][0]
        else:
            delta = 0.5*padded_rect[1][1]
        rotated = rotated[max(int(padded_rect[0][1]-delta), 0):int(padded_rect[0][1]+delta), :]
        fingerboard_lines = shift_center_houghlines([0, max(int(padded_rect[0][1]-delta), 0)], fingerboard_lines)

        template.register_shift([0, max(int(padded_rect[0][1]-delta), 0)])

        return rotated, fingerboard_lines
    
    def find_undistort_transform(self, rotated_edges, theta_range = 20/180*np.pi, is_visualize=True):
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

        if is_visualize and angles is not None:
            cdst = cv2.cvtColor(rotated_edges.copy(), cv2.COLOR_GRAY2BGR)
            for angle, dist in zip(angles, dists):
                draw_line(cdst, dist, angle)
            return homography, cdst, (angles, dists)
        return homography, None, (angles, dists)
    
    def locate_fretboard(self, rotated_frame, rotated_edges, fingerboard_lines, template):
        '''
        Locating Fretboard Pipeline:
        1. Detect vertical lines
        2. Merge closely spaced lines
        3. Delete fretline outliers by calculating second order difference
        4. Detect the 3, 5, 7, 9, 12, 15, 17, 19, 21 frets with mean brightness second order difference
        '''
        cdst = cv2.cvtColor(rotated_frame.copy(), cv2.COLOR_GRAY2BGR)

        h, theta, d = hough_line(rotated_edges, theta=np.array([0.0]))
        hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=3, min_angle=1)

        hspace, angles, dists = (list(item) for item in zip(*sorted(zip(hspace, angles, dists), key=lambda x: x[2])))

        hspace = np.array(hspace).astype(float)
        dists = np.array(dists)

        # first stage filter
        # while np.sum(np.diff(hspace) < -10) > 0:
        #     idxes = np.concatenate(([0], np.argwhere(np.diff(hspace) > -10).squeeze() + 1))
        #     hspace = hspace[idxes]
        #     dists = dists[idxes]
        
        # second stage filter (merge closely spaced lines)
        idxes = np.argwhere(np.diff(dists) < 10).squeeze()
        dists[idxes] = (dists[idxes] + dists[idxes+1]) / 2
        hspace[idxes] = (hspace[idxes] + hspace[idxes+1]) / 2
        dists = np.delete(dists, idxes+1)
        hspace = np.delete(hspace, idxes+1)

        # third stage filter (delete outliers)
        fret_dists = np.diff(dists)
        fret_second_order_difference_normalized = np.diff(np.diff(fret_dists)) / fret_dists[1:-1]
        outliers_idx = np.argwhere(fret_second_order_difference_normalized[5:] > 2) + 5 + 1
        if outliers_idx.size > 0:
            outliers_idx = list(outliers_idx.reshape((-1,)))
            for i in range(len(outliers_idx)):
                outlier = outliers_idx[i]
                if fret_ratio(dists[outlier-1], dists[outlier], dists[outlier+2]) < \
                fret_ratio(dists[outlier-1], dists[outlier+1], dists[outlier+2]):
                    outliers_idx[i] = outliers_idx[i]+1
                    # draw_line(cdst, dists[outlier+1], 0, color=(0, 0, 255), thickness=3)
                else:
                    # draw_line(cdst, dists[outlier], 0, color=(0, 0, 255), thickness=3)
                    pass
                # cv2.rectangle(cdst, (int(dists[outlier]), 0), (int(dists[outlier+1]), 20), (0, 255, 0), -1) 
            dists = np.delete(dists, outliers_idx)
            hspace = np.delete(hspace, outliers_idx)

        for dist in dists:
            draw_line(cdst, dist, 0)

        cv2.namedWindow("8: Fret Lines Detection and Filtering")
        def mouse_callback(event, x, y, flags, param):
            idx = np.argmin(np.abs(np.array(dists) - x))
            cdst2 = cdst.copy()
            if abs(dists[idx] - x ) < 5:
                draw_line(cdst2, dists[idx], 0, thickness=3)
                cv2.putText(cdst2, f'{hspace[idx]}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            cv2.imshow('8: Fret Lines Detection and Filtering', cdst2)
        cv2.setMouseCallback("8: Fret Lines Detection and Filtering", mouse_callback)

        rois = []
        features = np.zeros((dists.shape[0]-1, 1))
        for i, (ri, rj) in enumerate(zip(dists[0:-1], dists[1:])):
            point1 = line_line_intersection_fast(fingerboard_lines[0, :2], rj)
            point2 = line_line_intersection_fast(fingerboard_lines[1, :2], rj)

            if point1[1] > point2[1]:
                point1, point2 = point2, point1

            rect = [int(ri), int(point1[1]), int(rj)-int(ri), int(point2[1])-int(point1[1])] # [x y w h]
            rois.append(rect)
            # cv2.imshow('debug', rotated_frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
            # histr = cv2.calcHist([rotated_frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]], [0], None, [256], [0,255])
            # features[i, :] = histr.reshape((-1,))
            features[i, 0] = np.mean([rotated_frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]])

            # cv2.waitKey()
            # plt.figure()
            # plt.plot(histr)
            # plt.show()
        
        second_order_difference = np.diff(np.diff(features.reshape((-1,))))
        for i in range(len(second_order_difference)):
            cv2.putText(cdst, f'{second_order_difference[i]:.0f}', (int(dists[i+1]), 40+i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))

        kernel_12 = np.array([-1, -1, 1, -1, -1])
        idx_12 = np.argmin(np.convolve(second_order_difference, kernel_12, 'valid')) + 2 + 1

        kernel_quadruplets = np.array([-1, 2, -1, 2, -1, 2, -1, 2, -1])
        if len(second_order_difference) < 2 * len(kernel_quadruplets):
            idx_quadruplets = np.array([idx_12, idx_12])
        else:
            convolve = np.convolve(second_order_difference, kernel_quadruplets, 'valid')
            for i in range(len(convolve)):
                cv2.putText(cdst, f'{convolve[i]:.0f}', (int(dists[i+5]), 100), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255))

            smallest_idx = np.argmin(convolve)
            mask_idxes = [max(smallest_idx-4, 0), min(smallest_idx+4, len(convolve))]
            convolve[mask_idxes[0]:mask_idxes[1]] = np.max(convolve)
            second_smallest_idx = np.argmin(convolve)

            idx_quadruplets = np.array([smallest_idx, second_smallest_idx]) + 4 + 1

        # vailidation
        is_located = False
        if np.abs(idx_quadruplets[0] - idx_12) == 6 and np.abs(idx_quadruplets[1] - idx_12) == 6:
            is_located = True
            idx_high = np.max(idx_quadruplets)
            idx_low = np.min(idx_quadruplets)
            # fret_idx = np.array([
            #     idx_high+3, idx_high+1, idx_high-1, idx_high-3,
            #     idx_12,
            #     idx_low+3, idx_low+1, idx_low-1, idx_low-3
            # ])
            fret_idx = np.array([
                idx_high+4, idx_high+3, idx_high+2, idx_high+1, idx_high, idx_high-1, idx_high-2, idx_high-3,
                idx_12+1, idx_12,
                idx_low+4, idx_low+3, idx_low+2, idx_low+1, idx_low, idx_low-1, idx_low-2, idx_low-3
            ])

            for idx in np.array([
                idx_high+3, idx_high+1, idx_high-1, idx_high-3,
                idx_12,
                idx_low+3, idx_low+1, idx_low-1, idx_low-3
            ]):
                cv2.rectangle(cdst, (int(dists[idx]), 0), (int(dists[idx+1]), 20), (0, 0, 255), -1)
        else:
            fret_idx = None
            cv2.rectangle(cdst, (int(dists[idx_12]), 0), (int(dists[idx_12])+20, 20), (255, 255, 255), -1)
            if idx_quadruplets[0] <=len(dists) and idx_quadruplets[1] <=len(dists):
                cv2.rectangle(cdst, (int(dists[idx_quadruplets[0]]), 0), (int(dists[idx_quadruplets[0]])+20, 20), (255, 255, 255), -1)
                cv2.rectangle(cdst, (int(dists[idx_quadruplets[1]]), 0), (int(dists[idx_quadruplets[1]])+20, 20), (255, 255, 255), -1)

        cv2.imshow('9: Marker Detection', cdst)

        if is_located:
            template.register_final_shift([int(dists[fret_idx[-1]-1]), 0])
            template.register_fretlines(dists[fret_idx] - int(dists[fret_idx[-1]-1]))
            template.register_template_image(rotated_frame[:, int(dists[fret_idx[-1]-1]):])
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
        cv2.imshow('1: Original Image', frame)
        gray = self.preprocess(frame)
        cv2.imshow('2: Preprocess', gray)
        self.template.register_image(gray)

        edges = cv2.Canny(gray, 30, 100, 5)

        fingerboard_lines, fingerboard_cdst = self.fingerboard_lines_detection(edges, theta_tol=5, rho_tol=100, is_visualize=True)
        cv2.imshow('3: Prominent Line Detection', fingerboard_cdst)
        self.template.register_fingerboard_lines(fingerboard_lines)

        if fingerboard_lines is None:
            return

        rotated_frame, fingerboard_lines = self.rotate_fingerboard(gray, self.template)
        cv2.imshow('4: Cropped Rotated ROI', rotated_frame)


        rotated_frame, fingerboard_lines = self.rotate_fingerboard(gray, self.template)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        rotated_frame = clahe.apply(rotated_frame)

        thresh = cv2.adaptiveThreshold(rotated_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 15, 0)
        edges = skeletonize(thresh).astype("uint8") * 255

        cornerpoints = cornerpoints_from_line_pair(fingerboard_lines, rotated_frame.shape[1])
        mask = np.zeros_like(rotated_frame)
        cv2.fillPoly(mask, pts=[cornerpoints], color=(255, 255, 255))
        rotated_frame = cv2.bitwise_and(rotated_frame, mask)
        cv2.imshow('5: Masked Image', rotated_frame)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # rotated_frame = clahe.apply(rotated_frame)
        # rotated_frame = cv2.bitwise_and(rotated_frame, mask)

        rotated_edges = cv2.Canny(rotated_frame, 30, 150, 5)

        homography, find_undistort_cdst, fret_lines = self.find_undistort_transform(rotated_edges, is_visualize=True)
        self.template.register_homography(homography)
        cv2.imshow('6: Find Fret Lines', find_undistort_cdst)

        rotated_frame = cv2.warpPerspective(rotated_frame, homography, rotated_frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        # draw_line(rotated_frame, filtered_parallel_lines[0][1], filtered_parallel_lines[0][2])
        # draw_line(rotated_frame, filtered_parallel_lines[0][3], filtered_parallel_lines[0][4])
        cv2.imshow('7: Projectively Rectified Image', rotated_frame)

        rotated_edges = cv2.warpPerspective(rotated_edges, homography, rotated_edges.shape[1::-1], flags=cv2.INTER_LINEAR)

        flag = self.locate_fretboard(rotated_frame, rotated_edges, fingerboard_lines, self.template)

        if flag:
            self.is_initialized = True

            cdst = frame.copy()
            self.template.draw_fretboard(cdst, np.linalg.inv(self.template.total_mat))
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
            rect = (rect[0], (rect[1][0] + 30, rect[1][1] + 30), rect[2])

            self.oriented_bounding_box = rect
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(cdst, [box], 0, (0,0,255), 2)

            thresh = cv2.adaptiveThreshold(self.template.template_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 51, -10)
            # cv2.imshow('thresh', thresh)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1, 5))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            # cv2.imshow('closed', closed)

            analysis = cv2.connectedComponentsWithStats(closed, 4, cv2.CV_32S) 
            (totalLabels, label_ids, values, centroid) = analysis

            valid_idxes = []
            output = np.zeros(closed.shape, dtype="uint8")
            for i in range(0, totalLabels): 
                area = values[i, cv2.CC_STAT_AREA]   
            
                # adaptive area thresholding
                if (area > 2000) or (area < 50):
                    continue

                # if point_line_distance(fingerboard_lines[0, :], centroid[i]) > 0 and \
                #     point_line_distance(fingerboard_lines[1, :], centroid[i]) < 0:
                componentMask = (label_ids == i).astype("uint8") * 255
                contours, _ = cv2.findContours(componentMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # aspect ratio filtering
                form_factor = area / contours[0].size
                if form_factor < 2:
                    continue

                # convexity filtering
                solidity = area / cv2.contourArea(cv2.convexHull(contours[0]))
                if solidity < 0.7:
                    continue

                valid_idxes.append(i)
                output = cv2.bitwise_or(output, componentMask)
            # cv2.imshow('output', output)

            fret_pos = (self.template.template_fretlines[0::2, 0] + self.template.template_fretlines[1::2, 0]) / 2

            _, now_ind = linear_sum_assignment(np.abs(fret_pos.reshape((-1, 1)) - centroid[valid_idxes, 0]))

            now_marker_positions = np.zeros((9, 2))
            now_marker_area = np.zeros((9,))
            for i in range(9):
                now_marker_positions[i] = centroid[valid_idxes[now_ind[i]]]
                now_marker_area[i] = values[valid_idxes[now_ind[i]], cv2.CC_STAT_AREA]
            self.marker_areas = now_marker_area
            
            self.tracked_marker_positions = transform_points(now_marker_positions, np.linalg.inv(self.template.total_mat))
            self.template.register_template_marker_positions(self.tracked_marker_positions)

            self.tracked_fretlines = self.template.image_fretlines

            scale = np.linalg.norm(self.tracked_marker_positions[0, :] - self.tracked_marker_positions[-1, :])
            self.fretline_to_marker_dist = (self.template.template_fretlines[:, 0] - np.repeat(now_marker_positions[:, 0], 2))

            for point in self.tracked_marker_positions:
                cv2.circle(cdst, (int(point[0]), int(point[1])), 3, (0, 255, 0))

            cv2.imshow('Initialization Complete', cdst)

            # rotated_cropped_frame, crop_transform = crop_from_oriented_bounding_box(frame, self.rect)

            # cornerpoints = cornerpoints_from_line_pair(transform_houghlines(self.template.fingerboard_lines, np.linalg.inv(crop_transform)), rotated_cropped_frame.shape[1])
            # mask = np.zeros_like(rotated_cropped_frame)
            # cv2.fillPoly(mask, pts=[cornerpoints], color=(255, 255, 255))
            # rotated_cropped_frame = cv2.bitwise_and(rotated_cropped_frame, mask)
            # cv2.imshow('mask_rotated_cropped_frame', rotated_cropped_frame)

            # kmeans_cdst, label_img, centers = kmeans_color_quantization(rotated_cropped_frame, clusters=2)

            # if np.sum(label_img == 0) > np.sum(label_img == 1):
            #     self.fretcolor = centers[0]
            # else:
            #     self.fretcolor = centers[1]

            # cv2.imshow('kmeans', kmeans_cdst)

    def track_markers(self, frame):
        '''
        Image processing pipeline:
        1. Preprocess: convert to gray -> denoising (median filter) -> enhance contrast (clahe)

        '''
        rotated_cropped_frame, crop_transform = crop_from_oriented_bounding_box(frame, self.oriented_bounding_box)
        cv2.imshow('1: Cropped Rotated ROI', rotated_cropped_frame)

        self.colorthresholder.update(rotated_cropped_frame)
        rotated_cropped_frame = self.colorthresholder.output

        # kmeans_cdst, label_img, centers = kmeans_color_quantization(rotated_cropped_frame, clusters=3)

        # label = np.argmin(np.linalg.norm(centers.astype(np.float32) - self.fretcolor, axis=1))

        # cv2.imshow('kmeans', (label_img==label).astype(np.uint8) * 255)

        gray = self.preprocess(rotated_cropped_frame)
        cv2.imshow('2: Preprocess', rotated_cropped_frame)

        # while True:
        #     self.colorthresholder.update(rotated_cropped_frame)
        #     cv2.waitKey(1)

        # cv2.imshow('gray', gray)

        # mser = cv2.MSER_create(max_area=3000)
        # regions, regionbbs = mser.detectRegions(rotated_cropped_frame)

        # cdst = rotated_cropped_frame.copy()
        # # cdst = cv2.cvtColor(rotated_cropped_frame.copy(), cv2.COLOR_GRAY2BGR)
        # for bb_ in regionbbs:
        #     cv2.rectangle(cdst, (bb_[0],bb_[1]), (bb_[0]+bb_[2],bb_[1]+bb_[3]), (0, 255, 0), 1)

        # cv2.imshow('cdst', cdst)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 51, -10)
        # cv2.imshow('thresh', thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # closed = cv2.morphologyEx(opening, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 1)))
        closed = cv2.morphologyEx(opening, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)))
        cv2.imshow('3: Thresholding and Morphological Filtering', closed)

        analysis = cv2.connectedComponentsWithStats(closed, 4, cv2.CV_32S) 
        (totalLabels, label_ids, values, centroid) = analysis

        valid_idxes = []
        output = np.zeros(closed.shape, dtype="uint8")
        for i in range(0, totalLabels): 
            area = values[i, cv2.CC_STAT_AREA]   
        
            # adaptive area thresholding
            if (area > 1.3 * np.max(self.marker_areas)) or (area < 0.7 * np.min(self.marker_areas)):
                continue

            componentMask = (label_ids == i).astype("uint8") * 255
            contours, _ = cv2.findContours(componentMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # aspect ratio filtering
            form_factor = area / contours[0].size
            if form_factor < 2:
                continue

            # convexity filtering
            solidity = area / cv2.contourArea(cv2.convexHull(contours[0]))
            if solidity < 0.7:
                continue

            valid_idxes.append(i)
            output = cv2.bitwise_or(output, componentMask)
        cv2.imshow('4: Filtering by Marker Shape', output)

        # tracking
        prev_marker_positions = transform_points(self.tracked_marker_positions, crop_transform)

        cdst = cv2.cvtColor(output.copy(), cv2.COLOR_GRAY2BGR)
        for point in prev_marker_positions:
            cv2.circle(cdst, (int(point[0]), int(point[1])), 3, (0, 0, 255))
        for point in centroid[valid_idxes]:
            cv2.circle(cdst, (int(point[0]), int(point[1])), 3, (255, 0, 0))
        
        # establish correspondence
        # from munkres import Munkres
        # m = Munkres()
        cost_matrix = cdist(prev_marker_positions, centroid[valid_idxes])
        outlier_threshold = np.median(np.min(cost_matrix, axis=0))*5
        cost_matrix = np.concatenate((cost_matrix, np.ones((9, 9)) * outlier_threshold), axis=1)

        #     index = m.compute(cost_matrix)
        #     prev_ind = np.array([ind[0] for ind in index])
        #     now_ind = np.array([ind[1] for ind in index])

        prev_ind, now_ind = linear_sum_assignment(cost_matrix)
        prev_ind = prev_ind[now_ind <= len(valid_idxes) - 1]
        now_ind = now_ind[now_ind <= len(valid_idxes) - 1]

        not_detected_prev_marker_ind = np.setdiff1d(np.arange(prev_marker_positions.shape[0]), prev_ind)

        now_detected_marker_positions = np.zeros((now_ind.shape[0], 2))

        for i in range(now_ind.shape[0]):
            now_detected_marker_positions[i] = centroid[valid_idxes[now_ind[i]]]
            prev_point = (int(prev_marker_positions[prev_ind[i]][0]), int(prev_marker_positions[prev_ind[i]][1]))
            now_point = (int(centroid[valid_idxes[now_ind[i]]][0]), int(centroid[valid_idxes[now_ind[i]]][1]))
            cv2.arrowedLine(cdst, prev_point, now_point, (255, 0, 0), 1)
        cv2.imshow('5: Marker Matching', cdst)

        # filter outliers
        model_robust, inliers = ransac(
            (prev_marker_positions[prev_ind, :], now_detected_marker_positions), SimilarityTransform, min_samples=2, residual_threshold=10, max_trials=1000
        )
        outliers = inliers == False
        self.tmp_model = model_robust

        # tracked_marker_positions update
        crop_tracked_marker_positions = np.zeros((9, 2))
        crop_tracked_marker_positions[not_detected_prev_marker_ind, :] = transform_points(prev_marker_positions[not_detected_prev_marker_ind, :], model_robust)
        crop_tracked_marker_positions[prev_ind[inliers], :] = now_detected_marker_positions[inliers, :]
        if np.sum(outliers) > 0:
            # crop_tracked_marker_positions[prev_ind[outliers], :] = now_detected_marker_positions[outliers, :]
            crop_tracked_marker_positions[prev_ind[outliers], :] = transform_points(prev_marker_positions[prev_ind[outliers], :], model_robust)

        self.tracked_marker_positions = transform_points(crop_tracked_marker_positions, np.linalg.inv(crop_transform))

        # oriented bounding box update
        center = transform_points(np.array(self.oriented_bounding_box[0]).reshape((1, 2)), np.linalg.inv(crop_transform) @ model_robust.params @ crop_transform)
        delta_theta = np.arctan2(model_robust.params[0, 1], model_robust.params[0, 0])
        scale = np.linalg.norm(self.tracked_marker_positions[0, :] - self.tracked_marker_positions[-1, :])
        scaling = model_robust.params[0, 0] / np.cos(delta_theta)
        orientation = self.oriented_bounding_box[2] - 180 / np.pi * delta_theta
        prev_oriented_bounding_box = self.oriented_bounding_box
        if self.oriented_bounding_box[1][1] > self.oriented_bounding_box[1][0]:
            self.oriented_bounding_box = ((center[0, 0], center[0, 1]), (1.2 * scale / self.oriented_bounding_box[1][1] * self.oriented_bounding_box[1][0] , 1.2 * scale), orientation)
        else:
            self.oriented_bounding_box = ((center[0, 0], center[0, 1]), (1.2 * scale, 1.2 * scale / self.oriented_bounding_box[1][0] * self.oriented_bounding_box[1][1]), orientation)
        # self.oriented_bounding_box = ((center[0, 0], center[0, 1]), (scaling*self.oriented_bounding_box[1][0], scaling*self.oriented_bounding_box[1][1]), orientation)

        # cornerpoints = transform_points(np.intp(cv2.boxPoints(self.oriented_bounding_box)), crop_transform)
        # self.oriented_bounding_box = cv2.minAreaRect(transform_points(transform_points(cornerpoints, model_robust.params), np.linalg.inv(crop_transform)).astype(int))

        # marker area update
        self.marker_areas[not_detected_prev_marker_ind] = self.marker_areas[not_detected_prev_marker_ind] * scaling
        self.marker_areas[prev_ind[inliers]] = values[np.array(valid_idxes)[now_ind[inliers]], cv2.CC_STAT_AREA]
        if np.sum(outliers) > 0:
            self.marker_areas[prev_ind[outliers]] = self.marker_areas[prev_ind[outliers]] * scaling

        cdst = frame.copy()

        for point in self.tracked_marker_positions:
            cv2.circle(cdst, (int(point[0]), int(point[1])), 3, (0, 255, 0))
        
        # box = cv2.boxPoints(prev_oriented_bounding_box)
        # box = np.intp(box)
        # cv2.drawContours(cdst, [box], 0, (0,0,255), 1)

        box = cv2.boxPoints(self.oriented_bounding_box)
        box = np.intp(box)
        cv2.drawContours(cdst, [box], 0, (0, 255, 0), 2)

        cv2.putText(cdst, f'{self.counter}', (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        return cdst
        
        # two-step validation
        # prev_feature = (prev_marker_positions[:, 0] - prev_marker_positions[-1, 0]) / (prev_marker_positions[0, 0] - prev_marker_positions[-1, 0])

        # sol = np.linalg.lstsq(now_marker_positions, -np.ones((9,1)))
        # rho = 1/np.linalg.norm(sol[0])
        # theta = np.arctan2(-sol[0][1], -sol[0][0])
        # draw_line(cdst, rho, theta)

        # # collinearlity
        # if sol[1][0] < 0.1:
        #     # feature vector comparison
        #     now_feature = np.array([project_point_to_line([sol[0][0][0], sol[0][1][0]], point) for point in now_marker_positions])
        #     now_feature = (now_feature - now_feature[-1]) / (now_feature[0] - now_feature[-1])
        #     now_feature - prev_feature
        # else:
        #     pass

    
    def refine_fretboard(self, frame, final_cdst):
        rotated_cropped_frame, crop_transform = crop_from_oriented_bounding_box(frame, self.oriented_bounding_box)

        self.colorthresholder.update(rotated_cropped_frame)
        rotated_cropped_frame = self.colorthresholder.output

        gray = self.preprocess(rotated_cropped_frame)

        # edges = cv2.Canny(gray, 30, 100, 5)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 21, 0)
        edges = skeletonize(thresh).astype("uint8") * 255
        cv2.imshow('1: Preprocess', edges)

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, None, 30, 20)

        # filter by angle (> 45 deg)
        linesP = linesP[np.abs(linesP[:, 0, 0] - linesP[:, 0, 2]) <= np.abs(linesP[:, 0, 1] - linesP[:, 0, 3]), ...]

        # start = time.time()
        # vanishing_point = ransac_vanishing_point_estimation(linesP)
        # print(time.time() - start)

        # point1 = vanishing_point[:2] / vanishing_point[2]
        # point2 = np.array([rotated_cropped_frame.shape[1], rotated_cropped_frame.shape[0]])
        # parameter = np.linalg.inv(np.vstack((point1, point2))) @ np.array([-1, 0]).reshape((2, 1))
        # rectification = np.array([[1, 0, 0], [0, 1, 0], [parameter[0][0], parameter[1][0], 1]]).astype('float64')
        # cv2.imshow('rotated_cropped_frame', rotated_cropped_frame)
        # rotated_cropped_frame = cv2.warpPerspective(rotated_cropped_frame, rectification, rotated_cropped_frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        # cv2.imshow('rotated_cropped_frame_rect', rotated_cropped_frame)

        # if linesP is not None:
        #     for i in range(0, len(linesP)):
        #         l = linesP[i][0]
        #         cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)

        # theta_range = 20/180*np.pi
        # tested_angles = np.linspace(-theta_range, theta_range, 20, endpoint=False)
        # h, theta, d = hough_line(edges, theta=tested_angles)
        # hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=5, min_angle=2, threshold=20)

        cdst = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)

        # houghlinesp to hough line coordinates
        angles = np.arctan2(linesP[:, 0, 2] - linesP[:, 0, 0], linesP[:, 0, 1] - linesP[:, 0, 3])
        dists = linesP[:, 0, 2] * np.cos(angles) + linesP[:, 0, 3] * np.sin(angles)
        angles[dists < 0] = angles[dists < 0] - np.sign(angles[dists < 0]) * np.pi
        dists[dists < 0] = -dists[dists < 0]

        # lines = np.concatenate((dists[:, np.newaxis], angles[:, np.newaxis]), axis=1)

        # rect_lines = transform_houghlines(lines, np.linalg.inv(rectification))
        # for line in rect_lines:
            # draw_line(cdst, line[0], line[1])

        crop_marker_positions = transform_points(self.tracked_marker_positions, crop_transform)
        for point in crop_marker_positions:
            cv2.circle(cdst, (int(point[0]), int(point[1])), 3, (0, 255, 0))
        
        # further filter those that intersect marker points
        valid_index = []
        for i, (dist, angle) in enumerate(zip(dists, angles)):
            if np.min([abs(point_distance_to_line(marker, dist, angle)) for marker in crop_marker_positions]) >=3:
                valid_index.append(i)
        angles = angles[valid_index]
        dists = dists[valid_index]

        angles, dists = (list(item) for item in zip(*sorted(zip(angles, dists), key=lambda x: x[1], reverse=True)))
        angles = np.array(angles)
        dists = np.array(dists)
        for dist, angle in zip(dists, angles):
            draw_line(cdst, dist, angle)
        
        # match lines
        now_crop_tracked_fretlines = []
        prev_crop_tracked_fretlines = transform_houghlines(self.tracked_fretlines, np.linalg.inv(crop_transform))

        cv2.imshow('2: Houghlines Detection and Filtering', cdst)

        for i in range(9):
            marker = crop_marker_positions[i, :]

            prev_line_right = prev_crop_tracked_fretlines[2*i, :]
            prev_signed_dist_to_marker_right = self.fretline_to_marker_dist[2*i]
            prev_line_left = prev_crop_tracked_fretlines[2*i+1, :]
            prev_signed_dist_to_marker_left = self.fretline_to_marker_dist[2*i+1]

            signed_distances = np.array([point_distance_to_line(marker, dist, angle) for dist, angle in zip(dists, angles)])

            # find right candidates
            right_candidates = np.bitwise_and(np.abs(signed_distances - prev_signed_dist_to_marker_right) < 20, np.sign(signed_distances) == np.sign(prev_signed_dist_to_marker_right))
            right_candidates = np.argwhere(right_candidates)
            if right_candidates.shape[0] != 0:
                diff = angle_diff_np(angles[right_candidates[:, 0]], prev_line_right[1])
                right_candidates = right_candidates[diff <= 5/180 * np.pi]

            # find left candidates
            left_candidates = np.bitwise_and(np.abs(signed_distances - prev_signed_dist_to_marker_left) < 20, np.sign(signed_distances) == np.sign(prev_signed_dist_to_marker_left))
            left_candidates = np.argwhere(left_candidates)
            if left_candidates.shape[0] != 0:
                diff = angle_diff_np(angles[left_candidates[:, 0]], prev_line_left[1])
                left_candidates = left_candidates[diff <= 5/180 * np.pi]

            if right_candidates.shape[0] == 0 and left_candidates.shape[0] == 0:
                now_crop_tracked_fretlines.append([0, 0])
                now_crop_tracked_fretlines.append([0, 0])
            elif right_candidates.shape[0] == 0:
                now_crop_tracked_fretlines.append([0, 0])
                index_ = np.argmin(np.abs(signed_distances[left_candidates] - prev_signed_dist_to_marker_left))
                if np.abs(signed_distances[left_candidates[index_]] - prev_signed_dist_to_marker_left) / np.abs(prev_signed_dist_to_marker_left) < 0.15:
                    now_crop_tracked_fretlines.append([dists[left_candidates[index_, 0]], angles[left_candidates[index_, 0]]])
                else:
                    now_crop_tracked_fretlines.append([0, 0])
            elif left_candidates.shape[0] == 0:
                index_ = np.argmin(np.abs(signed_distances[right_candidates] - prev_signed_dist_to_marker_right))
                if np.abs(signed_distances[right_candidates[index_]] - prev_signed_dist_to_marker_right) / np.abs(prev_signed_dist_to_marker_right) < 0.15:
                    now_crop_tracked_fretlines.append([dists[right_candidates[index_, 0]], angles[right_candidates[index_, 0]]])
                else:
                    now_crop_tracked_fretlines.append([0, 0])
                now_crop_tracked_fretlines.append([0, 0])
            else:
                # consider only parrallel lines
                combinations = np.array(list(product(right_candidates[:, 0], left_candidates[:, 0])))
                diff = np.abs(angles[combinations[:, 0]] - angles[combinations[:, 1]])
                combinations = combinations[diff <= max(3 * np.min(diff), 5/180*np.pi)]

                index_ = np.argmin(np.abs(signed_distances[combinations[:, 0]] - signed_distances[combinations[:, 1]] - (prev_signed_dist_to_marker_right - prev_signed_dist_to_marker_left)))
                if abs((signed_distances[combinations[index_, 0]] - signed_distances[combinations[index_, 1]] - (prev_signed_dist_to_marker_right - prev_signed_dist_to_marker_left))) / (prev_signed_dist_to_marker_right - prev_signed_dist_to_marker_left) > 0.16:
                    lr_index_ = np.argmax([abs(signed_distances[combinations[index_, 0]] - prev_signed_dist_to_marker_right), abs(signed_distances[combinations[index_, 1]] - prev_signed_dist_to_marker_left)])
                    right_index_, left_index_ = combinations[index_, :]
                    if lr_index_ == 0:
                        now_crop_tracked_fretlines.append([0, 0])
                        now_crop_tracked_fretlines.append([dists[left_index_], angles[left_index_]])
                    else:
                        now_crop_tracked_fretlines.append([dists[right_index_], angles[right_index_]])
                        now_crop_tracked_fretlines.append([0, 0])
                else:
                    right_index_, left_index_ = combinations[index_, :]
                    now_crop_tracked_fretlines.append([dists[right_index_], angles[right_index_]])
                    now_crop_tracked_fretlines.append([dists[left_index_], angles[left_index_]])
        now_crop_tracked_fretlines = np.array(now_crop_tracked_fretlines)

        # remove duplicate matching
        u, c = np.unique(now_crop_tracked_fretlines[:, 0], return_counts=True)
        dup = u[c > 1]
        if dup[dup!=0].shape[0] > 0:
            for dup_dist in dup[dup!=0]:
                dup_index = np.argwhere(now_crop_tracked_fretlines[:, 0] == dup_dist).reshape((-1,))

                signed_distances_dup = [point_distance_to_line(marker, now_crop_tracked_fretlines[dup_index[0], 0], now_crop_tracked_fretlines[dup_index[0], 1]) for marker in crop_marker_positions[dup_index // 2, :]]

                index_ = np.argmin(np.abs(self.fretline_to_marker_dist[dup_index] - signed_distances_dup))

                now_crop_tracked_fretlines[dup_index[np.arange(dup_index.shape[0]) != index_], :] = 0
        
        matched_index = np.argwhere(now_crop_tracked_fretlines[:, 0] != 0).reshape((-1,))

        for line in now_crop_tracked_fretlines:
            if line is None:
                continue
            draw_line(cdst, line[0], line[1], color=(255, 0, 0))
        cv2.imshow('3: Fret Lines Matching', cdst)

        transform_mat, inliers = find_homography_from_matched_fretlines(prev_crop_tracked_fretlines[matched_index, :], now_crop_tracked_fretlines[matched_index, :])

        # update tracked lines
        not_matched_index = np.argwhere(now_crop_tracked_fretlines[:, 0] == 0).reshape((-1,))
        self.tracked_fretlines[not_matched_index, :] = transform_houghlines(prev_crop_tracked_fretlines[not_matched_index, :], np.linalg.inv(self.tmp_model) @ crop_transform)
        # self.tracked_fretlines[not_matched_index, :] = transform_houghlines(prev_crop_tracked_fretlines[not_matched_index, :], transform_mat @ crop_transform)
        # self.tracked_fretlines[matched_index[inliers], :] = transform_houghlines(prev_crop_tracked_fretlines[matched_index[inliers], :], transform_mat @ crop_transform)
        self.tracked_fretlines[matched_index[inliers], :] = transform_houghlines(now_crop_tracked_fretlines[matched_index[inliers], :], crop_transform)
        outliers = inliers == False

        if np.sum(outliers) > 0:
            self.tracked_fretlines[matched_index[outliers], :] = transform_houghlines(now_crop_tracked_fretlines[matched_index[outliers], :], crop_transform)
            # self.tracked_fretlines[matched_index[outliers], :] = transform_houghlines(prev_crop_tracked_fretlines[matched_index[outliers], :], np.linalg.inv(self.tmp_model) @ crop_transform)
        
        tmp = self.tracked_fretlines[:, 0] < 0
        self.tracked_fretlines[tmp, 1] = self.tracked_fretlines[tmp, 1] - np.sign(self.tracked_fretlines[tmp, 1]) * np.pi
        self.tracked_fretlines[tmp, 0] = -self.tracked_fretlines[tmp, 0]

        pts1, pts2 = prepare_draw_fretboard(self.oriented_bounding_box[1][0], self.tracked_fretlines, crop_transform)

        # update feature
        delta_theta = np.arctan2(self.tmp_model.params[0, 1], self.tmp_model.params[0, 0])
        scaling = self.tmp_model.params[0, 0] / np.cos(delta_theta)
        self.fretline_to_marker_dist[not_matched_index] = self.fretline_to_marker_dist[not_matched_index] * scaling
        for i, (marker, line) in enumerate(zip(crop_marker_positions[matched_index[inliers] // 2], now_crop_tracked_fretlines[matched_index[inliers], :])):
            self.fretline_to_marker_dist[matched_index[inliers][i]] = point_distance_to_line(marker, line[0], line[1])
        for i, (marker, line) in enumerate(zip(crop_marker_positions[matched_index[outliers] // 2], now_crop_tracked_fretlines[matched_index[outliers], :])):
            self.fretline_to_marker_dist[matched_index[outliers][i]] = point_distance_to_line(marker, line[0], line[1])

        # for line in self.tracked_fretlines:
        #     draw_line(final_cdst, line[0], line[1], color=(0, 0, 255))
        for pt1, pt2 in zip(pts1, pts2):
            cv2.line(final_cdst, pt1, pt2, color=(0, 255, 0), thickness=2)
        for line in transform_houghlines(self.tracked_fretlines, np.linalg.inv(crop_transform)):
            draw_line(cdst, line[0], line[1], color=(0, 0, 255))
        for line in transform_houghlines(self.tracked_fretlines[matched_index[inliers], :], np.linalg.inv(crop_transform)):
            draw_line(cdst, line[0], line[1], color=(0, 0, 255), thickness=2)
        for line in transform_houghlines(self.tracked_fretlines[not_matched_index, :], np.linalg.inv(crop_transform)):
            draw_line(cdst, line[0], line[1], color=(0, 255, 255), thickness=2)
        if not_matched_index.shape[0] > 0:
            for pt1, pt2 in zip(pts1[not_matched_index, :], pts2[not_matched_index, :]):
                cv2.line(final_cdst, pt1, pt2, color=(0, 255, 255), thickness=2)
            
        cv2.imshow('4: Red (Tracked) and Yellow (Not-Tracked) Fret Lines', cdst)
        cv2.imshow('Final Tracking Result', final_cdst)
        cv2.waitKey()
        return final_cdst
        # self.video.write(final_cdst)
    
    def apply(self, frame):
        if not self.is_initialized:
            self.init_detect(frame)
            cv2.waitKey()
            cv2.destroyAllWindows()
            self.colorthresholder = ColorThresholder(self.screensize)
            # self.video = cv2.VideoWriter('video.mp4',  
            #         cv2.VideoWriter_fourcc(*'H264'), p
            #         25, (frame.shape[1], frame.shape[0]))
        else:
            cdst = self.track_markers(frame)
            cv2.destroyAllWindows()
            self.colorthresholder = ColorThresholder(self.screensize)
            final_cdst = self.refine_fretboard(frame, cdst)
            self.videoplayback.put(final_cdst)
            self.videoplayback.imshow()
        self.counter = self.counter + 1

if __name__ == '__main__':
    cap = cv2.VideoCapture('test\\guitar2.mp4')

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

    # model = svm.LinearSVC(C=1, max_iter=10000)

    # data = np.concatenate((data1, data2), axis=0)
    # label = np.concatenate((np.ones((data1.shape[0], 1)), -np.ones((data2.shape[0], 1))), axis=0)

    # model.fit(data, label)
    # show_multiple_images_2d(model, data1, data2)
    # plt.show()

    fd = FretboardDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # ret, frame = cap.read()
        fd.apply(frame)
        cv2.waitKey(1)
    # fd.video.release()
