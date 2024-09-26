import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from itertools import combinations
from utils.utils import angle_diff
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation
from template import Template, transform_houghlines, shift_center_houghlines, rotate_line_pair, transform_points
from itertools import combinations
from sklearn import linear_model
from scipy.optimize import curve_fit
from scipy.optimize import least_squares, minimize

def point_line_distance(linert, pnt):
    return pnt[0] * np.cos(linert[1]) + pnt[1] * np.sin(linert[1]) - linert[0]

def quad_area(quad):
    x = quad[:, 0]
    y = quad[:, 1]
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)

def polygon_area(x, y):
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)

def line_line_intersection(riti, rjtj):
    return np.linalg.inv(np.array([
        [np.cos(riti[1]), np.sin(riti[1])],
        [np.cos(rjtj[1]), np.sin(rjtj[1])]
        ])) @ np.array([riti[0], rjtj[0]])

def line_line_intersection_fast(riti, rj):
    return np.linalg.inv(np.array([
        [np.cos(riti[1]), np.sin(riti[1])],
        [1, 0]
        ])) @ np.array([riti[0], rj])

def rotate_image(image, image_center, angle):
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  # deg
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result, rot_mat

def draw_line(image, rho, theta, **kwargs):
    length = max(image.shape[0], image.shape[1])

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + length*(-b)), int(y0 + length*(a)))
    pt2 = (int(x0 - length*(-b)), int(y0 - length*(a)))
    if 'thickness' in kwargs:
        thickness = kwargs['thickness']
    else:
        thickness = 1
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = (0, 255, 0)
    cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)

def cornerpoints_from_line_pair(line_pair, width):
    '''
    line_pair = [[r1, t1, h1], [r2, t2, h2]]
    '''
    w = width
    r1 = line_pair[0, 0]
    t1 = line_pair[0, 1]
    r2 = line_pair[1, 0]
    t2 = line_pair[1, 1]
    cornerpoints = np.array([[
        [0, int(r1/np.sin(t1))],
        [w, int((r1-w*np.cos(t1))/np.sin(t1))],
        [w, int((r2-w*np.cos(t2))/np.sin(t2))],
        [0, int(r2/np.sin(t2))]
    ]])
    return cornerpoints

def fingerboard_lines_detection(edges, theta_tol=5, rho_tol=100, is_visualize=False):
    '''
    fingerboard_lines = [[r1, t1, h1], [r2, t2, h2]]
    '''
    rho_tol = 100  # pixel
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(edges, theta=tested_angles)
    hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10, threshold=100)
    parallel_lines = []
    for [[hi, ri, ti], [hj, rj, tj]] in combinations(zip(hspace, dists, angles), 2):
        if angle_diff(ti, tj) < theta_tol / 180 * np.pi and abs(ri - rj) > rho_tol:
            parallel_lines.append([ri, ti, hi, rj, tj, hj])
    parallel_lines = np.array(parallel_lines)

    # find the pair of lines that has highest total count
    idx = np.argmax(parallel_lines[:, 2] + parallel_lines[:, 5])
    filtered_parallel_lines = parallel_lines[idx, :]
    
    # format data
    fingerboard_lines = np.zeros((2, 3))
    fingerboard_lines[0, :] = filtered_parallel_lines[:3]
    fingerboard_lines[1, :] = filtered_parallel_lines[3:]

    if is_visualize:
        cdst = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
        for line in fingerboard_lines:
            draw_line(cdst, line[0], line[1])
        return fingerboard_lines, cdst
    return fingerboard_lines, None

def rotate_fingerboard(frame, template):
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

def find_undistort_transform(rotated_edges, is_visualize=False):
    theta_range = 10/180*np.pi
    theta_tol = 5  # degree
    rho_tol = 10  # pixel
    tested_angles = np.linspace(-theta_range, theta_range, 10, endpoint=False)
    h, theta, d = hough_line(rotated_edges, theta=tested_angles)
    hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10)

    # sort by dist
    hspace, angles, dists = (list(item) for item in zip(*sorted(zip(hspace, angles, dists), key=lambda x: x[1])))
    
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

    if is_visualize:
        cdst = cv2.cvtColor(rotated_edges.copy(), cv2.COLOR_GRAY2BGR)
        for angle, dist in zip(angles, dists):
            draw_line(cdst, dist, angle)
        return homography, cdst, (hspace, angles, dists)
    return homography, None, (hspace, angles, dists)

def locate_fretboard(rotated_frame, rotated_edges, fingerboard_lines, template):
    cdst = cv2.cvtColor(rotated_frame.copy(), cv2.COLOR_GRAY2BGR)

    h, theta, d = hough_line(rotated_edges, theta=np.array([0.0]))
    hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=3, min_angle=1)

    hspace, angles, dists = (list(item) for item in zip(*sorted(zip(hspace, angles, dists), key=lambda x: x[2])))

    # first stage filter
    hspace = np.array(hspace).astype(float)
    dists = np.array(dists)
    while np.sum(np.diff(hspace) < -10) > 0:
        idxes = np.concatenate(([0], np.argwhere(np.diff(hspace) > -10).squeeze() + 1))
        hspace = hspace[idxes]
        dists = dists[idxes]
    
    # second stage filter (merge closely spaced lines)
    idxes = np.argwhere(np.diff(dists) < 10).squeeze()
    dists[idxes] = (dists[idxes] + dists[idxes+1]) / 2
    hspace[idxes] = (hspace[idxes] + hspace[idxes+1]) / 2
    dists = np.delete(dists, idxes+1)
    hspace = np.delete(hspace, idxes+1)

    # third stage filter
    fret_dists = np.diff(dists)
    fret_second_order_difference_normalized = np.diff(np.diff(fret_dists)) / fret_dists[1:-1]
    outliers_idx = np.argwhere(fret_second_order_difference_normalized[5:] > 1) + 5 + 1
    if outliers_idx is not None:
        # for outlier in list(outliers_idx):
        #     cv2.rectangle(cdst, (int(dists[outlier[0]]), 0), (int(dists[outlier[0]])+20, 20), (255, 255, 0), -1) 
        outliers_idx = outliers_idx.squeeze()
        dists = np.delete(dists, outliers_idx)
        hspace = np.delete(hspace, outliers_idx)

    for dist in dists:
        draw_line(cdst, dist, 0)

    cv2.namedWindow("locate_fretboard_debug")
    def mouse_callback(event, x, y, flags, param):
        idx = np.argmin(np.abs(np.array(dists) - x))
        cdst2 = cdst.copy()
        if abs(dists[idx] - x ) < 5:
            draw_line(cdst2, dists[idx], 0, thickness=3)
            cv2.putText(cdst2, f'{hspace[idx]}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
        cv2.imshow('locate_fretboard_debug', cdst2)
    cv2.setMouseCallback("locate_fretboard_debug", mouse_callback)

    rois = []
    features = np.zeros((dists.shape[0]-1, 1))
    for i, (ri, rj) in enumerate(zip(dists[0:-1], dists[1:])):
        point1 = line_line_intersection_fast(fingerboard_lines[0, :2], rj)
        point2 = line_line_intersection_fast(fingerboard_lines[1, :2], rj)

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
        for idx in fret_idx:
            cv2.rectangle(cdst, (int(dists[idx]), 0), (int(dists[idx+1]), 20), (0, 0, 255), -1)
    else:
        fret_idx = None
        cv2.rectangle(cdst, (int(dists[idx_12]), 0), (int(dists[idx_12])+20, 20), (255, 255, 255), -1)
        if idx_quadruplets[0] <=len(dists) and idx_quadruplets[1] <=len(dists):
            cv2.rectangle(cdst, (int(dists[idx_quadruplets[0]]), 0), (int(dists[idx_quadruplets[0]])+20, 20), (255, 255, 255), -1)
            cv2.rectangle(cdst, (int(dists[idx_quadruplets[1]]), 0), (int(dists[idx_quadruplets[1]])+20, 20), (255, 255, 255), -1)

    cv2.imshow('locate_fretboard_debug', cdst)

    if is_located:
        template.register_final_shift([int(dists[fret_idx[-1]-1]), 0])
        template.register_fretlines(dists[fret_idx] - int(dists[fret_idx[-1]-1]))
        template.register_template_image(rotated_frame[:, int(dists[fret_idx[-1]-1]):])
        rotated_frame_bb = [int(dists[fret_idx[-1]-1]), 0, int(dists[fret_idx[0]+2])-int(dists[fret_idx[-1]-1]), rotated_frame.shape[0]-1]
        return True
    else:
        return False

def guitar_detection(gray_img):
    template = Template()
    template.register_image(gray_img)

    # First stage: guitar direction detection
    edges = cv2.Canny(gray_img, 30, 100, 5)
    # edges = cv2.Sobel(frame, cv2.CV_8U, 1, 0, 3)

    fingerboard_lines, cdst = fingerboard_lines_detection(edges, theta_tol=5, rho_tol=100, is_visualize=True)
    template.register_fingerboard_lines(fingerboard_lines)

    rotated_frame, fingerboard_lines = rotate_fingerboard(gray_img, template)

    cornerpoints = cornerpoints_from_line_pair(fingerboard_lines, rotated_frame.shape[1])
    mask = np.zeros_like(rotated_frame)
    cv2.fillPoly(mask, pts=[cornerpoints], color=(255, 255, 255))
    rotated_frame = cv2.bitwise_and(rotated_frame, mask)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    rotated_frame = clahe.apply(rotated_frame)
    rotated_frame = cv2.bitwise_and(rotated_frame, mask)
    # rotated_frame = cv2.equalizeHist(rotated_frame)

    rotated_edges = cv2.Canny(rotated_frame, 50, 150, 5)
    
    # Second stage: fret detection
    homography, cdst, fret_lines = find_undistort_transform(rotated_edges, is_visualize=False)
    # cv2.imshow('fret detection', cdst)
    template.register_homography(homography)

    rotated_frame = cv2.warpPerspective(rotated_frame, homography, rotated_frame.shape[1::-1], flags=cv2.INTER_LINEAR)
    # draw_line(rotated_frame, filtered_parallel_lines[0][1], filtered_parallel_lines[0][2])
    # draw_line(rotated_frame, filtered_parallel_lines[0][3], filtered_parallel_lines[0][4])
    # cv2.imshow('rotated_frame', rotated_frame)

    rotated_edges = cv2.warpPerspective(rotated_edges, homography, rotated_edges.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    flag = locate_fretboard(rotated_frame, rotated_edges, fingerboard_lines, template)    
    if flag is True:
        # cdst = cv2.cvtColor(gray_img.copy(), cv2.COLOR_GRAY2BGR)
        # for line in template.fingerboard_lines:
        #     draw_line(cdst, line[0], line[1])
        # for fretline in template.image_fretlines:
        #     draw_line(cdst, fretline[0], fretline[1])
        # cv2.imshow('', cdst)
        # cv2.waitKey()
        return True, template
    else:
        return False, None

def boundingbox_12(template, homography, padding=0.2):
    fretlines = transform_houghlines(template.template_fretlines, homography)
    fingerboard_lines = transform_houghlines(template.template_fingerboard_lines, homography)

    cornerpoints = np.array([
        line_line_intersection(fingerboard_lines[0], fretlines[3]),
        line_line_intersection(fingerboard_lines[1], fretlines[3]),
        line_line_intersection(fingerboard_lines[1], fretlines[2]),
        line_line_intersection(fingerboard_lines[0], fretlines[2]),
    ])
    bb = cv2.boundingRect(cornerpoints.astype('float32'))
    return (int(bb[0] - bb[2] * 0.5 * padding), int(bb[1] - bb[3] * 0.5 * padding), int(bb[2] * (1+padding)), int(bb[3] * (1+padding))), (int(np.mean(cornerpoints, axis=0)[0]), int(np.mean(cornerpoints, axis=0)[1])), cornerpoints

def match_region(frame, template, homography, is_visualize=False):
    bb, _, cornerpoints = boundingbox_12(template, homography, padding=0)
    
    if is_visualize:
        cdst = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        for point in cornerpoints:
            cv2.circle(cdst, (int(point[0]), int(point[1])), 3, (255, 255, 0))
        cv2.imshow('cdst', cdst)
        cv2.waitKey()

    third_shift_mat = np.eye(3)
    third_shift_mat[0, 2] = bb[0]
    third_shift_mat[1, 2] = bb[1]

    fourth_shift_mat = np.eye(3)
    fourth_shift_mat[0, 2] = -template.template_fretlines[3, 0]
    fourth_shift_mat[1, 2] = 0

    debug = frame[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
    debug = cv2.warpPerspective(debug, fourth_shift_mat @ homography @ third_shift_mat, (int(template.template_fretlines[2, 0])-int(template.template_fretlines[3, 0]), template.template_image.shape[0]), flags=cv2.INTER_LINEAR)

    error = np.mean(np.abs(
        debug - 
        template.template_image[:, int(template.template_fretlines[3, 0]):int(template.template_fretlines[2, 0])]))
    
    if is_visualize:
        cv2.imshow('debug', debug)
        cv2.imshow('debug2', template.template_image_edge[:, int(template.template_fretlines[3, 0]):int(template.template_fretlines[2, 0])])
        cv2.waitKey()
    return error

def construct_epsilon_covering_pose_set(thetax, tz, epsilon=0.25):
    delta_tz = epsilon*tz**2 / (1-epsilon*tz)
    delta_tx = epsilon*(tz - np.sqrt(2)*np.sin(thetax))
    delta_ty = delta_tx 
    
    delta_theta_zt = epsilon*tz
    delta_theta_zc = epsilon*tz
    delta_theta_x = np.arcsin(tz - 1/(epsilon + 1/(tz - np.sin(thetax)))) - thetax
    return (delta_tx, delta_tz, delta_theta_x, delta_theta_zc)

def extract_keypoints(angles, dists, image):
    kps = []
    for idx1, idx2 in list(combinations(np.arange(len(angles)), 2)):
        if angle_diff(angles[idx1], angles[idx2]) > 0.01:
            kps.append(line_line_intersection([dists[idx1], angles[idx1]], [dists[idx2], angles[idx2]]))
    kps = np.array(kps)
    if kps.shape[0] == 0:
        return kps
    else:
        return kps[(kps[:, 0] >= 0) & (kps[:, 0] <= image.shape[1]) & (kps[:, 1] >= 0) & (kps[:, 1] <= image.shape[0]), :]

def arange_quadrangle_points(quad):
    quad_sorted = np.zeros((4, 2))
    sorted_x = np.argsort(quad[:, 0])
    if quad[sorted_x[0], 1] < quad[sorted_x[1], 1]:
        quad_sorted[[0,1], :] = quad[[sorted_x[0], sorted_x[1]], :]
    else:
        quad_sorted[[0,1], :] = quad[[sorted_x[1], sorted_x[0]], :]
    
    if quad[sorted_x[2], 1] < quad[sorted_x[3], 1]:
        quad_sorted[[3, 2], :] = quad[[sorted_x[2], sorted_x[3]], :]
    else:
        quad_sorted[[2, 3], :] = quad[[sorted_x[3], sorted_x[2]], :]
    return quad_sorted

def quadrangle_shape_difference(quad1, quad2):
    '''
    quad1 = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    quad2 = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    '''
    quad1_feature = quad1 - np.mean(quad1, axis=0)
    quad2_feature = quad2 - np.mean(quad2, axis=0)
    # quad1_feature = quad1
    # quad2_feature = quad2
    return np.sum(np.sum((quad1_feature - quad2_feature)**2))

def quadrangle_ratio(quad):
    length1 = np.linalg.norm(quad[1, :] - quad[0, :])
    length2 = np.linalg.norm(quad[2, :] - quad[1, :])
    return length2 / length1

def draw_quadrangle(cdst, quad, color=(0, 0, 255)):
    for point in quad:
        cv2.circle(cdst, (int(point[0]), int(point[1])), 3, color)
    for idx in ((0, 1), (1, 2), (2, 3), (3, 0)):
        cv2.line(cdst, quad[idx[0], :].astype('int'), quad[idx[1], :].astype('int'), color, 1)

def detect_fingerboardline_near_angles(edges, line, theta_tol_deg=5, rho_tol=30):
    tested_angles = np.linspace(line[1]-theta_tol_deg/180*np.pi, line[1]+theta_tol_deg/180*np.pi, 2*theta_tol_deg+1, endpoint=False)
    h, theta, d = hough_line(edges, theta=tested_angles)
    hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=5, min_angle=3, threshold=int(50))
    idx = np.abs(dists - line[0]) <= rho_tol
    return hspace[idx], angles[idx], dists[idx]

def detect_fretline_near_angles(edges, line, theta_tol_deg=5, rho_tol=30):
    tested_angles = np.linspace(line[1]-theta_tol_deg/180*np.pi, line[1]+theta_tol_deg/180*np.pi, 2*theta_tol_deg+1, endpoint=False)
    h, theta, d = hough_line(edges, theta=tested_angles)
    hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=5, min_angle=5, threshold=int(30))
    idx = np.abs(dists - line[0]) <= rho_tol
    return hspace[idx], angles[idx], dists[idx]

def detect_lines_near_angles(edges, lines_rt, theta_tol_deg=5, threshold=None):
    '''
    lines_rt = np.array([[r1, t1], [r2, t2], ...])
    '''
    tested_angles = np.array(
        [np.linspace(line[1]-theta_tol_deg/180*np.pi, line[1]+theta_tol_deg/180*np.pi, 2*theta_tol_deg+1, endpoint=False) for line in lines_rt]
        ).flatten()
    tested_angles = np.unique(tested_angles)
    h, theta, d = hough_line(edges, theta=tested_angles)
    if threshold is None:
        hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=11, min_angle=3, threshold=int(0.8*np.min([line[2] for line in lines_rt])))
    else:
        hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=11, min_angle=3, threshold=threshold)
    return hspace, angles, dists

def calculate_ncc(image, quad, template, template_quad):
    bb = cv2.boundingRect(quad.astype('float32'))
    bb2 = cv2.boundingRect(template_quad.astype('float32'))

    homography, _ = cv2.findHomography(quad, template_quad, cv2.RANSAC,5.0)

    third_shift_mat = np.eye(3)
    third_shift_mat[0, 2] = bb[0]
    third_shift_mat[1, 2] = bb[1]

    Ic = image[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
    Ic = cv2.warpPerspective(Ic, homography @ third_shift_mat, (template.template_image.shape[1], template.template_image.shape[0]), flags=cv2.INTER_LINEAR)
    Ic = Ic[bb2[1]:bb2[1]+bb2[3], bb2[0]:bb2[0]+bb2[2]]

    It = template.template_image[bb2[1]:bb2[1]+bb2[3], bb2[0]:bb2[0]+bb2[2]]
    mask = It != 0

    return np.mean((Ic[mask] - np.mean(Ic[mask])) * (It[mask] - np.mean(It[mask]))) / np.std(Ic[mask]) / np.std(It[mask])

def inspect_ncc(image, quad, template, template_quad):
    bb = cv2.boundingRect(quad.astype('float32'))
    bb2 = cv2.boundingRect(template_quad.astype('float32'))

    homography, _ = cv2.findHomography(quad, template_quad, cv2.RANSAC,5.0)

    third_shift_mat = np.eye(3)
    third_shift_mat[0, 2] = bb[0]
    third_shift_mat[1, 2] = bb[1]

    Ic = image[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
    Ic = cv2.warpPerspective(Ic, homography @ third_shift_mat, (template.template_image.shape[1], template.template_image.shape[0]), flags=cv2.INTER_LINEAR)
    Ic = Ic[bb2[1]:bb2[1]+bb2[3], bb2[0]:bb2[0]+bb2[2]]

    It = template.template_image[bb2[1]:bb2[1]+bb2[3], bb2[0]:bb2[0]+bb2[2]]
    mask = It != 0

    Ic[~mask] = 0
    return Ic, It

def first_stage_filter(quadrangles_idxes, quadrangles, prev_quad, area_tol=0.2, ratio_tol=0.2):
        areas = np.array([quad_area(quad) for quad in quadrangles])
        filtered_idx = np.abs(areas / quad_area(prev_quad) - 1 ) <= area_tol
        # print(quadrangles_idxes.shape[0], np.sum(filtered_idx))

        quadrangles_idxes = quadrangles_idxes[filtered_idx]
        quadrangles = quadrangles[filtered_idx, ...]

        ratios = np.array([quadrangle_ratio(quad) for quad in quadrangles])
        filtered_idx = np.abs(ratios - quadrangle_ratio(prev_quad)) <= ratio_tol
        # print(quadrangles_idxes.shape[0], np.sum(filtered_idx))

        quadrangles_idxes = quadrangles_idxes[filtered_idx]
        quadrangles = quadrangles[filtered_idx, ...]
        return quadrangles_idxes, quadrangles

def quad_from_quad_line(quad_lines):
    quad = np.array([
        line_line_intersection(quad_lines[0], quad_lines[3]),
        line_line_intersection(quad_lines[1], quad_lines[3]),
        line_line_intersection(quad_lines[1], quad_lines[2]),
        line_line_intersection(quad_lines[0], quad_lines[2]),
    ])
    return quad

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    quad = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ])
    return (xmin, ymin, xmax-xmin, ymax-ymin), quad

def training(template, xmin, xmax, H, image, n_samples=int(1e3)):
    H = H / H[2, 2]

    It = template.template_image[:, xmin:xmax]
    mask = It != 0
    bb_It, quad_It = bbox2(It)

    Ic = Ic_from_homography(xmin, xmax, template, image, H)
    # cv2.imshow('It', It)
    # cv2.imshow('Ic', Ic)
    # cv2.waitKey()

    dI = []
    Vref = Ic[mask].astype('float32') / 255
    
    M = np.random.uniform(-1, 1, (8, n_samples)) 
    M[[0, 1, 3, 4], :] = M[[0, 1, 3, 4], :] * 3e-3
    M[[2, 5], :] = M[[2, 5], :] * 10
    M[6:8, :] = M[6:8, :]*1e-6
    for i in range(n_samples):

        deltaH = np.eye(3)
        deltaH[0, :] = deltaH[0, :] + M[:3, i]
        deltaH[1, :] = deltaH[1, :] + M[3:6, i]
        deltaH[2, :2] = M[6:8, i]
        deltaH = H @ deltaH

        Ic = Ic_from_homography(xmin, xmax, template, image, deltaH)
        dI.append(Vref - Ic[mask].astype('float32') / 255)
        # cv2.imshow('Ic', Ic)
        # cv2.waitKey()
    dI = np.array(dI).T
    
    A = np.linalg.lstsq(dI.T, M.T)[0].T
    return A, mask, Vref, M, dI

def npprint(*data):
    with np.printoptions(precision=3, suppress=True):
        print(*data)

def robust_findhomography(X, Y):
    '''
    Find H such that H @ X = Y
    X is n * 2
    Y is n * 2
    '''
    Xmean = np.mean(X, axis=0)
    Xscale = np.sqrt(2) / np.mean(np.linalg.norm(X - Xmean, axis=1))

    A = np.eye(3)
    A[:2, 2] = (-Xmean).reshape((2,))
    A[:2, :] = A[:2, :] * Xscale

    X = np.hstack((X, np.ones((X.shape[0], 1)))).T
    Xprime = A @ X

    Ymean = np.mean(Y, axis=0)
    Yscale = np.sqrt(2) / np.mean(np.linalg.norm(Y - Ymean, axis=1))

    B = np.eye(3)
    B[:2, 2] = (-Ymean).reshape((2,))
    B[:2, :] = B[:2, :] * Yscale

    Y = np.hstack((Y, np.ones((Y.shape[0], 1)))).T
    Yprime = B @ Y

    Hprime, _ = cv2.findHomography(Xprime[:2, :].T, Yprime[:2, :].T, cv2.RANSAC,5.0)

    H = np.linalg.inv(B) @ Hprime @ A

    return H / H[2, 2]

def Ic_from_homography(xmin, xmax, template, frame, homography):
    It = template.template_image[:, xmin:xmax]
    bb_It, quad_It = bbox2(It)

    bb_Ic = cv2.boundingRect(transform_points(quad_It + np.array([xmin, 0]).reshape((1, 2)), np.linalg.inv(homography)).astype('float32'))
    third_shift_mat = np.eye(3)
    third_shift_mat[0, 2] = bb_Ic[0]
    third_shift_mat[1, 2] = bb_Ic[1]
    fourth_shift_mat = np.eye(3)
    fourth_shift_mat[0, 2] = -xmin
    fourth_shift_mat[1, 2] = 0

    Ic = frame[bb_Ic[1]:bb_Ic[1]+bb_Ic[3], bb_Ic[0]:bb_Ic[0]+bb_Ic[2]]
    Ic = cv2.warpPerspective(Ic, fourth_shift_mat @ homography @ third_shift_mat, (It.shape[1], It.shape[0]), flags=cv2.INTER_LINEAR)
    return Ic

def update_homography(xmin, xmax, template, frame, homography, Vref, mask, A):
    Ic = Ic_from_homography(xmin, xmax, template, frame, homography)
    # cv2.imshow('Ic', Ic)
    
    deltaI = Vref - Ic[mask].astype('float32') / 255
    # deltamu = M[:, np.argmin(np.linalg.norm(dI - deltaI.reshape((-1,1)), axis=0))]
    deltamu = A @ deltaI

    deltaH = np.eye(3)
    deltaH[0, :] = deltaH[0, :] + deltamu[:3]
    deltaH[1, :] = deltaH[1, :] + deltamu[3:6]
    deltaH[2, :2] = deltamu[6:8]

    return homography @ np.linalg.inv(deltaH), deltaH

def jac(R, t, n, target_image, template, xmin, xmax):
    H = R + t.reshape((3, 1)) @ n[0].T

    It = template.template_image[:, xmin:xmax]
    mask = It != 0

    fourth_shift_mat = np.eye(3)
    fourth_shift_mat[0, 2] = template.template_fretlines[3, 0]
    fourth_shift_mat[1, 2] = 0

    x = transform_points(np.argwhere(mask.T != 0), np.linalg.inv(H) @ fourth_shift_mat)
    x = np.vstack((x.T, np.ones((1, x.shape[0]))))

    J = np.zeros((x.shape[1], 6))

    pos = x.T[:, :2].astype('int')

    delta_x = x.T[:, 0] - pos[:, 0]
    dJdv = ((1 - delta_x) * target_image[pos[:, 1]+1, pos[:, 0]] + delta_x * target_image[pos[:, 1]+1, pos[:, 0]+1]) - \
    ((1 - delta_x) * target_image[pos[:, 1], pos[:, 0]] + delta_x * target_image[pos[:, 1], pos[:, 0]+1])
    dJdv = dJdv / 255

    delta_y = x.T[:, 1] - pos[:, 1]
    dJdu = ((1 - delta_y) * target_image[pos[:, 1], pos[:, 0]+1] + delta_y * target_image[pos[:, 1]+1, pos[:, 0]+1]) - \
    ((1 - delta_y) * target_image[pos[:, 1], pos[:, 0]] + delta_y * target_image[pos[:, 1]+1, pos[:, 0]])
    dJdu = dJdu / 255

    x_tilde = H @ x

    dRdr = cv2.Rodrigues(cv2.Rodrigues(R)[0])[1].T

    for i in range(J.shape[0]):
        x_ = x[:, i]
        x_tilde_ = x_tilde[:, i]

        dudx_tilde = np.array([
            [1 / x_tilde_[2], 0, -x_tilde_[0]/x_tilde_[2]**2],
            [0, 1 / x_tilde_[2], -x_tilde_[1]/x_tilde_[2]**2]
        ])

        dudt = np.dot(n[0].T, x_tilde_) * dudx_tilde

        dx_tildedR = np.array([
            [x_[0], x_[1], x_[2], 0, 0, 0, 0, 0, 0],
            [0, 0, 0, x_[0], x_[1], x_[2], 0, 0, 0],
            [0, 0, 0, 0, 0, 0, x_[0], x_[1], x_[2]]
        ])

        dudr = dudx_tilde @ dx_tildedR @ dRdr

        J[i, :] = np.array([dJdu[i], dJdv[i]]).reshape((1, 2)) @ np.hstack((dudr, dudt))

    return J

def delta_p(rvec, t, n, target_image, template, xmin, xmax):
    R = cv2.Rodrigues(rvec)[0]
    H = R + t.reshape((3, 1)) @ n[0].T
    J = jac(R, t, n, target_image, template, xmin, xmax)

    It = template.template_image[:, xmin:xmax]
    mask = It != 0
    It = It.T[mask.T]

    Ic = Ic_from_homography(xmin, xmax, template, target_image, H) 
    Ic = Ic.T[mask.T]

    grad = np.mean((It - Ic).reshape((-1,1)) * J, axis=0) * 2
    step = np.linalg.inv(J.T @ J) @ J.T @ (It - Ic) / 255
    return step, grad

def matching_error(rvec, tvec, n, image, template, xmin, xmax):
    R = cv2.Rodrigues(rvec)[0]
    H = R + tvec.reshape((3, 1)) @ n[0].T

    It = template.template_image[:, xmin:xmax]
    mask = It != 0

    Ic = Ic_from_homography(xmin, xmax, template, image, H)

    error = np.mean((
        Ic[mask] - 
        It[mask])**2)
    # return np.mean((Ic[mask] - np.mean(Ic[mask])) * (It[mask] - np.mean(It[mask]))) / np.std(Ic[mask]) / np.std(It[mask])
    return error

def refine_pose(H, frame, xmin, xmax, template):
    _, rot, t, n = cv2.decomposeHomographyMat(H, np.eye(3))

    norm_tol = 1e-3
    alpha = 0.5
    c = 1e-4
    norm = 1
    rot_vec = cv2.Rodrigues(rot[0])[0]
    t_vec = t[0]
    while norm > norm_tol:
        deltap, grad = delta_p(rot_vec, t_vec, n, frame, template, xmin, xmax)
        while matching_error(rot_vec+deltap[:3].reshape((3, 1)), 
                             t_vec+deltap[3:].reshape((3, 1)), n, 
                             frame, template, xmin, xmax) > matching_error(rot_vec, t_vec, n, frame, template, xmin, xmax) + c * grad @ deltap:
            deltap = deltap * alpha

        # cdst = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        # homography = cv2.Rodrigues(rot_vec)[0] + t_vec.reshape((3, 1)) @ n[0].T
        # _, _, cornerpoints = boundingbox_12(template, homography, padding=0)
        # draw_quadrangle(cdst, cornerpoints)
        
        rot_vec = rot_vec+deltap[:3].reshape((3, 1))
        t_vec = t_vec+deltap[3:].reshape((3, 1))
        norm = np.linalg.norm(deltap)

    homography = cv2.Rodrigues(rot_vec)[0] + t_vec.reshape((3, 1)) @ n[0].T
    return homography

def coords_in_hough(lines0, theta, d):
    np.searchsorted(d, lines0[:, 0]) - 1
    np.searchsorted(theta, lines0[:, 1]) - 1

    coords = np.vstack((np.searchsorted(d, lines0[:, 0]) - 1, np.searchsorted(theta, lines0[:, 1]) - 1)).T
    deltax = (lines0[:, 0] - d[coords[:, 0]]) / (d[1] - d[0])
    deltay = (lines0[:, 1] - theta[coords[:, 1]]) / (theta[1] - theta[0])
    return coords, deltax, deltay

def interpolate_from_coords(coords, deltax, deltay, image):
    fxy1 = (1 - deltax) * image[coords[:, 0], coords[:, 1]] + deltax * image[coords[:, 0], coords[:, 1] + 1]
    fxy2 = (1 - deltax) * image[coords[:, 0]+1, coords[:, 1]] + deltax * image[coords[:, 0]+1, coords[:, 1] + 1]
    return (1 - deltay) * fxy1 + deltay * fxy2

def bb_from_outer_quad(outer_quad):
    rect = cv2.minAreaRect(outer_quad) # ((x-coordinate, y-coordinate),(width, height), rotation)
    if rect[1][1] > rect[1][0]:
        angle = -(90-rect[2])
        padding = 0.2*rect[1][0]
        padded_rect = (rect[0], (rect[1][0]+2*padding, rect[1][1]), rect[2])
    else:
        angle = rect[2]
        padding = 0.2*rect[1][1]
        padded_rect = (rect[0], (rect[1][0], rect[1][1]+2*padding), rect[2])
    rot_mat = cv2.getRotationMatrix2D(rect[0], angle, 1.0)

    # clip rotated image
    if padded_rect[1][1] > padded_rect[1][0]:
        delta = 0.5*padded_rect[1][0]
    else:
        delta = 0.5*padded_rect[1][1]
    
    template.register_shift([0, max(int(padded_rect[0][1]-delta), 0)])
    return padded_rect, rot_mat, [0, max(int(padded_rect[0][1]-delta), 0)]

def point_set_registration_cpd(template, array):
    '''
    find scale and offset such that:
        scale * template + offset <-> subset of array
    constraints:
        min(template) > 0

    offset
    |            |\ 
    |            | \ offset < np.max(array) - template[0] * scale
    |            |  \ 
    |            |   |
    |   scale_min|   |scale_max
    |            |   |
    |            |___|
    |             offset_min
    ---------------------------------- scale
    '''
    array_max = np.max(array)

    offset_min = np.min(array)

    span_template = template[0] - template[-1]
    span_array = array_max - np.min(array)
    scale_min = 0.4 * span_array / span_template
    scale_max = 0.9 * span_array / span_template
    offset_bound_fun = lambda scale: array_max - template[0] * scale

    plt.figure()
    # plt.axhline(y=scale_min, xmin=offset_min, xmax=offset_bound_fun(scale_min))
    plt.axhline(y=offset_min, color='k')
    plt.axvline(x=scale_min, color='k')
    plt.axvline(x=scale_max, color='k')
    plt.plot(np.linspace(scale_min, scale_max), offset_bound_fun(np.linspace(scale_min, scale_max)))
    plt.xlim(scale_min-0.1, scale_max+0.1)
    plt.ylim(offset_min-20, offset_bound_fun(scale_min)+20)
    plt.show()

    # init
    M = template.shape[0]
    N = array.shape[0]
    # scale = 0.934
    # offset = 267
    scale = (scale_min + scale_max) / 2
    offset = (offset_min + offset_bound_fun(scale)) / 2
    w = 0.5
    var = np.sum(((scale * template + offset).reshape((-1, 1)) - array.reshape((1, -1)))**2) / M / N

    counter = 1
    while counter < 50:
        print(scale, offset, var)
        if counter % 10 == 0:
            plt.figure()
            plt.scatter(array, np.zeros_like(array))
            plt.scatter(scale * template + offset, np.zeros_like(template))
            plt.show()

        # E-step: compute P
        numerator = np.exp(-((scale * template + offset).reshape((-1, 1)) - array.reshape((1, -1)))**2 / 2 / var)
        denominator = np.sum(numerator, axis=0) + np.sqrt(2*np.pi*var)*w/(1-w)*M / N
        P = numerator / denominator

        # M-step: compute scale, offset
        c1 = np.sum(P)
        cx = np.sum(P @ array.reshape((N, 1)))
        cy = np.sum(template.reshape((1, M)) @ P)
        cy2 = np.sum(template.reshape((1, M))**2 @ P)
        cxy = template.reshape((1, M)) @ P @ array.reshape((N, 1))

        scale = ((cxy - cx*cy/c1) / (cy2 - cy**2/c1))[0, 0]
        offset = (cx - cy * scale) / c1
        var = 1/c1 * np.sum(P * ((scale * template + offset).reshape((-1, 1)) - array.reshape((1, -1)))**2)
        counter = counter + 1


def point_set_registration(template, array):
    '''
    find scale and offset such that:
        scale * template + offset <-> subset of array
    constraints:
        min(template) > 0

    offset
    |            |\
    |            | \ offset < np.max(array) - template[0] * scale
    |            |  \ 
    |            |   |
    |   scale_min|   |scale_max
    |            |   |
    |            |___|
    |             offset_min
    ---------------------------------- scale
    '''
    array_max = np.max(array)

    offset_min = np.min(array)

    span_template = template[0] - template[-1]
    span_array = array_max - np.min(array)
    scale_min = 0.4 * span_array / span_template
    scale_max = 0.9 * span_array / span_template
    offset_bound_fun = lambda scale: array_max - template[0] * scale

    plt.figure()
    # plt.axhline(y=scale_min, xmin=offset_min, xmax=offset_bound_fun(scale_min))
    plt.axhline(y=offset_min, color='k')
    plt.axvline(x=scale_min, color='k')
    plt.axvline(x=scale_max, color='k')
    plt.plot(np.linspace(scale_min, scale_max), offset_bound_fun(np.linspace(scale_min, scale_max)))
    plt.xlim(scale_min-0.1, scale_max+0.1)
    plt.ylim(offset_min-20, offset_bound_fun(scale_min)+20)
    plt.show()

    # init
    M = template.shape[0]
    N = array.shape[0]
    # scale = 0.934
    # offset = 267

    # choose two random correspondence



if __name__ == '__main__':
    cap = cv2.VideoCapture('test\\guitar2.mp4')

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 450)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(frame, 30, 100, 5)

    flag, template = guitar_detection(frame)

    template_outer_quad = np.array([
            line_line_intersection(template.template_fingerboard_lines[0], template.template_fretlines[-1]),
            line_line_intersection(template.template_fingerboard_lines[1], template.template_fretlines[-1]),
            line_line_intersection(template.template_fingerboard_lines[1], template.template_fretlines[0]),
            line_line_intersection(template.template_fingerboard_lines[0], template.template_fretlines[0])
        ]).astype('float32')
    
    lines_template = []
    lines_template.append(template.template_fingerboard_lines[0][:2])
    lines_template.append(template.template_fingerboard_lines[1][:2])
    for i in range(9):
        lines_template.append(template.template_fretlines[2*i])
        lines_template.append(template.template_fretlines[2*i+1])
    lines_template = np.array(lines_template)

    outer_quad = transform_points(template_outer_quad, np.linalg.inv(template.total_mat)).astype('float32')
    padded_rect, rot_mat_, shift_ = bb_from_outer_quad(outer_quad)

    rot_mat = np.zeros((3, 3))
    rot_mat[2, 2] = 1
    rot_mat[:2, :] = rot_mat_
    
    shift_mat = np.eye(3)
    shift_mat[0, 2] = -shift_[0]
    shift_mat[1, 2] = -shift_[1]

    rotated_edges = cv2.warpPerspective(edges, shift_mat @ rot_mat, (edges.shape[1], int(min(padded_rect[1]))), flags=cv2.INTER_LINEAR)
    lines0 = transform_houghlines(lines_template, template.total_mat @ np.linalg.inv(shift_mat @ rot_mat))
    idxes = np.argwhere((lines0[:, 1] > np.pi/2) | (lines0[:, 1] < -np.pi/2))
    for idx in idxes:
        lines0[idx[0], 0] = -lines0[idx[0], 0]
        lines0[idx[0], 1] = lines0[idx[0], 1] - np.sign(lines0[idx[0], 1]) * np.pi

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=True)
    h, theta, d = hough_line(rotated_edges, theta=tested_angles)

    coords, deltax, deltay = coords_in_hough(lines0, theta, d)
    assert(np.all(coords > 0))
    h_template = interpolate_from_coords(coords, deltax, deltay, h)

    line_params = np.linalg.lstsq(coords[2:, [1, 0]], np.ones((coords.shape[0]-2, 1)))[0]

    h_n = (h / np.max(h)).astype('float32')
    cdst = cv2.cvtColor(h_n.copy(), cv2.COLOR_GRAY2BGR)
    draw_line(cdst, 1 / np.linalg.norm(line_params), np.arctan2(line_params[1] / np.linalg.norm(line_params), line_params[0] / np.linalg.norm(line_params))[0])
    cdst = cv2.resize(cdst, (0, 0), fx = 1, fy = 0.25)
    for coord in coords:
        cv2.circle(cdst, (int(coord[1]), int(coord[0]/4)), 1, (255, 255, 0), -1)
    cv2.imshow('rotated_edges', rotated_edges)
    cv2.imshow('houghline', cdst)
    # cv2.waitKey()

    fret_num = [3, 5, 7, 9, 12, 15, 17, 19, 21]
    template_quad = {}
    for i in range(9):
        template_quad[fret_num[i]] = np.array([
            line_line_intersection(template.template_fingerboard_lines[0], template.template_fretlines[2*i+1]),
            line_line_intersection(template.template_fingerboard_lines[1], template.template_fretlines[2*i+1]),
            line_line_intersection(template.template_fingerboard_lines[1], template.template_fretlines[2*i]),
            line_line_intersection(template.template_fingerboard_lines[0], template.template_fretlines[2*i])
        ])
    
    homography = template.total_mat

    template_quads_center = np.array([np.mean(template_quad[fret], axis=0) for fret in fret_num])
    prev_quads_center = transform_points(template_quads_center, np.linalg.inv(homography))
    
    # prev_quad_frets = [3, 5, 7, 9, 12, 15, 17, 19, 21]
    # prev_quad_lines = []
    # for i in range(9):
    #     prev_quad_lines.append([
    #         template.fingerboard_lines[0],
    #         template.fingerboard_lines[1],
    #         template.image_fretlines[2*i],
    #         template.image_fretlines[2*i+1]
    #     ])
    # prev_quads = [quad_from_quad_line(x) for x in prev_quad_lines]
    # prev_quads_center = [np.mean(x, axis=0) for x in prev_quads]
    # prev_quads_center = np.array(prev_quads_center)

    # MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(history=10000, detectShadows = False)
    # KNN_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold = 4000.0, detectShadows = False)
    # bg_substractor = MOG2_subtractor

    counter = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # foreground_mask = bg_substractor.apply(frame)
        # cv2.imshow('foreground_mask', foreground_mask)
        # cv2.imshow('background_image', bg_substractor.getBackgroundImage())

        rotated_frame = cv2.warpPerspective(frame, shift_mat @ rot_mat, (frame.shape[1], int(min(padded_rect[1]))), flags=cv2.INTER_LINEAR)

        edges = cv2.Canny(rotated_frame, 30, 150, 5)

        cdst = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 180, minLineLength=10, maxLineGap=50)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(cdst, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow('edges_cdst', cdst)

        thresh = cv2.adaptiveThreshold(rotated_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 0)
        cv2.imshow('thresh', thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9, 9))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
        opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('opening', opening)

        # edges = cv2.Canny(opening, 30, 100, 5)
        # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        # h, theta, d = hough_line(edges, theta=tested_angles)
        # hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10, threshold=200)
        # cdst = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
        # for line in zip(dists, angles):
        #     draw_line(cdst, line[0], line[1])
        # cv2.imshow('fingerboard_lines', cdst)

        # h_n = (h / np.max(h)).astype('float32')
        # cdst_h = cv2.cvtColor(h_n.copy(), cv2.COLOR_GRAY2BGR)
        # draw_line(cdst, 125, -0.035)
        # draw_line(cdst_h, 1 / np.linalg.norm(line_
        # params), np.arctan2(line_params[1] / np.linalg.norm(line_params), line_params[0] / np.linalg.norm(line_params))[0])
        # coords, deltax, deltay = coords_in_hough(np.hstack((dists.reshape((-1,1)), angles.reshape((-1,1)))), theta, d)
        # cdst_h = cv2.resize(cdst_h, (0, 0), fx = 1, fy = 0.25)
        # for coord in coords:
            # cv2.circle(cdst_h, (int(coord[1]), int(coord[0]/4)), 1, (255, 255, 0), -1)
        # cv2.imshow('houghline', cdst_h)

        # valid_lines_idxes = np.arange(dists.size)
        
        # min_idx = np.argmin(dists[valid_lines_idxes])
        # max_idx = np.argmax(dists[valid_lines_idxes])
        # fingerboard_lines = np.zeros((2, 2))
        # fingerboard_lines[0, 0] = dists[valid_lines_idxes[min_idx]]
        # fingerboard_lines[0, 1] = angles[valid_lines_idxes[min_idx]]
        # fingerboard_lines[1, 0] = dists[valid_lines_idxes[max_idx]]
        # fingerboard_lines[1, 1] = angles[valid_lines_idxes[max_idx]]
        # idxes = np.argwhere(fingerboard_lines[:, 0] < 0)
        # for idx in idxes:
        #     fingerboard_lines[idx[0], 0] = -fingerboard_lines[idx[0], 0]
        #     fingerboard_lines[idx[0], 1] = fingerboard_lines[idx[0], 1] - np.sign(fingerboard_lines[idx[0], 1]) * np.pi

        # cdst = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
        # for line in fingerboard_lines:
        #     draw_line(cdst, line[0], line[1])
        # cv2.imshow('fingerboard_lines_filtered', cdst)
        # cv2.waitKey()

        analysis = cv2.connectedComponentsWithStats(opening, 4, cv2.CV_32S) 
        (totalLabels, label_ids, values, centroid) = analysis

        valid_idxes = []
        output = np.zeros(opening.shape, dtype="uint8")
        for i in range(1, totalLabels): 
            area = values[i, cv2.CC_STAT_AREA]   
        
            # adaptive area thresholding
            if (area > 4000) or (area < 50):
                continue

            # if point_line_distance(fingerboard_lines[0, :], centroid[i]) > 0 and \
            #     point_line_distance(fingerboard_lines[1, :], centroid[i]) < 0:
            valid_idxes.append(i)
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)

        rotated_prev_quads_center = transform_points(prev_quads_center, shift_mat @ rot_mat)
        rotated_tracked_quads_center = []
        min_dist = []
        for center in rotated_prev_quads_center:
            idx_ = np.argmin(np.linalg.norm(centroid[valid_idxes] - center, axis=1))
            rotated_tracked_quads_center.append(centroid[valid_idxes[idx_]])
            min_dist.append(np.linalg.norm(centroid[valid_idxes[idx_]] - center))
        rotated_tracked_quads_center = np.array(rotated_tracked_quads_center)

        rotated_tracked_quads_center = rotated_tracked_quads_center

        cdst_quads = cv2.cvtColor(output.copy(), cv2.COLOR_GRAY2BGR)
        for point in rotated_tracked_quads_center:
            cv2.circle(cdst_quads, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

        affine_mat, inliers = cv2.estimateAffinePartial2D(rotated_prev_quads_center, 
                                                          rotated_tracked_quads_center, method=cv2.RANSAC, ransacReprojThreshold=10)
        print(np.linalg.norm(affine_mat[:, 0]))
        affine_mat = np.vstack((affine_mat, np.array([0, 0, 1]).reshape((1, 3))))

        # see if we can add more inliers
        inliers_line = np.linalg.lstsq(rotated_tracked_quads_center[inliers.squeeze().astype('bool')], np.ones((np.sum(inliers), 1)))[0]
        inliers_line_r = 1 / np.linalg.norm(inliers_line)
        inliers_line_dists = ((rotated_tracked_quads_center @ inliers_line - 1) * inliers_line_r).squeeze()
        inliers_threshold = np.max(np.abs(inliers_line_dists[inliers.squeeze().astype('bool')]))
        # outliers_idx = np.argwhere(~inliers.squeeze().astype('bool')).reshape((-1,))
        # if np.sum(~inliers.squeeze().astype('bool')) > 0:
        #     outliers_idx = np.array([idx for idx in outliers_idx if np.abs(inliers_line_dists[idx]) < inliers_threshold])
        #     if outliers_idx.size > 0:
        #         inliers.reshape((-1,))[outliers_idx] = 1
        #         affine_mat, inliers2 = cv2.estimateAffinePartial2D(rotated_prev_quads_center[inliers.squeeze().astype('bool')], 
        #                                                           rotated_tracked_quads_center[inliers.squeeze().astype('bool')], 
        #                                                           method=cv2.RANSAC, ransacReprojThreshold=np.inf)
        #         affine_mat = np.vstack((affine_mat, np.array([0, 0, 1]).reshape((1, 3))))

        prev_quads_center = transform_points(prev_quads_center, np.linalg.inv(shift_mat @ rot_mat) @ affine_mat @ shift_mat @ rot_mat)
        for point in rotated_prev_quads_center:
            cv2.circle(cdst_quads, (int(point[0]), int(point[1])), 7, (255, 255, 0), -1)
        for point in transform_points(prev_quads_center, shift_mat @ rot_mat):
            cv2.circle(cdst_quads, (int(point[0]), int(point[1])), 3, (255, 0, 255), -1)
        for point in transform_points(prev_quads_center, shift_mat @ rot_mat)[inliers.squeeze().astype('bool'), :]:
            cv2.circle(cdst_quads, (int(point[0]), int(point[1])), 9, (255, 0, 255), 3)
        
        cv2.imshow('cdst_quads', cdst_quads)

        outer_quad = transform_points(outer_quad, np.linalg.inv(shift_mat @ rot_mat) @ affine_mat @ shift_mat @ rot_mat).astype('float32')

        padded_rect, rot_mat_, shift_ = bb_from_outer_quad(outer_quad)

        rot_mat = np.zeros((3, 3))
        rot_mat[2, 2] = 1
        rot_mat[:2, :] = rot_mat_
        
        shift_mat = np.eye(3)
        shift_mat[0, 2] = -shift_[0]
        shift_mat[1, 2] = -shift_[1]

        print(counter)

        # h_n = (h / np.max(h)).astype('float32')
        # cdst = cv2.cvtColor(h_n.copy(), cv2.COLOR_GRAY2BGR)
        # coords, _, _ = coords_in_hough(np.vstack((dists, angles)).T, theta, d)

        # idxes = np.argwhere(np.abs(((coords[:, [1, 0]] @ line_params - 1)) / np.linalg.norm(line_params)) < 10)[:, 0]

        # # estimate vanishing point
        # vanishing_point = np.linalg.lstsq(np.vstack((np.cos(angles[idxes]), np.sin(angles[idxes]))).T, dists[idxes])[0]

        # l1, l2 = np.linalg.lstsq(np.array([[vanishing_point[0], vanishing_point[1]],[edges.shape[1], edges.shape[0]]]), [-1, 0])[0]

        # # sine_params = np.linalg.lstsq(np.vstack((angles[idxes], dists[idxes])).T, np.ones((idxes.shape[0], 1)))[0]
        # # sine_r = 1 / np.linalg.norm(sine_params)
        # # sine_params = sine_params * sine_r

        # # plt.figure()
        # # plt.scatter(angles[idxes], dists[idxes])
        # # def my_sin(x, amplitude, phase):
        #     # return np.cos(x - phase) * amplitude
        # # fit = curve_fit(my_sin, angles[idxes], dists[idxes], p0=[2000, 1.2])
        # # plt.plot(np.sort(angles[idxes]), my_sin(np.sort(angles[idxes]), *fit[0]))

        # # plt.plot(np.sort(angles[idxes]), (-np.sort(angles[idxes])*sine_params[0] + sine_r) / sine_params[1])

        # # ransac = linear_model.RANSACRegressor()
        # # ransac.fit(angles[idxes].reshape((-1, 1)), dists[idxes])
        # # plt.plot(np.sort(angles[idxes]), ransac.predict(np.sort(angles[idxes]).reshape((-1, 1))))
        # # plt.show()
        
        # cdst = cv2.resize(cdst, (0, 0), fx = 1, fy = 0.25)
        # cdst2 = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
        # for i in idxes:
        #     cv2.circle(cdst, (int(coords[i, 1]+1), int((coords[i, 0]+1)/4)), 1, (255, 255, 0), -1)
        #     draw_line(cdst2, dists[i], angles[i])
        
        # stratification_mat = np.array([
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [l1, l2, 1]
        # ])
        # stratified = cv2.warpPerspective(edges, stratification_mat, (edges.shape[1], int(min(padded_rect[1]))), flags=cv2.INTER_LINEAR)

        # stratified_lines = transform_houghlines(np.vstack((dists[idxes], angles[idxes])).T, np.linalg.inv(stratification_mat))
        # # stratified_lines[:, 0]

        # # point_set_registration(lines_template[2:, 0], stratified_lines[:, 0])
        # # point_set_registration(lines_template[2:, 0], lines_template[2:, 0] * 1.2 + 100)

        # cdst_stratified = cv2.cvtColor(stratified.copy(), cv2.COLOR_GRAY2BGR)
        # scale = 0.934
        # offset = 267
        # scale * lines_template[2:, 0] + offset
        # for rho in scale * lines_template[2:, 0] + offset:
        #     draw_line(cdst_stratified, rho, np.mean(stratified_lines[:, 1]), color=(255, 255, 0))
        # for line in stratified_lines:
        #     draw_line(cdst_stratified, line[0], line[1])

        # cv2.imshow('cdst_stratified', cdst_stratified)

        # # tested_angles = np.linspace(-np.pi / 4, np.pi / 4, 180, endpoint=False)
        # # h2, theta2, d2 = hough_line((h / np.max(h)).astype('float32'), theta=tested_angles)
        # # h2space, angles2, dists2 = hough_line_peaks(h2, theta2, d2, min_distance=5, min_angle=5, threshold=None)
        # # for _, angle, dist in zip(h2space, angles2, dists2):
        # #     draw_line(cdst, dist, angle)
        # # h2_n = (h2 / np.max(h2)).astype('float32')
        # # h2_n = cv2.resize(h2_n, (0, 0), fx = 1, fy = 0.1)
        # # cdst = cv2.resize(cdst, (0, 0), fx = 1, fy = 0.25)
        # cv2.imshow('houghline', cdst)
        # # cv2.imshow('h2', h2_n)

        # # lines = transform_houghlines(lines_template, homography @ np.linalg.inv(shift_mat @ rot_mat))
        # # idxes = np.argwhere((lines[:, 1] > np.pi/2) | (lines[:, 1] < -np.pi/2))
        # # for idx in idxes:
        # #     lines[idx[0], 0] = -lines[idx[0], 0]
        # #     lines[idx[0], 1] = lines[idx[0], 1] - np.sign(lines[idx[0], 1]) * np.pi
        
        # # coords, deltax, deltay = coords_in_hough(lines, theta, d)
        # # assert(np.all(coords > 0))
        # # h2 = interpolate_from_coords(coords, deltax, deltay, h)

        # cv2.imshow('rotated_edges', cdst2)
        counter = counter + 1
        cv2.waitKey()