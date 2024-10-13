import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from itertools import combinations
from utils.utils import angle_diff
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation
from template import Template, transform_houghlines, shift_center_houghlines, rotate_line_pair, transform_points
from itertools import combinations

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
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
    pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
    if 'thickness' in kwargs:
        thickness = kwargs['thickness']
    else:
        thickness = 1        
    cv2.line(image, pt1, pt2, (0, 255, 0), thickness, cv2.LINE_AA)

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
    theta_tol = 5  # degree
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


if __name__ == '__main__':
    a = np.array([
        [906, 277],
        [941, 276],
        [938, 329],
        [906, 333]
    ]).astype('float32')
    b = np.array([
        [902, 275],
        [934, 272],
        [934, 325],
        [901, 328]
    ]).astype('float32')
    c = np.array([
        [658, 73, 0],
        [705, 75, 0],
        [705, 148, 0],
        [658, 148, 0]
    ]).astype('float32')
    retval, rvec, tvec = cv2.solvePnP(c, a, 
            np.eye(3), None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
    h1pnp = np.hstack((cv2.Rodrigues(rvec)[0][:, :2], tvec))
    npprint(rvec, tvec)
    retval, rvec2, tvec2 = cv2.solvePnP(c, b, 
            np.eye(3), None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    h2pnp = np.hstack((cv2.Rodrigues(rvec2)[0][:, :2], tvec2))
    npprint(rvec2, tvec2)
    npprint(h1pnp / h1pnp[2, 2])
    npprint(h2pnp / h2pnp[2, 2])
    # npprint(h1)
    # npprint(h2)
    # npprint(np.linalg.inv(h1) @ h2)
    # npprint(h1prime)
    # npprint(h2prime)
    npprint(np.linalg.inv(h1pnp) @ h2pnp)

    cap = cv2.VideoCapture('test\\guitar2.mp4')

    # bounding box
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_edge = cv2.Canny(frame, 30, 100, 5)

    flag, template = guitar_detection(frame)

    fret_num = [3, 5, 7, 9, 12, 15, 17, 19, 21]
    template_quad = {}
    for i in range(9):
        template_quad[fret_num[i]] = np.array([
            line_line_intersection(template.template_fingerboard_lines[0], template.template_fretlines[2*i+1]),
            line_line_intersection(template.template_fingerboard_lines[1], template.template_fretlines[2*i+1]),
            line_line_intersection(template.template_fingerboard_lines[1], template.template_fretlines[2*i]),
            line_line_intersection(template.template_fingerboard_lines[0], template.template_fretlines[2*i])
        ])
    
    prev_quad_frets = [3, 5, 7, 9, 12, 15, 17, 19, 21]
    prev_quad_lines = []
    for i in range(9):
        prev_quad_lines.append([
            template.fingerboard_lines[0],
            template.fingerboard_lines[1],
            template.image_fretlines[2*i],
            template.image_fretlines[2*i+1]
        ])
    prev_quads = [quad_from_quad_line(x) for x in prev_quad_lines]

    F0 = template.total_mat
    A = []
    mask = []
    Vref = []
    for i in range(9):
        xmin = int(template.template_fretlines[2*i+1, 0])
        xmax = int(template.template_fretlines[2*i, 0])
        A_, mask_, Vref_, _, _ = training(template, xmin, xmax, F0, frame)
        A.append(A_)
        mask.append(mask_)
        Vref.append(Vref_)
    
    homography = []
    for i in range(9):
        homography.append(template.total_mat)
    # np.save('mask.npy', mask)
    # np.save('Vref.npy', Vref)
    # np.save('A.npy', A)
    # A = np.load('A.npy')
    # Vref = np.load('Vref.npy')
    # mask = np.load('mask.npy')

    counter = 1
    while True:
        # if counter % 10 == 0:
        #     A = []
        #     mask = []
        #     Vref = []
        #     for i in range(9):
        #         xmin = int(template.template_fretlines[2*i+1, 0])
        #         xmax = int(template.template_fretlines[2*i, 0])
        #         A_, mask_, Vref_, _, _ = training(template, xmin, xmax, homography[i], frame)
        #         A.append(A_)
        #         mask.append(mask_)
        #         Vref.append(Vref_)

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        quad = []
        homography_ = []
        for i in range(9):
            xmin = int(template.template_fretlines[2*i+1, 0])
            xmax = int(template.template_fretlines[2*i, 0])
            
            H_, deltaH = update_homography(xmin, xmax, template, frame, homography[i], Vref[i], mask[i], A[i])
            H_, deltaH = update_homography(xmin, xmax, template, frame, H_, Vref[i], mask[i], A[i])

            # H_ = refine_pose(H_)
            # H_ = refine_pose(homography[i], frame, xmin, xmax, template)
            homography_.append(H_)
            quad.append(transform_points(template_quad[fret_num[i]], np.linalg.inv(H_)))
        
        h_, _ = cv2.findHomography(np.array(quad).reshape((-1, 2)), 
                                           np.array([template_quad[fret] for fret in fret_num]).reshape((-1, 2)), cv2.RANSAC,5.0)        
        
        cdst_verified = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        for i in range(9):
            fretboard = np.array([
                line_line_intersection(template.template_fingerboard_lines[0], template.template_fretlines[2*i+1]),
                line_line_intersection(template.template_fingerboard_lines[1], template.template_fretlines[2*i+1]),
                line_line_intersection(template.template_fingerboard_lines[1], template.template_fretlines[2*i]),
                line_line_intersection(template.template_fingerboard_lines[0], template.template_fretlines[2*i])
            ])
            fretboard2 = transform_points(fretboard, np.linalg.inv(homography[i]))
            draw_quadrangle(cdst_verified, fretboard2, color=(255, 255, 0))
            fretboard2 = transform_points(fretboard, np.linalg.inv(homography_[i]))
            draw_quadrangle(cdst_verified, fretboard2, color=(0, 0, 255))
        
        for i in range(9):
            # homography[i] = homography_[i]
            homography[i] = h_
        counter = counter + 1

        # # prev_quad_frets = tracked_frets
        # # prev_quads = tracked_quads
        # not_tracked_frets = list(set(prev_quad_frets) - set(tracked_frets))
        # not_tracked_frets.sort()
        # for fret in not_tracked_frets:
        #     fret_idx = prev_quad_frets.index(fret)
        #     tracked_quad_lines.insert(fret_idx, transform_houghlines(np.array([
        #             template.template_fingerboard_lines[0, :2],
        #             template.template_fingerboard_lines[1, :2],
        #             template.template_fretlines[2*fret_idx],
        #             template.template_fretlines[2*fret_idx+1]
        #         ]), homography))
        
        # prev_quad_lines = tracked_quad_lines

        # for quad_line in prev_quad_lines:
        #     draw_quadrangle(cdst_verified, quad_from_quad_line(quad_line), color=(0, 255, 0))
        cv2.imshow('cdst_verified', cdst_verified)
        cv2.waitKey()