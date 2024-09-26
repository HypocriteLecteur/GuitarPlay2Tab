import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from itertools import combinations
from utils.utils import angle_diff
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation
from scipy.optimize import minimize, least_squares
from template import Template, transform_houghlines, shift_center_houghlines, rotate_line_pair, transform_points
from itertools import combinations

fret_length_scaling = 2**(1/12) / (2**(1/12) - 1)

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
            idx_high+4, idx_high-3,
            idx_12+1, idx_12,
            idx_low+4, idx_low-3
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
    cv2.waitKey()

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

# def match_region(frame, template, homography, is_visualize=False):
#     bb = boundingbox_12(template, homography, padding=0.5)
#     if is_visualize:
#         cdst = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
#         for point in cornerpoints:
#             cv2.circle(cdst, (int(point[0]), int(point[1])), 3, (255, 255, 255))
#         cv2.imshow('cdst', cdst)
#         cv2.waitKey()

#     third_shift_mat = np.eye(3)
#     third_shift_mat[0, 2] = -bb[0]
#     third_shift_mat[1, 2] = -bb[1]

#     fourth_shift_mat = np.eye(3)
#     fourth_shift_mat[0, 2] = template.template_fretlines[3, 0]
#     fourth_shift_mat[1, 2] = 0

#     debug = frame[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
#     debug2 = cv2.warpPerspective(template.template_image_edge[:, int(template.template_fretlines[3, 0]):int(template.template_fretlines[2, 0])], 
#                                  third_shift_mat @ np.linalg.inv(homography) @ fourth_shift_mat, (bb[2], bb[3]), flags=cv2.INTER_LINEAR)

#     error = np.mean(np.abs(
#         debug - 
#         debug2))
    
#     if is_visualize:
#         cv2.imshow('debug', debug)
#         cv2.imshow('debug2', debug2)
#         cv2.waitKey()
#     return error

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
    # quad1_feature = quad1 - np.mean(quad1, axis=0)
    # quad2_feature = quad2 - np.mean(quad2, axis=0)
    quad1_feature = quad1
    quad2_feature = quad2
    return np.sum(np.sum((quad1_feature - quad2_feature)**2))

def draw_quadrangle(cdst, quad, color=(0, 0, 255)):
    for point in quad:
        cv2.circle(cdst, (int(point[0]), int(point[1])), 3, color)
    for idx in ((0, 1), (1, 2), (2, 3), (3, 0)):
        cv2.line(cdst, quad[idx[0], :].astype('int'), quad[idx[1], :].astype('int'), color, 1)

def detect_lines_near_angles(lines_rt, theta_tol_deg=5):
    '''
    lines_rt = np.array([[r1, t1], [r2, t2], ...])
    '''
    tested_angles = np.array(
        [np.linspace(line[1]-theta_tol_deg/180*np.pi, line[1]+theta_tol_deg/180*np.pi, 2*theta_tol_deg+1, endpoint=False) for line in lines_rt]
        ).flatten()
    tested_angles = np.unique(tested_angles)
    h, theta, d = hough_line(edges, theta=tested_angles)
    hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=10, min_angle=10, threshold=200)
    return hspace, angles, dists

def jac(R, t, target_image, template):
    H = R + t.reshape((3, 1)) @ n[0].T
    bb, _, _ = boundingbox_12(template, H, padding=0)

    It = template.template_image[:, int(template.template_fretlines[3, 0]):int(template.template_fretlines[2, 0])]
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

def delta_p(rvec, t, target_image, template):
    R = cv2.Rodrigues(rvec)[0]
    H = R + t.reshape((3, 1)) @ n[0].T
    J = jac(R, t, target_image, template)

    It = template.template_image[:, int(template.template_fretlines[3, 0]):int(template.template_fretlines[2, 0])]
    mask = It != 0
    It = It.T[mask.T]

    bb, _, _ = boundingbox_12(template, H, padding=0)
    third_shift_mat = np.eye(3)
    third_shift_mat[0, 2] = bb[0]
    third_shift_mat[1, 2] = bb[1]

    fourth_shift_mat = np.eye(3)
    fourth_shift_mat[0, 2] = -template.template_fretlines[3, 0]
    fourth_shift_mat[1, 2] = 0

    debug = target_image[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
    debug = cv2.warpPerspective(debug, fourth_shift_mat @ H @ third_shift_mat, (int(template.template_fretlines[2, 0])-int(template.template_fretlines[3, 0]), template.template_image.shape[0]), flags=cv2.INTER_LINEAR)    
    Ic = debug.T[mask.T]

    grad = np.mean((It - Ic).reshape((-1,1)) * J, axis=0) * 2
    step = np.linalg.inv(J.T @ J) @ J.T @ (It - Ic) / 255
    return step, grad

if __name__ == '__main__':
    cap = cv2.VideoCapture('test\\guitar2.mp4')

    # bounding box
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_edge = cv2.Canny(frame, 30, 100, 5)

    flag, template = guitar_detection(frame)

    prev_frame = frame.copy()
    mask = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3)).astype('uint8')
    mask[..., 1] = 255

    homography = template.total_mat
    counter = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(frame, 30, 100, 5)

        # flow = cv2.calcOpticalFlowFarneback(prev_frame, frame,  
        #                             None, 
        #                             0.5, 3, 15, 3, 5, 1.2, 0)
        # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow  
        # direction 
        # mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow 
        # magnitude (normalized) 
        # mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        
        # Converts HSV to RGB (BGR) color representation 
        # rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR) 


        bb, template_center, cornerpoints = boundingbox_12(template, homography, padding=0.5)
        bb2, _, _ = boundingbox_12(template, homography, padding=0)


        # cv2.imshow("dense optical flow", rgb[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]])

        target_image = frame

        bb_shift = np.eye(3)
        bb_shift[0, 2] = -bb[0]
        bb_shift[1, 2] = -bb[1]

        template_shift = np.eye(3)
        template_shift[0, 2] = template.template_fretlines[3, 0]
        template_shift[1, 2] = 0

        scale_min = 0.99
        scale_max = 1.01
        scale_step = 0.005
        angle_min = -3
        angle_max = 3
        angle_step = 1
        translation_min = -15
        translation_max = 15
        translation_step = 1
        test_transforms = np.mgrid[
            scale_min:scale_max:scale_step,
            angle_min:angle_max:angle_step, 
            translation_min:translation_max:translation_step, 
            translation_min:translation_max:translation_step
            ]

        ncc_matrix = np.zeros((test_transforms.shape[1], test_transforms.shape[2], test_transforms.shape[3], test_transforms.shape[4]))
        for i in range(test_transforms.shape[1]):
            for j in range(test_transforms.shape[2]):
                scale = test_transforms[0, i, 0, 0, 0]
                angle = test_transforms[1, 0, j, 0, 0]

                scale_mat = np.array([
                    [scale, 0, (scale-1)*template_center[0]],
                    [0, scale, (scale-1)*template_center[1]],
                    [0, 0, 1]
                ])

                rot_mat = cv2.getRotationMatrix2D(template_center, angle, 1.0)
                rot_mat = np.vstack((rot_mat, np.array([0, 0, 1])))
                # np.linalg.norm(transform_points(cornerpoints, rot_mat) - cornerpoints, axis=1)

                cornerpoints_ = transform_points(cornerpoints, rot_mat @ scale_mat)
                bb2 = cv2.boundingRect(cornerpoints_.astype('float32'))
                bb2_shift = np.eye(3)
                bb2_shift[0, 2] = -bb2[0]
                bb2_shift[1, 2] = -bb2[1]

                template_image = cv2.warpPerspective(template.template_image[:, int(template.template_fretlines[3, 0]):int(template.template_fretlines[2, 0])], 
                                            bb2_shift @ rot_mat @ scale_mat @ np.linalg.inv(homography) @ template_shift, (bb2[2], bb2[3]), flags=cv2.INTER_LINEAR).astype('float32')
                mask = template_image != 0
                template_image[mask] = (template_image[mask] - np.mean(template_image[mask])) / np.std(template_image[mask])

                for k in range(test_transforms.shape[2]):
                    for l in range(test_transforms.shape[3]):
                        tx, ty = test_transforms[2:, i, j, k, l]

                        target_image_ = target_image[bb2[1]+int(ty):bb2[1]+int(ty)+bb2[3], bb2[0]+int(tx):bb2[0]+int(tx)+bb2[2]]

                        ncc_matrix[i, j, k, l] = np.sum((target_image_[mask] - np.mean(target_image_[mask])) * template_image[mask]) / np.std(target_image_[mask])

        idx = np.unravel_index(np.argmax(ncc_matrix, axis=None), ncc_matrix.shape)
        print(test_transforms[:, *idx])
        scale_, angle_, tx_, ty_ = test_transforms[:, *idx]

        scale_mat_ = np.array([
                [scale_, 0, (scale_-1)*template_center[0]],
                [0, scale_, (scale_-1)*template_center[1]],
                [0, 0, 1]
            ])
        rot_mat_ = cv2.getRotationMatrix2D(template_center, angle_, 1.0)
        rot_mat_ = np.vstack((rot_mat_, np.array([0, 0, 1])))
        shift_mat_ = np.array([
            [1, 0, tx_],
            [0, 1, ty_],
            [0, 0, 1]
        ])

        homography = np.linalg.inv(shift_mat_ @ rot_mat_ @ scale_mat_ @ np.linalg.inv(homography))

        _, _, cornerpoints = boundingbox_12(template, homography, padding=0)
        cdst = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        draw_quadrangle(cdst, cornerpoints)
        cv2.imshow('coarse tracking result', cdst)

        _, rot, t, n = cv2.decomposeHomographyMat(homography, np.eye(3))
        # theta_zc, theta_x, theta_zt = Rotation.from_matrix(rot[0]).as_euler('zxz', degrees=True)

        def fun(rvec, tvec, target_image, template, n):
            R = cv2.Rodrigues(rvec)[0]
            H = R + tvec.reshape((3, 1)) @ n[0].T

            bb, _, _ = boundingbox_12(template, H, padding=0)
            
            third_shift_mat = np.eye(3)
            third_shift_mat[0, 2] = bb[0]
            third_shift_mat[1, 2] = bb[1]

            fourth_shift_mat = np.eye(3)
            fourth_shift_mat[0, 2] = -template.template_fretlines[3, 0]
            fourth_shift_mat[1, 2] = 0

            debug = target_image[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
            debug = cv2.warpPerspective(debug, fourth_shift_mat @ H @ third_shift_mat, (int(template.template_fretlines[2, 0])-int(template.template_fretlines[3, 0]), template.template_image.shape[0]), flags=cv2.INTER_LINEAR)

            template_image = template.template_image[:, int(template.template_fretlines[3, 0]):int(template.template_fretlines[2, 0])]
            mask = template_image != 0
            error = np.mean((
                debug[mask] - 
                template_image[mask])**2)
            return error

        # sol = least_squares(fun, (theta_zc, theta_x, theta_zt, t[0][0, 0], t[0][1, 0], t[0][2, 0]), args=(target_image, template, n))

        norm_tol = 1e-3
        alpha = 0.5
        c = 1e-4
        norm = 1e-4
        rot_vec = cv2.Rodrigues(rot[0])[0]
        t_vec = t[0]
        while norm > norm_tol:
            deltap, grad = delta_p(rot_vec, t_vec, target_image, template)
            while fun(rot_vec+deltap[:3].reshape((3, 1)), t_vec+deltap[3:].reshape((3, 1)), target_image, template, n) > fun(rot_vec, t_vec, target_image, template, n) + c * grad @ deltap:
                deltap = deltap * alpha

            # cdst = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
            # homography = cv2.Rodrigues(rot_vec)[0] + t_vec.reshape((3, 1)) @ n[0].T
            # _, _, cornerpoints = boundingbox_12(template, homography, padding=0)
            # draw_quadrangle(cdst, cornerpoints)
            
            rot_vec = rot_vec+deltap[:3].reshape((3, 1))
            t_vec = t_vec+deltap[3:].reshape((3, 1))
            norm = np.linalg.norm(deltap)

            # homography = cv2.Rodrigues(rot_vec)[0] + t_vec.reshape((3, 1)) @ n[0].T
            # _, _, cornerpoints = boundingbox_12(template, homography, padding=0)
            # draw_quadrangle(cdst, cornerpoints, color=(0, 255, 0))

            # cv2.imshow('debug ', cdst)
            # cv2.waitKey()

        # R = Rotation.from_euler('zxz', [sol.x[0], sol.x[1], sol.x[2]], degrees=True).as_matrix()
        homography = cv2.Rodrigues(rot_vec)[0] + t_vec.reshape((3, 1)) @ n[0].T
        # pose_refine(homography, target_image, template)
        # homography = rot[0] + t[0] @ n[0].T

        _, _, cornerpoints = boundingbox_12(template, homography, padding=0)
        cdst = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        draw_quadrangle(cdst, cornerpoints)
        cv2.imshow('tracking result', cdst)

        cv2.waitKey()
        counter = counter + 1
        prev_frame = frame.copy()