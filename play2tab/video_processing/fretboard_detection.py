import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from itertools import combinations
from utils.utils import angle_diff
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering

fret_length_scaling = 2**(1/12) / (2**(1/12) - 1)

def line_line_intersection(riti, rjtj):
    return np.array([
        [np.cos(riti[1]), np.cos(riti[1])],
        [np.cos(rjtj[1]), np.sin(rjtj[1])]
        ]) @ np.array([riti[0], rjtj[0]])

def line_line_intersection_fast(riti, rj):
    return np.linalg.inv(np.array([
        [np.cos(riti[1]), np.sin(riti[1])],
        [1, 0]
        ])) @ np.array([riti[0], rj])

def rotate_image(image, image_center, angle):
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  # deg
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result, rot_mat

def rotate_line_pair(angle, rot_mat, line_pair):
    '''
    angle: rad
    rot_mat: the 2 * 3 transformation
    line_pair: [h, r1, t1, r2, t2]
    '''
    m = -rot_mat [:, :2].T @ rot_mat [:, 2]
    line_pair[1] = -(np.cos(line_pair[2]) * m[0] + np.sin(line_pair[2]) * m[1] - line_pair[1])
    line_pair[3] = -(np.cos(line_pair[4]) * m[0] + np.sin(line_pair[4]) * m[1] - line_pair[3])

    line_pair[2] = line_pair[2] - angle
    line_pair[4] = line_pair[4] - angle
    return line_pair

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

def cornerpoints_from_line_pair(linepair, width):
    w = width
    r1 = linepair[1]
    t1 = linepair[2]
    r2 = linepair[3]
    t2 = linepair[4]
    cornerpoints = np.array([[
        [0, int(r1/np.sin(t1))],
        [w, int((r1-w*np.cos(t1))/np.sin(t1))],
        [w, int((r2-w*np.cos(t2))/np.sin(t2))],
        [0, int(r2/np.sin(t2))]
    ]])
    return cornerpoints

def parallel_lines_detection(edges, theta_tol=5, rho_tol=100, is_visualize=False):
    '''
    filtered_parallel_lines = [h, r1, t1, r2, t2]
    '''
    theta_tol = 5  # degree
    rho_tol = 100  # pixel
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(edges, theta=tested_angles)
    hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10, threshold=100)
    parallel_lines = []
    for [[hi, ri, ti], [hj, rj, tj]] in combinations(zip(hspace, dists, angles), 2):
        if angle_diff(ti, tj) < theta_tol / 180 * np.pi and abs(ri - rj) > rho_tol:
            parallel_lines.append([hi+hj, ri, ti, rj, tj])
    parallel_lines = np.array(parallel_lines)

    filtered_parallel_lines = parallel_lines[parallel_lines[:, 0] >= 1*np.max(parallel_lines[:, 0]), :]

    if is_visualize:
        cdst = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
        for line_pair in filtered_parallel_lines:
            draw_line(cdst, line_pair[1], line_pair[2])
            draw_line(cdst, line_pair[3], line_pair[4])
        return filtered_parallel_lines, cdst
    return filtered_parallel_lines, None

def rotate_fretboard(frame, filtered_parallel_lines):
    fretboard_linepair = filtered_parallel_lines[0]
    cornerpoints = cornerpoints_from_line_pair(fretboard_linepair, frame.shape[1])

    rect = cv2.minAreaRect(cornerpoints) # (x-coordinate, y-coordinate),(width, height), rotation)
    if rect[1][1] > rect[1][0]:
        angle = -(90-rect[2])
        rotated, rot_mat = rotate_image(frame, rect[0], angle)
        filtered_parallel_lines[0] = rotate_line_pair(angle/180*np.pi, rot_mat, fretboard_linepair)

        y0 = max(int(rect[0][1]-0.6*rect[1][0]), 0)
        rotated = rotated[y0:int(rect[0][1]+0.6*rect[1][0]), :]
    else:
        angle = rect[2]
        rotated, rot_mat = rotate_image(frame, rect[0], angle)
        filtered_parallel_lines[0] = rotate_line_pair(angle/180*np.pi, rot_mat, fretboard_linepair)

        y0 = max(int(rect[0][1]-0.6*rect[1][1]), 0)
        rotated = rotated[y0:int(rect[0][1]+0.6*rect[1][1]), :]
    filtered_parallel_lines[0][1] = filtered_parallel_lines[0][1] - y0 * np.cos(np.pi/2 - filtered_parallel_lines[0][2])
    filtered_parallel_lines[0][3] = filtered_parallel_lines[0][3] - y0 * np.cos(np.pi/2 - filtered_parallel_lines[0][4])
    return rotated, filtered_parallel_lines

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

def locate_fretboard(rotated_frame, rotated_edges, filtered_parallel_lines):
    # cdst = cv2.cvtColor(rotated_edges.copy(), cv2.COLOR_GRAY2BGR)
    # draw_line(cdst, filtered_parallel_lines[0][1], filtered_parallel_lines[0][2])
    # draw_line(cdst, filtered_parallel_lines[0][3], filtered_parallel_lines[0][4])

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

    # cv2.namedWindow("cdst")
    # def mouse_callback(event, x, y, flags, param):
    #     idx = np.argmin(np.abs(np.array(dists) - x))
    #     cdst2 = cdst.copy()
    #     if abs(dists[idx] - x ) < 5:
    #         draw_line(cdst2, dists[idx], 0, thickness=3)
    #         cv2.putText(cdst2, f'{hspace[idx]}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
    #     cv2.imshow('cdst', cdst2)
    # cv2.setMouseCallback("cdst", mouse_callback)

    rois = []
    features = np.zeros((dists.shape[0]-1, 1))
    for i, (ri, rj) in enumerate(zip(dists[0:-1], dists[1:])):
        point1 = line_line_intersection_fast(filtered_parallel_lines[0][1:3], rj)
        point2 = line_line_intersection_fast(filtered_parallel_lines[0][3:], rj)

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

    kernel_12 = np.array([-1, -1, 1, -1, -1])
    idx_12 = np.argmin(np.convolve(second_order_difference, kernel_12, 'valid')) + 2 + 1

    kernel_quadruplets = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1])
    if len(second_order_difference) < 2 * len(kernel_quadruplets):
        idx_quadruplets = np.array([idx_12, idx_12])
    else:
        convolve = np.convolve(second_order_difference, kernel_quadruplets, 'valid')
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
        fret_idx = np.array([
            idx_high+3, idx_high+1, idx_high-1, idx_high-3,
            idx_12,
            idx_low+3, idx_low+1, idx_low-1, idx_low-3
        ])
        # for idx in fret_idx:
        #     cv2.rectangle(cdst, (int(dists[idx]), 0), (int(dists[idx+1]), 20), (0, 0, 255), -1)
        # cv2.imshow('cdst', cdst)
    else:
        fret_idx = None
        # cv2.rectangle(cdst, (int(dists[idx_12]), 0), (int(dists[idx_12])+20, 20), (255, 255, 255), -1)
        # if idx_quadruplets[0] <=len(dists) and idx_quadruplets[1] <=len(dists):
        #     cv2.rectangle(cdst, (int(dists[idx_quadruplets[0]]), 0), (int(dists[idx_quadruplets[0]])+20, 20), (255, 255, 255), -1)
        #     cv2.rectangle(cdst, (int(dists[idx_quadruplets[1]]), 0), (int(dists[idx_quadruplets[1]])+20, 20), (255, 255, 255), -1)

    # template_1 = cv2.cvtColor(cv2.imread('test\\template_1.jpg'), cv2.COLOR_BGR2GRAY)
    # feature1 = cv2.calcHist([template_1], [0], None, [256], [0,255])
    # cv2.normalize(feature1, feature1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # template_2 = cv2.cvtColor(cv2.imread('test\\template_2.jpg'), cv2.COLOR_BGR2GRAY)
    # feature2 = cv2.calcHist([template_2], [0], None, [256], [0,255])
    # cv2.normalize(feature2, feature2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # template_n1 = cv2.cvtColor(cv2.imread('test\\template_n1.jpg'), cv2.COLOR_BGR2GRAY)
    # featuren1 = cv2.calcHist([template_n1], [0], None, [256], [0,255])
    # cv2.normalize(featuren1, featuren1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # template_n2 = cv2.cvtColor(cv2.imread('test\\template_n2.jpg'), cv2.COLOR_BGR2GRAY)
    # featuren2 = cv2.calcHist([template_n2], [0], None, [256], [0,255])
    # cv2.normalize(featuren2, featuren2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # for i in range(len(dists)-1):
    #     rect = rois[i]
        # cv2.putText(cdst, f'{np.mean([rotated_frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]]):.0f}', (int(dists[i]), 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        # cv2.putText(cdst, f'{np.std([rotated_frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]]):.0f}', (int(dists[i]), 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        # cv2.normalize(features[i, :], features[i, :], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # metric_1 = cv2.compareHist(feature1.reshape((-1,)), features[i, :].astype('float32'), cv2.HISTCMP_CORREL)
        # metric_2 = cv2.compareHist(feature2.reshape((-1,)), features[i, :].astype('float32'), cv2.HISTCMP_CORREL)
        # metric_n1 = cv2.compareHist(featuren1.reshape((-1,)), features[i, :].astype('float32'), cv2.HISTCMP_CORREL)
        # metric_n2 = cv2.compareHist(featuren2.reshape((-1,)), features[i, :].astype('float32'), cv2.HISTCMP_CORREL)
        # plt.figure()
        # plt.plot(feature1)
        # plt.plot(feature2)
        # plt.plot(featuren1)
        # plt.plot(featuren2)
        # plt.plot(features[i, :])
        # plt.legend(['feature1', 'feature2', 'featuren1', 'featuren2', 'real'])
        # plt.show()
        # if max(metric_1, metric_2) > max(metric_n1, metric_n2):
        #     cv2.rectangle(cdst, (int(dists[i]), 0), (int(dists[i])+20, 20), (255, 0, 0), -1) 
        # else:
        #     cv2.rectangle(cdst, (int(dists[i]), 0), (int(dists[i])+20, 20), (0, 0, 255), -1) 

    # distance_matrix = squareform(pdist(features))
    # affinity_matrix = 1 - distance_matrix / np.max(distance_matrix)
    # rois = np.array(rois)
    # clustering = SpectralClustering(n_clusters=3,
    #     assign_labels='discretize',
    #     random_state=0,
    #     affinity='precomputed').fit(affinity_matrix)
    # for i in range(len(clustering.labels_)):
    #     if clustering.labels_[i]:
    #         cv2.rectangle(cdst, (int(dists[i]), 0), (int(dists[i])+20, 20), (255, 0, 0), -1) 
    #     else:
    #         cv2.rectangle(cdst, (int(dists[i]), 0), (int(dists[i])+20, 20), (0, 0, 255), -1) 
    # cv2.imshow('cdst', cdst)

    if is_located:
        return rotated_frame[:, int(dists[fret_idx[-1]-1]):int(dists[fret_idx[0]+2])], dists[fret_idx]
    else:
        return None, None

if __name__ == '__main__':
    cap = cv2.VideoCapture('test\\guitar2.mp4')

    template = None
    counter = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First stage: guitar direction detection
        lower_threshold = 30
        high_threshold = 100
        aperture_size = 5
        is_L2gradient = True
        edges = cv2.Canny(frame, 30, 100, 5)
        # edges = cv2.Sobel(frame, cv2.CV_8U, 1, 0, 3)

        filtered_parallel_lines, cdst = parallel_lines_detection(edges, theta_tol=5, rho_tol=100, is_visualize=False)

        rotated_frame, filtered_parallel_lines = rotate_fretboard(frame, filtered_parallel_lines)

        cornerpoints = cornerpoints_from_line_pair(filtered_parallel_lines[0], rotated_frame.shape[1])
        mask = np.zeros_like(rotated_frame)
        cv2.fillPoly(mask, pts=[cornerpoints], color=(255, 255, 255))
        rotated_frame = cv2.bitwise_and(rotated_frame, mask)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        rotated_frame = clahe.apply(rotated_frame)
        # rotated_frame = cv2.equalizeHist(rotated_frame)

        rotated_edges = cv2.Canny(rotated_frame, 50, 150, 5)
        
        # Second stage: fret detection
        homography, cdst, fret_lines = find_undistort_transform(rotated_edges, is_visualize=False)
        # cv2.imshow('fret detection', cdst)
        
        rotated_frame = cv2.warpPerspective(rotated_frame, homography, rotated_frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        # draw_line(rotated_frame, filtered_parallel_lines[0][1], filtered_parallel_lines[0][2])
        # draw_line(rotated_frame, filtered_parallel_lines[0][3], filtered_parallel_lines[0][4])
        # cv2.imshow('rotated_frame', rotated_frame)

        rotated_edges = cv2.warpPerspective(rotated_edges, homography, rotated_edges.shape[1::-1], flags=cv2.INTER_LINEAR)
        if template is None:
            template, template_fretlines = locate_fretboard(rotated_frame, rotated_edges, filtered_parallel_lines)
        if template is not None:
            cv2.imshow('', template)

        # tol = 0.20
        # dists.sort()
        # dist_diffs = np.diff(dists)
        # fret_length_diffs = dist_diffs[:-1:] / dist_diffs[1::]
        # dists_idxes = np.argwhere(abs(fret_length_diffs - np.median(fret_length_diffs)) <= tol)[:, 0]
        # dists_idxes = np.unique(np.concatenate((dists_idxes, dists_idxes+1, dists_idxes+2)))
        # for dist in dists[dists_idxes]:
        #     draw_line(cdst3, dist, 0)
        
        # cv2.imshow('direction', cdst3)
        # cv2.imshow('frets', cdst2)
        cv2.waitKey()
        counter = counter + 1
