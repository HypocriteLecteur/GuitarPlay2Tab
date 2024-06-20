import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from itertools import combinations
from utils.utils import angle_diff

fret_length_scaling = 2**(1/12) / (2**(1/12) - 1)

line_template = np.array([   0.,   14.,   44.,   71.,   94.,  130.,  157.,  190.,  226.,
                          269.,  310.,  352.,  385.,  397.,  445.,  496.,  550.,  607.,
                          651.,  662.,  726.,  783.,  799.,  871.,  947., 1023.])
hspace_template = np.array([89, 134, 125, 124, 103, 100, 120, 110, 97, 111, 81, 114, 
                            86, 85, 106, 110, 111, 107, 78, 94, 75, 71, 99, 99, 97, 90])

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

def draw_line(image, rho, theta, color=(0,0,255)):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
    pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
    cv2.line(image, pt1, pt2, color, 1, cv2.LINE_AA)

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
    w = frame.shape[1]
    r1 = fretboard_linepair[1]
    t1 = fretboard_linepair[2]
    r2 = fretboard_linepair[3]
    t2 = fretboard_linepair[4]
    cornerpoints = np.array([[
        [0, int(r1/np.sin(t1))],
        [w, int((r1-w*np.cos(t1))/np.sin(t1))],
        [0, int(r2/np.sin(t2))],
        [w, int((r2-w*np.cos(t2))/np.sin(t2))]
    ]])

    rect = cv2.minAreaRect(cornerpoints) # (x-coordinate, y-coordinate),(width, height), rotation)
    if rect[1][1] > rect[1][0]:
        rotated, rot_mat = rotate_image(frame, rect[0], -(90-rect[2]))
        rotated = rotated[max(int(rect[0][1]-0.6*rect[1][0]), 0):int(rect[0][1]+0.6*rect[1][0]), :]
        # if int(rect[0][1]-0.6*rect[1][0]) < 0:
            # tmp = np.array([rect[0][0], rect[0][1]]) + rot_mat [:, :2].T @ np.array([-0.5*rect[1][1], -rect[0][1]])
        # else:
            # tmp = np.array([rect[0][0], rect[0][1]]) + rot_mat [:, :2].T @ np.array([-0.5*rect[1][1], -0.6*rect[1][0]])
        # filtered_parallel_lines[0][1] = abs(np.cos(t1) * tmp[0] + np.sin(t1) * tmp[1] - r1)
        # filtered_parallel_lines[0][2] = filtered_parallel_lines[0][2] +(90-rect[2])/180*np.pi
        # filtered_parallel_lines[0][3] = abs(np.cos(t2) * tmp[0] + np.sin(t2) * tmp[1] - r2)
        # filtered_parallel_lines[0][4] = filtered_parallel_lines[0][4] +(90-rect[2])/180*np.pi
    else:
        rotated, rot_mat = rotate_image(frame, rect[0], rect[2])
        rotated = rotated[max(int(rect[0][1]-0.6*rect[1][1]), 0):int(rect[0][1]+0.6*rect[1][1]), :]
        # tmp = np.array([rect[0][0], rect[0][1]]) + rot_mat [:, :2].T @ np.array([-0.5*rect[1][0], -0.6*rect[1][1], 0])
        # filtered_parallel_lines[0][1] = abs(np.cos(t1) * tmp[0] + np.sin(t1) * tmp[1] - r1)
        # filtered_parallel_lines[0][2] = filtered_parallel_lines[0][2] - rect[2]/180*np.pi
        # filtered_parallel_lines[0][3] = abs(np.cos(t2) * tmp[0] + np.sin(t2) * tmp[1] - r2)
        # filtered_parallel_lines[0][4] = filtered_parallel_lines[0][4] - rect[2]/180*np.pi
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

def locate_fretboard(rotated_frame, rotated_edges, threshold=0.6):
    filtered_parallel_lines, cdst = parallel_lines_detection(rotated_edges, theta_tol=5, rho_tol=100, is_visualize=True)
    # cdst2 = cv2.cvtColor(rotated_edges.copy(), cv2.COLOR_GRAY2BGR)
    # cv2.imshow('cdst', cdst)        

    h, theta, d = hough_line(rotated_edges, theta=np.array([0.0]))
    hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10)

    hspace, angles, dists = (list(item) for item in zip(*sorted(zip(hspace, angles, dists), key=lambda x: x[2])))

    rois = []
    for ri, rj in zip(dists[0:-1], dists[1:]):
        point1 = line_line_intersection_fast(filtered_parallel_lines[0][1:3], rj)
        point2 = line_line_intersection_fast(filtered_parallel_lines[0][3:], rj)

        rect = [int(ri), int(point1[1]), int(rj)-int(ri), int(point2[1])-int(point1[1])] # [x y w h]
        rois.append(rect)
        # cv2.imshow('debug', rotated_frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
        # cv2.waitKey()
    rois = np.array(rois)
    second_order_difference = np.abs(np.diff(np.diff(rois[:, 2])))
    mask = second_order_difference / rois[1:-1, 2] <= threshold
    idxes = np.argwhere(mask).squeeze() + 1

    average_intensity = np.zeros((rois.shape[0]))
    for i in range(rois.shape[0]):
        rect = rois[i, :]
        average_intensity[i] = np.sum(rotated_frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]) / rect[2] / rect[3]

    diff_average_intensity = np.diff(average_intensity[idxes]) / average_intensity[idxes][:-1]
    print(diff_average_intensity)
    average_intensity_idxes = idxes[np.argwhere(diff_average_intensity>=0.1).squeeze()+1]
    for i in average_intensity_idxes:
        draw_line(cdst, rois[i, 0], 0)
        cv2.putText(cdst, f'{i}', (rois[i, 0], rois[i, 1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
    for dist in rois[np.argwhere(np.invert(mask)).squeeze() + 1, 0]:
        draw_line(cdst, dist, 0, color=(0, 255, 0))
    cv2.imshow('cdst', cdst)    

if __name__ == '__main__':
    frame_number = 500
    cap = cv2.VideoCapture('test\\guitar2.mp4')

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

        rotated_frame, _ = rotate_fretboard(frame, filtered_parallel_lines)

        rotated_frame = cv2.equalizeHist(rotated_frame)

        rotated_edges = cv2.Canny(rotated_frame, 50, 150, 5)

        # Second stage: fret detection
        homography, cdst, _ = find_undistort_transform(rotated_edges, is_visualize=False)
        # cv2.imshow('fret detection', cdst)

        # # test = rotated_frame.copy()
        # test = cv2.warpPerspective(rotated_frame.copy(), homography, rotated_frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        # # test = cv2.copyMakeBorder(test, 200, 200, 200, 200, cv2.BORDER_CONSTANT)
        
        # template = cv2.imread('test\\template.jpg')
        # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # # template = cv2.resize(template, (0, 0), fx = 1, fy = 1)
        # # cv2.imshow('match', res / np.max(res))
        
        rotated_frame = cv2.warpPerspective(rotated_frame, homography, rotated_frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        rotated_edges = cv2.warpPerspective(rotated_edges, homography, rotated_edges.shape[1::-1], flags=cv2.INTER_LINEAR)
        locate_fretboard(rotated_frame, rotated_edges)

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
