import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from itertools import combinations
from utils.utils import angle_diff
from scipy.stats import circmean

fret_length_scaling = 2**(1/12) / (2**(1/12) - 1)

def rotate_image(image, image_center, angle):
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  # deg
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def draw_line(image, rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
    pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
    cv2.line(image, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

if __name__ == '__main__':
    frame_number = 500
    cap = cv2.VideoCapture('test\\guitar.mp4')

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
        edges = cv2.Canny(frame, lower_threshold, high_threshold, aperture_size)
        # edges = cv2.Sobel(frame, cv2.CV_8U, 1, 0, 3)

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

        cdst = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
        for line_pair in filtered_parallel_lines:
            draw_line(cdst, line_pair[1], line_pair[2])
            draw_line(cdst, line_pair[3], line_pair[4])
        
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

        rect = cv2.minAreaRect(cornerpoints)
        if rect[1][1] > rect[1][0]:
            rotated = rotate_image(frame, rect[0], -(90-rect[2]))
            mask = np.zeros_like(frame)
            mask = cv2.rectangle(mask, (0,int(rect[0][1]-0.6*rect[1][0])),(w,int(rect[0][1]+0.6*rect[1][0])), 255, -1)
            rotated = rotated[int(rect[0][1]-0.6*rect[1][0]):int(rect[0][1]+0.6*rect[1][0]), :]
            # rotated = cv2.bitwise_and(rotated, mask)
        else:
            rotated = rotate_image(frame, rect[0], rect[2])
            mask = np.zeros_like(frame)
            mask = cv2.rectangle(mask, (0,int(rect[0][1]-0.6*rect[1][1])),(w,int(rect[0][1]+0.6*rect[1][1])), 255, -1)
            rotated = rotated[int(rect[0][1]-0.6*rect[1][1]):int(rect[0][1]+0.6*rect[1][1]), :]
            # rotated = cv2.bitwise_and(rotated, mask)
        rotated = cv2.equalizeHist(rotated)

        rotated_edges = cv2.Canny(rotated, 50, 150, 5)

        # Second stage: fret detection
        # fret_theta = circmean([filtered_parallel_lines[0, 2], filtered_parallel_lines[0, 4]], low=-np.pi/2, high=np.pi/2)
        # fret_theta = fret_theta + np.pi/2
        # if fret_theta >= np.pi / 2:
        #     fret_theta = fret_theta - np.pi
        
        # theta_range = 10/180*np.pi
        # theta_tol = 5  # degree
        # rho_tol = 10  # pixel
        # tested_angles = np.linspace(fret_theta-theta_range, fret_theta+theta_range, 10, endpoint=False)
        # h, theta, d = hough_line(edges, theta=tested_angles)
        # hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10)
    
        theta_range = 10/180*np.pi
        theta_tol = 5  # degree
        rho_tol = 10  # pixel
        tested_angles = np.linspace(-theta_range, theta_range, 10, endpoint=False)
        h, theta, d = hough_line(rotated_edges, theta=tested_angles)
        hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10)

        # cdst2 = cv2.cvtColor(rotated_edges.copy(), cv2.COLOR_GRAY2BGR)
        # for angle, dist in zip(angles, dists):
        #     draw_line(cdst2, dist, angle)

        # sort by dist
        angles, dists = (list(item) for item in zip(*sorted(zip(angles, dists), key=lambda x: x[1])))
        
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
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        cdst2 = cv2.cvtColor(rotated_edges.copy(), cv2.COLOR_GRAY2BGR)
        for angle, dist in zip(angles, dists):
            draw_line(cdst2, dist, angle)

        cdst2 = cv2.warpPerspective(cdst2, M, rotated_edges.shape[1::-1], flags=cv2.INTER_LINEAR)

        # theta_range = 10/180*np.pi
        # theta_tol = 5  # degree
        # rho_tol = 10  # pixel
        # tested_angles = np.linspace(-theta_range, theta_range, 10, endpoint=False)
        # h, theta, d = hough_line(rotated_edges, theta=tested_angles)
        # hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10)
        
        
        # cv2.imshow('rotated', result)
        # cv2.waitKey(1)
        
        # cv2.imshow('direction', cdst)
        cv2.imshow('frets', cdst2)
        cv2.waitKey()
