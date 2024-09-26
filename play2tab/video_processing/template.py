import numpy as np
import cv2
from utils.utils import draw_line

def transform_points(points, mat):
    '''
    points  N * 2
    mat 3 * 3
    output: mat @ [x, y, 1].T
    '''
    points_transformed = np.vstack((points.T, np.ones((1, points.shape[0]))))
    points_transformed = mat @ points_transformed
    points_transformed = points_transformed / points_transformed[2, :].reshape((1,-1))
    return points_transformed[:2, :].T

def transform_houghlines(lines_rt, mat):
    '''
    lines_rt = np.array([[r1, t1], [r2, t2], ...])
    mat = homography, 3*3

    [cos(t1), sin(t1), -r1] -> [cos(t1), sin(t1), -r1] @ mat -> [r1', t1']
    '''
    if lines_rt.ndim == 1:
        lines_rt = lines_rt.reshape((1, 2))
    lines_rt_transformed = np.zeros_like(lines_rt)

    tmp = np.zeros((lines_rt_transformed.shape[0], 3))
    tmp[:, 0] = np.cos(lines_rt[:, 1])
    tmp[:, 1] = np.sin(lines_rt[:, 1])
    tmp[:, 2] = -lines_rt[:, 0]
    tmp = tmp @ mat
    tmp = tmp / np.linalg.norm(tmp[:, :2], axis=1).reshape((-1, 1))

    lines_rt_transformed[:, 0] = -tmp[:, 2]
    lines_rt_transformed[:, 1] = np.arctan2(tmp[:, 1] , tmp[:, 0])
    return lines_rt_transformed


def shift_center_houghlines(new_center_xy, lines_rt):
    '''
    lines_rt = np.array([[r1, t1], [r2, t2], ...])
    '''
    x0, y0 = new_center_xy
    phi = np.arctan2(y0, x0)
    shifted_lines_rt = lines_rt.copy()
    shifted_lines_rt[:, 0] = shifted_lines_rt[:, 0] - np.sqrt(x0**2 + y0**2) * np.cos(phi - shifted_lines_rt[:, 1])
    return shifted_lines_rt

def rotate_line_pair(angle, rot_mat, line_pair):
    '''
    angle: rad
    rot_mat: the 2 * 3 transformation
    line_pair: [[r1, t1, h1], [r2, t2, h2], ...]
    '''
    rotated_line_pair = line_pair.copy()
    m = -rot_mat [:, :2].T @ rot_mat [:, 2]
    rotated_line_pair[:, 0] = -np.cos(rotated_line_pair[:, 1]) * m[0] - np.sin(rotated_line_pair[:, 1]) * m[1] + rotated_line_pair[:, 0]

    rotated_line_pair[:, 1] = rotated_line_pair[:, 1] - angle
    return rotated_line_pair

class Template:
    def __init__(self) -> None:
        self.image = None
        self.cdst = None
        self.fingerboard_lines = None
        self.fretlines = None
    
    def register_image(self, image):
        self.image = image
        self.cdst = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)

    def register_fingerboard_lines(self, fingerboard_lines):
        self.fingerboard_lines = fingerboard_lines

    def register_rotation(self, center, angle, rot_mat):
        self.center = center
        self.rot_angle = angle
        self.rot_mat = rot_mat

    def register_padded_rect(self, padded_rect):
        self.padded_rect = padded_rect
    
    def register_shift(self, new_xy):
        self.shift = new_xy
    
    def register_homography(self, homography):
        self.homography = homography
    
    def register_final_shift(self, new_xy):
        self.final_shift = new_xy
    
    def register_fretlines(self, fretlines):
        self.template_fretlines = np.zeros((len(fretlines), 2))
        self.template_fretlines[:, 0] = fretlines # fretlines in the template coordinate system

        rot_mat = np.zeros((3, 3))
        rot_mat[2, 2] = 1
        rot_mat[:2, :] = self.rot_mat
        
        shift_mat = np.eye(3)
        shift_mat[0, 2] = -self.shift[0]
        shift_mat[1, 2] = -self.shift[1]

        second_shift_mat = np.eye(3)
        second_shift_mat[0, 2] = -self.final_shift[0]
        second_shift_mat[1, 2] = -self.final_shift[1]

        self.total_mat = second_shift_mat @ self.homography @ shift_mat @ rot_mat

        # _, rot, t, n = cv2.decomposeHomographyMat(self.total_mat, np.eye(3))
        # homography = rot[0] + t[0] @ n[0].T

        self.image_fretlines = transform_houghlines(self.template_fretlines, self.total_mat)

        self.template_fingerboard_lines = transform_houghlines(self.fingerboard_lines, np.linalg.inv(self.total_mat))

    def register_template_image(self, image):
        self.template_image = image
        # self.template_image_edge = cv2.Canny(image, 30, 100, 5)
    
    def register_template_marker_positions(self, template_marker_positions):
        self.template_marker_positions = transform_points(template_marker_positions, np.linalg.inv(self.total_mat))
    
    def draw_fretboard(self, cdst, transform_mat, draw_fingerboard_lines=True, color=(0, 255, 0)):
        transform_mat_inv = np.linalg.inv(transform_mat)
        if draw_fingerboard_lines is True:
            for line in transform_houghlines(self.template_fingerboard_lines, transform_mat_inv):
                draw_line(cdst, line[0], line[1], color=color)
        for line in transform_houghlines(self.template_fretlines, transform_mat_inv):
            draw_line(cdst, line[0], line[1], color=color)