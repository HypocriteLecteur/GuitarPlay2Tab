import numpy as np
import cv2
import math
from sklearn.linear_model import RANSACRegressor, LinearRegression

from typing import Tuple, Union, List, Callable
from numpy.typing import NDArray
# from cv2.typing import MatLike
MatLike = NDArray

from ..fretboard import Fretboard

# ------------------------------------------------------------------------------------------------------------
def npprint(*data):
    with np.printoptions(precision=1, suppress=True):
        print(*data)

def angle_diff(ang1: float, ang2: float) -> float:
    diff = (ang1 - ang2) % np.pi
    if diff > np.pi/2:
        return np.pi - diff
    else:
        return diff

def angle_diff_np(ang_array: NDArray, ang2: float) -> NDArray:
    diff = (ang_array - ang2) % np.pi
    diff[diff > np.pi / 2] = np.pi - diff[diff > np.pi / 2]
    return diff

def find_longest_consecutive_subset(input_list: List, condition: Callable) -> List:
    max_subset = []
    current_max_subset = []
    max_subset_begin_index = -1
    current_subset_begin_index = -1
    flag = False

    for i, number in enumerate(input_list):
        if condition(number):
            if not flag:
                current_subset_begin_index = i
                flag = True
            current_max_subset.append(number)
        else:
            flag = False
            if len(current_max_subset) > len(max_subset):
                max_subset = current_max_subset
                max_subset_begin_index = current_subset_begin_index
            current_max_subset = []
    if flag and len(current_max_subset) > len(max_subset):
        max_subset = current_max_subset
    
    if max_subset_begin_index == -1:
        return 0, []
    return max_subset_begin_index, max_subset

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

# ------------------------------------------------------------------------------------------------------------
def transform_houghlines(lines_rt: NDArray, mat: NDArray) -> NDArray:
    '''
    Input:
        lines_rt = np.array([[r1, t1], [r2, t2], ...])
        mat = 3*3 homography matrix

    [cos(t1), sin(t1), -r1] -> [cos(t1), sin(t1), -r1] @ mat = [cos(t1'), sin(t1'), -r1']

    Output:
        lines_rt_transformed = np.array([[r1', t1'], [r2', t2'], ...])
    '''
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

def transform_points(points: NDArray, mat: NDArray) -> NDArray:
    '''
    Input:
        points = np.array([[x1, y1], [x2, y2], ...])
        mat = 3*3 homography matrix

    [x1, y1, 1].T -> mat @ [x1, y1, 1].T =  [x1', y1', 1].T

    output: 
    points_transformed = np.array([[x1', y1'], [x2', y2'], ...])
    '''
    points_transformed = np.vstack((points.T, np.ones((1, points.shape[0]))))
    points_transformed = mat @ points_transformed
    points_transformed = points_transformed / points_transformed[2, :].reshape((1,-1))
    return points_transformed[:2, :].T

def transform_fretboard(fretboard: Fretboard, bb: Tuple) -> Fretboard:
    shift_mat = np.eye(3)
    shift_mat[0, 2] = -bb[0]
    shift_mat[1, 2] = -bb[1]
    return Fretboard(transform_houghlines(fretboard.frets, shift_mat), 
                     transform_houghlines(fretboard.strings, shift_mat), 
                     ((fretboard.oriented_bb[0][0]+bb[0], fretboard.oriented_bb[0][1]+bb[1]), 
                      fretboard.oriented_bb[1], fretboard.oriented_bb[2]))

def linesP_to_houghlines(linesP, sort=True):
    angles = np.arctan2(linesP[:, 0, 2] - linesP[:, 0, 0], linesP[:, 0, 1] - linesP[:, 0, 3])
    dists = linesP[:, 0, 2] * np.cos(angles) + linesP[:, 0, 3] * np.sin(angles)
    angles[dists < 0] = angles[dists < 0] - np.sign(angles[dists < 0]) * np.pi
    dists[dists < 0] = -dists[dists < 0]

    if sort:
        angles, dists = (list(item) for item in zip(*sorted(zip(angles, dists), key=lambda x: x[1], reverse=False)))
    angles = np.array(angles)
    dists = np.array(dists)
    return np.concatenate((dists[:, np.newaxis], angles[:, np.newaxis]), axis=1)

def houghlines_y_from_x(lines, x):
    return (lines[:, 0] - x*np.cos(lines[:, 1])) / np.sin(lines[:, 1])

def houghlines_x_from_y(lines, y):
    return (lines[:, 0] - y*np.sin(lines[:, 1])) / np.cos(lines[:, 1])

def houghline_from_kb(slope, intercept):
    return np.array([-intercept/np.sqrt(slope**2 + 1), np.arctan2(-1/np.sqrt(slope**2 + 1), slope/np.sqrt(slope**2 + 1))])

class HoughBundler:     
    def __init__(self,min_distance=5,min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle
    
    def get_orientation(self, line):
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < -0.5) or (u > 1.5):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])
      
        if(len(lines) == 1):
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0],points[-1]]])

    def process_lines(self, lines):
        lines_horizontal  = []
        lines_vertical  = []
  
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical  = sorted(lines_vertical , key=lambda line: line[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)
                    
        return np.asarray(merged_lines_all)

# ------------------------------------------------------------------------------------------------------------
def line_image_intersection(line: Union[np.typing.NDArray, List], shape: Tuple[int]) -> NDArray:
    '''
    This assumes the line is within the image's boundaries.

    Input:
        line = [r, t]
        shape = (height, width)
    Output:
        Two intersection points
    '''
    if line[1] == 0:
        return np.array([
            [line[0], 0],
            [line[0], shape[0]]
        ]).astype(int)
    elif line[1] == np.pi/2:
        return np.array([
            [0, line[0]],
            [shape[1], line[0]]
        ]).astype(int)
    else:
        four_intersection = np.array([
            [0, line[0] / np.sin(line[1])],
            [line[0] / np.cos(line[1]), 0],
            [shape[1], (line[0] - shape[1]*np.cos(line[1])) / np.sin(line[1])],
            [(line[0] - shape[0]*np.sin(line[1])) / np.cos(line[1]), shape[0]]
        ]).astype(int)
        valid_intersections = (four_intersection[:, 0] >= 0) & (four_intersection[:, 0] <= shape[1]) & \
            (four_intersection[:, 1] >= 0) & (four_intersection[:, 1] <= shape[0])
        return four_intersection[valid_intersections]

def cornerpoints_from_line_pair(line_pair: NDArray, shape: Tuple[int]) -> NDArray:
    cornerpoints = np.vstack((line_image_intersection(line_pair[0], shape), 
                              line_image_intersection(line_pair[1], shape)))
    return cornerpoints[[0, 1, 3, 2]]

def oriented_bb_from_line_pair(line_pair: NDArray, shape: Tuple[int]) -> Tuple:
    return cv2.minAreaRect(cornerpoints_from_line_pair(line_pair, shape))

def crop_from_oriented_bb(frame: MatLike, rect: Tuple) -> Tuple[MatLike, NDArray]:
    '''
    Robustly crop the image from oriented bounding box even when outside of image boundary.
    Returns the cropped image and the transformation matrix.

    The transformation matrix is needed to transform lines and points to the new cropped image
    '''
    bb = cv2.boundingRect(np.intp(cv2.boxPoints(rect)))
    if bb[0] < 0 or bb[0] > frame.shape[1] or bb[1] < 0 or bb[1] > frame.shape[0]:
        top_pad = max(0, -bb[1])
        down_pad = max(0, bb[3] - frame.shape[0])
        left_pad = max(0, -bb[0])
        right_pad = max(0, bb[2] - frame.shape[1])
        paddded_frame = cv2.copyMakeBorder(frame, top_pad, down_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, None, 0)
        padded_bb = (bb[0] + left_pad, bb[1] + top_pad, bb[2], bb[3])
        cropped_frame = paddded_frame[padded_bb[1]:padded_bb[1]+padded_bb[3], padded_bb[0]:padded_bb[0]+padded_bb[2]]
    else:
        cropped_frame = frame[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]

    shift_mat = np.eye(3)
    shift_mat[0, 2] = -bb[0]
    shift_mat[1, 2] = -bb[1]

    if rect[1][1] > rect[1][0]:
        angle = -(90-rect[2])
        h = rect[1][0]
        w = rect[1][1]
    else:
        angle = rect[2]
        h = rect[1][1]
        w = rect[1][0]
    
    # rotated_cropped_frame, rot_mat = rotate_image(cropped_frame, 
    #                                               np.array(rect[0]) - [bb[0], bb[1]], 
    #                                               angle, 
    #                                               custom_shape=(int(bb[2]/2+w/2 - min(bb[2]/2-w/2, 0)), cropped_frame.shape[0]), 
    #                                               additional_shift=[-(rect[0][0] - bb[0] - w/2), 0])
    # rotated_cropped_frame = rotated_cropped_frame[int(bb[3]/2-h/2):int(bb[3]/2+h/2), 0:int(w)]
    cropped_center = np.array(rect[0]) - [bb[0], bb[1]]
    rotated_cropped_frame, rot_mat = rotate_image(cropped_frame, 
                                                  cropped_center, 
                                                  angle, 
                                                  custom_shape=(int(cropped_center[0] + w/2), cropped_frame.shape[0]))
    rotated_cropped_frame = rotated_cropped_frame[int(cropped_center[1] - h/2):int(cropped_center[1] + h/2), max(int(cropped_center[0] - w/2), 0):]

    shift_mat2 = np.eye(3)
    shift_mat2[0, 2] = -max(int(cropped_center[0] - w/2), 0)
    shift_mat2[1, 2] = -int(cropped_center[1] - h/2)
    
    return rotated_cropped_frame, shift_mat2 @ np.concatenate((rot_mat, np.array([0, 0, 1]).reshape((1, 3))), axis=0) @ shift_mat

def crop_mat_from_oriented_bb(rect: Tuple) -> NDArray:
    if rect[1][1] > rect[1][0]:
        angle = -(90-rect[2])
        h = rect[1][0]
        w = rect[1][1]
    else:
        angle = rect[2]
        h = rect[1][1]
        w = rect[1][0]
    transform_mat = np.eye(3)
    transform_mat[:2, :] = cv2.getRotationMatrix2D(np.array(rect[0]), angle, 1.0)
    transform_mat[0, 2] = transform_mat[0, 2] - rect[0][0] + w/2
    transform_mat[1, 2] = transform_mat[1, 2] - rect[0][1] + h/2
    return transform_mat

def rotate_image(image, image_center, angle, custom_shape=None, additional_shift=None):
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  # deg
  if additional_shift is not None:
    rot_mat[0, 2] = rot_mat[0, 2] + additional_shift[0]
    rot_mat[1, 2] = rot_mat[1, 2] + additional_shift[1]
  if custom_shape is None:
      custom_shape = image.shape[1::-1]
  result = cv2.warpAffine(image, rot_mat, custom_shape, flags=cv2.INTER_LINEAR)
  return result, rot_mat

def line_line_intersection(l1, l2):
    tmp = np.cross([np.cos(l1[1]), np.sin(l1[1]), -l1[0]], [np.cos(l2[1]), np.sin(l2[1]), -l2[0]])
    return tmp[:2] / tmp[2]

def line_line_intersection_batch(lines, l2):
    tmp = np.cross([np.cos(lines[:, 1]), np.sin(lines[:, 1]), -lines[:, 0]], [np.cos(l2[1]), np.sin(l2[1]), -l2[0]])
    return tmp[:2] / tmp[2]

def oriented_bb_from_frets_strings(frets: NDArray, strings: NDArray) -> Tuple:
    cornerpoints = np.array([
        line_line_intersection(strings[0], frets[0]),
        line_line_intersection(strings[-1], frets[0]),
        line_line_intersection(strings[-1], frets[-1]),
        line_line_intersection(strings[0], frets[-1]),
    ])
    return cv2.minAreaRect(cornerpoints.astype(int))

def prepare_draw_frets(oriented_bb, frets):
    transform = crop_mat_from_oriented_bb(oriented_bb)
    if oriented_bb[1][1] > oriented_bb[1][0]:
        h = oriented_bb[1][0]
        w = oriented_bb[1][1]
    else:
        h = oriented_bb[1][1]
        w = oriented_bb[1][0]
    
    lines = transform_houghlines(frets, np.linalg.inv(transform))
    pts1 = np.zeros_like(frets)
    pts2 = np.zeros_like(frets)

    pts1[:, 0] = houghlines_x_from_y(lines, 0)

    pts2[:, 1] = h
    pts2[:, 0] = houghlines_x_from_y(lines, h)
    return transform_points(pts1, np.linalg.inv(transform)).astype(int), transform_points(pts2, np.linalg.inv(transform)).astype(int)

def prepare_draw_strings(oriented_bb, strings):
    transform = crop_mat_from_oriented_bb(oriented_bb)
    if oriented_bb[1][1] > oriented_bb[1][0]:
        h = oriented_bb[1][0]
        w = oriented_bb[1][1]
    else:
        h = oriented_bb[1][1]
        w = oriented_bb[1][0]

    lines = transform_houghlines(strings, np.linalg.inv(transform))
    pts1 = np.zeros_like(strings)
    pts2 = np.zeros_like(strings)

    pts1[:, 1] = houghlines_y_from_x(lines, 0)

    pts2[:, 0] = w
    pts2[:, 1] = houghlines_y_from_x(lines, w)
    return transform_points(pts1, np.linalg.inv(transform)).astype(int), transform_points(pts2, np.linalg.inv(transform)).astype(int)

def mask_out_oriented_bb(frame, oriented_bb):
    mask = (np.ones_like(frame)*255).astype(np.uint8)
    cv2.fillPoly(mask, pts=[np.int0(cv2.boxPoints(oriented_bb))], color=(0, 0, 0))
    frame = cv2.bitwise_and(frame, mask)
    return frame

def ransac_vanishing_point_estimation(linesP):
    angles = np.arctan2(linesP[:, 0, 2] - linesP[:, 0, 0], linesP[:, 0, 1] - linesP[:, 0, 3])
    dists = linesP[:, 0, 2] * np.cos(angles) + linesP[:, 0, 3] * np.sin(angles)

    line_coords = np.vstack((np.cos(angles), np.sin(angles), -dists)).T

    ransac_lm = RANSACRegressor(estimator=LinearRegression(fit_intercept=False))
    ransac_lm.fit(line_coords[:, :2], -line_coords[:, 2])
    vanishing_point = ransac_lm.estimator_.coef_
    return vanishing_point, ransac_lm.inlier_mask_

def ransac_vanishing_point_estimation_lines(lines):
    line_coords = np.vstack((np.cos(lines[:, 1]), np.sin(lines[:, 1]), -lines[:, 0])).T

    ransac_lm = RANSACRegressor(estimator=LinearRegression(fit_intercept=False))
    ransac_lm.fit(line_coords[:, :2], -line_coords[:, 2])
    vanishing_point = ransac_lm.estimator_.coef_
    return vanishing_point, ransac_lm.inlier_mask_

# Not robust
# --------------------------------------------
# def find_projective_rectification(vp):
#     homography = np.array([
#         [1, 0, 0], 
#         [0, 1, 0], 
#         [0, -1/vp[1], 1]
#     ])
#     affine = np.array([
#         [1, -vp[0] / vp[1], 0],
#         [0, 1, 0],
#         [0, 0, 1]
#     ])
#     return affine @ homography
# --------------------------------------------

def find_rectification(lines, shape):
    h = shape[0]
    src_pts = np.zeros((2*lines.shape[0], 2))
    dst_pts = np.zeros((2*lines.shape[0], 2))

    dst_pts[:, 0] = np.repeat(houghlines_x_from_y(lines, 0), 2)
    dst_pts[1::2, 1] = h

    src_pts[0:-1:2, 0] = dst_pts[0:-1:2, 0]
    src_pts[1::2, 0] = houghlines_x_from_y(lines, h)
    src_pts[1::2, 1] = h
    homography, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return homography, inliers

def compute_homography(image, vp1, vp2, clip=True):
    """Compute homography from vanishing points and warp the image.

    It is assumed that vp1 and vp2 correspond to horizontal and vertical
    directions, although the order is not assumed.
    Firstly, projective transform is computed to make the vanishing points go
    to infinty so that we have a fronto parellel view. Then,Computes affine
    transfom  to make axes corresponding to vanishing points orthogonal.
    Finally, Image is translated so that the image is not missed. Note that
    this image can be very large. `clip` is provided to deal with this.

    Parameters
    ----------
    image: ndarray
        Image which has to be wrapped.
    vp1: ndarray of shape (3, )
        First vanishing point in homogenous coordinate system.
    vp2: ndarray of shape (3, )
        Second vanishing point in homogenous coordinate system.
    clip: bool, optional
        If True, image is clipped to clip_factor.
    clip_factor: float, optional
        Proportion of image in multiples of image size to be retained if gone
        out of bounds after homography.
    Returns
    -------
    warped_img: ndarray
        Image warped using homography as described above.
    """
    # Find Projective Transform
    vanishing_line = np.cross(vp1, vp2)
    H = np.eye(3)
    H[2] = vanishing_line / vanishing_line[2]
    H = H / H[2, 2]

    # Find directions corresponding to vanishing points
    v_post1 = np.dot(H, vp1)
    v_post2 = np.dot(H, vp2)
    v_post1 = v_post1 / np.sqrt(v_post1[0]**2 + v_post1[1]**2)
    v_post2 = v_post2 / np.sqrt(v_post2[0]**2 + v_post2[1]**2)

    directions = np.array([[v_post1[0], -v_post1[0], v_post2[0], -v_post2[0]],
                           [v_post1[1], -v_post1[1], v_post2[1], -v_post2[1]]])

    thetas = np.arctan2(directions[1], directions[0])

    # Find direction closest to horizontal axis
    h_ind = np.argmin(np.abs(thetas))

    # Find positve angle among the rest for the vertical axis
    if h_ind // 2 == 0:
        v_ind = 2 + np.argmax([thetas[2], thetas[3]])
    else:
        v_ind = np.argmax([thetas[2], thetas[3]])

    A1 = np.array([[directions[0, h_ind], directions[0, v_ind], 0],
                   [directions[1, h_ind], directions[1, v_ind], 0],
                   [0, 0, 1]])
    # Might be a reflection. If so, remove reflection.
    if np.linalg.det(A1) < 0:
        A1[:, 0] = -A1[:, 0]

    A = np.linalg.inv(A1)

    # Translate so that whole of the image is covered
    inter_matrix = np.dot(A, H)

    cords = np.dot(inter_matrix, [[0, 0, image.shape[1], image.shape[1]],
                                  [0, image.shape[0], 0, image.shape[0]],
                                  [1, 1, 1, 1]])
    cords = cords[:2] / cords[2]

    tx = min(0, cords[0].min())
    ty = min(0, cords[1].min())

    if clip:
        # These might be too large. Clip them.
        sx = image.shape[1] / (max(cords[0]) - tx)
        sy = image.shape[0] / (max(cords[1]) - ty)
    else:
        sx = 1
        sy = 1
    
    T = np.array([[sx, 0, -sx*tx],
                  [0, sy, -sy*ty],
                  [0, 0, 1]])

    final_homography = np.dot(T, inter_matrix)
    return final_homography