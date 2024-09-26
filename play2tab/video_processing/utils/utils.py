import cv2
import numpy as np
from skimage.transform import SimilarityTransform
from skimage.measure import ransac

def npprint(*data):
    with np.printoptions(precision=3, suppress=True):
        print(*data)

def get_screensize():
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return screensize

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

def prepare_draw_fretboard(gray, tracked_fretlines, crop_transform):
    lines = transform_houghlines(tracked_fretlines, np.linalg.inv(crop_transform))
    pts1 = np.zeros_like(tracked_fretlines)
    pts2 = np.zeros_like(tracked_fretlines)

    pts1[:, 0] = lines[:, 0] / np.cos(lines[:, 1])

    pts2[:, 1] = gray.shape[0]
    pts2[:, 0] = (lines[:, 0] - pts2[:, 1] * np.sin(lines[:, 1])) / np.cos(lines[:, 1])
    return transform_points(pts1, np.linalg.inv(crop_transform)).astype(int), transform_points(pts2, np.linalg.inv(crop_transform)).astype(int)

def angle_diff(ang1, ang2):
    diff = (ang1 - ang2) % np.pi
    if diff > np.pi/2:
        return np.pi - diff
    else:
        return diff

def angle_diff_np(ang_array, angle2):
    diff = (ang_array - angle2) % np.pi
    diff[diff > np.pi / 2] = np.pi - diff[diff > np.pi / 2]
    return diff

def second_order_difference(x1, x2, x3):
    return (x1 + x3 - 2*x2) / x2

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

def fret_ratio(x1, x2, x3):
    return max(x3 - x2, x2 - x1) / min(x3 - x2, x2 - x1)

def nothing(x):
    pass

class ColorThresholder:
    def __init__(self, screensize):
        cv2.namedWindow('color thresholder')

        cv2.createTrackbar('HMin','color thresholder',0,179,nothing) # Hue is from 0-179 for Opencv
        cv2.createTrackbar('SMin','color thresholder',0,255,nothing)
        cv2.createTrackbar('VMin','color thresholder',0,255,nothing)
        cv2.createTrackbar('HMax','color thresholder',0,179,nothing)
        cv2.createTrackbar('SMax','color thresholder',0,255,nothing)
        cv2.createTrackbar('VMax','color thresholder',0,255,nothing)

        # Set default value for MAX HSV trackbars.
        cv2.setTrackbarPos('HMin', 'color thresholder', 0)
        cv2.setTrackbarPos('SMin', 'color thresholder', 0)
        cv2.setTrackbarPos('VMin', 'color thresholder', 0)
        cv2.setTrackbarPos('HMax', 'color thresholder', 255)
        cv2.setTrackbarPos('SMax', 'color thresholder', 136)
        cv2.setTrackbarPos('VMax', 'color thresholder', 140)

        self.hMin = self.sMin = self.vMin = self.hMax = self.sMax = self.vMax = 0

        self.screensize = screensize

        self.output = None
    
    def update(self, image):
        self.hMin = cv2.getTrackbarPos('HMin','color thresholder')
        self.sMin = cv2.getTrackbarPos('SMin','color thresholder')
        self.vMin = cv2.getTrackbarPos('VMin','color thresholder')

        self.hMax = cv2.getTrackbarPos('HMax','color thresholder')
        self.sMax = cv2.getTrackbarPos('SMax','color thresholder')
        self.vMax = cv2.getTrackbarPos('VMax','color thresholder')

        # Set minimum and max HSV values to display
        lower = np.array([self.hMin, self.sMin, self.vMin])
        upper = np.array([self.hMax, self.sMax, self.vMax])

        # Scale Image
        # image = cv2.resize(image, (int(self.screensize[0]/4), int(self.screensize[1]/4)))

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(hsv, lower, upper)
        self.output = cv2.bitwise_and(image, image, mask= mask)

        self.display = cv2.resize(self.output, (int(self.screensize[0]/4), int(self.screensize[1]/4)))

        cv2.imshow('color thresholder', self.display)

def cornerpoints_from_line_pair(line_pair, width):
    '''
    line_pair = [[r1, t1], [r2, t2]]
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

def rotate_image(image, image_center, angle, custom_shape=None, additional_shift=None):
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  # deg
  if additional_shift is not None:
    rot_mat[0, 2] = rot_mat[0, 2] + additional_shift[0]
    rot_mat[1, 2] = rot_mat[1, 2] + additional_shift[1]
  if custom_shape is None:
      custom_shape = image.shape[1::-1]
  result = cv2.warpAffine(image, rot_mat, custom_shape, flags=cv2.INTER_LINEAR)
  return result, rot_mat

def crop_from_oriented_bounding_box(frame, rect):
    bb = cv2.boundingRect(np.intp(cv2.boxPoints(rect)))
    # robustly crop the image for when bb is outside of image boundary
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

    rotated_cropped_frame, rot_mat = rotate_image(cropped_frame, np.array(rect[0]) - [bb[0], bb[1]], angle, (int(bb[2]/2+w/2 - min(bb[2]/2-w/2, 0)), cropped_frame.shape[0]), additional_shift=[-(rect[0][0] - bb[0] - w/2), 0])

    rotated_cropped_frame = rotated_cropped_frame[int(bb[3]/2-h/2):int(bb[3]/2+h/2), 0:int(w)]

    shift_mat2 = np.eye(3)
    shift_mat2[0, 2] = 0
    shift_mat2[1, 2] = -int(bb[3]/2-h/2)
    
    return rotated_cropped_frame, shift_mat2 @ np.concatenate((rot_mat, np.array([0, 0, 1]).reshape((1, 3))), axis=0) @ shift_mat

def project_point_to_line(line_param, point):
    '''
    line_param: [a, b], ax + by + 1 = 0
    point: [x, y]

    the line can be parameterized as: p0 + t * v, where p0 is a point on the line, and
    v is a directional vector.

    after projecting the point on the line, we can solve for t:
        t = dot(point - p0, v) / norm(v)
    
    we choose p0 = [0, -1/b], v = [-b, a]
    '''
    return np.dot(point - [0, -1/line_param[1]], [-line_param[1], line_param[0]]) / np.linalg.norm(line_param)

def point_distance_to_line(point, dist, angle):
    return dist - point[0] * np.cos(angle) - point[1] * np.sin(angle)

def transform_houghlines(lines_rt, mat):
    '''
    lines_rt = np.array([[r1, t1], [r2, t2], ...])
    mat = homography, 3*3

    [cos(t1), sin(t1), -r1] -> [cos(t1), sin(t1), -r1] @ mat -> [r1', t1']
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

# def find_homography_from_matched_fretlines(template_lines, des_lines):
#     src_pts = np.zeros((template_lines.shape[0], 2))
#     src_pts[:, 0] = -np.cos(template_lines[:, 1]) / template_lines[:, 0]
#     src_pts[:, 1] = -np.sin(template_lines[:, 1]) / template_lines[:, 0]

#     dst_pts = np.zeros((des_lines.shape[0], 2))
#     dst_pts[:, 0] = -np.cos(des_lines[:, 1]) / des_lines[:, 0]
#     dst_pts[:, 1] = -np.sin(des_lines[:, 1]) / des_lines[:, 0]

#     homography, mask = cv2.findHomography(src_pts, dst_pts, 0,5.0)
#     inliers = np.argwhere(mask[:, 0]).squeeze()
#     return np.linalg.inv(homography), inliers

def find_homography_from_matched_fretlines(template_lines, des_lines):
    des_homography = find_projective_rectification(des_lines)

    des_lines_rect = transform_houghlines(des_lines, des_homography)
    des_lines_rect[:, 1] = np.zeros((des_lines_rect.shape[0],))

    template_homography = find_projective_rectification(template_lines)

    template_homography_rect = transform_houghlines(template_lines, template_homography)
    template_homography_rect[:, 1] = np.zeros((template_lines.shape[0],))

    model_robust, inliers = ransac(
        (template_homography_rect, des_lines_rect), SimilarityTransform, min_samples=2, residual_threshold=20, max_trials=1000
    )
    return np.linalg.inv(des_homography) @ np.linalg.inv(model_robust.params) @ template_homography, inliers

def find_projective_rectification(lines):
    src_pts = np.empty((0, 2))
    dst_pts = np.empty((0, 2))
    for dist, angle in lines:
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
    return np.linalg.inv(homography)

def ransac_vanishing_point_estimation(linesP, num_ransac_iter=100, threshold_inlier=5):
    locations = (linesP[:, 0, [0, 1]] + linesP[:, 0, [2, 3]]) / 2
    directions = linesP[:, 0, [2, 3]] - linesP[:, 0, [0, 1]]
    strengths = np.linalg.norm(directions, axis=1)

    directions = directions / strengths[:, np.newaxis]

    lines = homogeneous_coords_from_linesP(locations, directions)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(
            (locations, directions, strengths), current_model, threshold_inlier)
        
        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            # logging.info("Current best model has {} votes at iteration {}".format(
                # current_votes.sum(), ransac_iter))
    return best_model

def homogeneous_coords_from_linesP(locations, directions):
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines

def compute_votes(edgelets, model, threshold_inlier=5):
    """Compute votes for each of the edgelet against a given vanishing point.

    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.

    Returns
    -------
    votes: ndarry of shape (n_edgelets,)
        Votes towards vanishing point model for each of the edgelet.

    """
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths