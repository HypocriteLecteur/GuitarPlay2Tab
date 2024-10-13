import numpy as np
import matplotlib.pyplot as plt

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

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def npprint(*data):
    with np.printoptions(precision=3, suppress=True):
        print(*data)

distrubance = np.array([
    [1, 0, 0], 
    [0, 1, 0], 
    [0, 0, 1]
])

pnts = []
for i in range(15):
    pnts.append([i*100 - 700, 0, 1])
pnts = np.array(pnts)

pnts_dist = []
for i in range(pnts.shape[0]):
    pnts_dist.append(distrubance @ pnts[i])
pnts_dist = np.array(pnts_dist)
pnts_dist = pnts_dist / pnts_dist[:, 2].reshape((-1, 1))

vp = np.array([0, 1000, 1])

vp_dist = np.array(distrubance @ vp)

lines = []
for i in range(pnts.shape[0]):
    lines.append(np.cross(pnts_dist[i], vp_dist))
lines = np.array(lines)

pnts2 = []
for i in range(pnts.shape[0]):
    pnts2.append([(-lines[i][2] - lines[i][1]*200) / lines[i][0], 200, 1])
pnts2 = np.array(pnts2)

vp2 = np.array([10, 999, 1])
vp2_dist = np.array(distrubance @ vp2)

vl = np.cross(vp_dist, vp2_dist)
vl = vl / vl[2]

# vl = np.array([-2.00299549e-03,  2.76803683e-05,  1.00000000e+00])

homography = np.array([
    [1, 0, 0], 
    [0, 1, 0], 
    [vl[0], vl[1], vl[2]]
])
affine = np.array([
    [1, -vp[0] / vp[1], 0],
    [0, 1, 0],
    [0, 0, 1]
])
# homography = affine @ homography
# homography = np.random.random((3, 3))
# homography = homography / homography[2, 2]

pnts_rect = []
for i in range(pnts.shape[0]):
    pnts_rect.append(homography @ pnts_dist[i])
pnts_rect = np.array(pnts_rect)
pnts_rect = pnts_rect / pnts_rect[:, 2].reshape((-1, 1))

pnts2_rect = [] # transform_points(pnts2[:, :2], homography)
for i in range(pnts.shape[0]):
    pnts2_rect.append(homography @ pnts2[i])
pnts2_rect = np.array(pnts2_rect)
pnts2_rect = pnts2_rect / pnts2_rect[:, 2].reshape((-1, 1))

lines_rect = []
for i in range(pnts.shape[0]):
    lines_rect.append(np.linalg.inv(homography).T @ lines[i])
lines_rect = np.array(lines_rect)
# lines_rect = lines_rect / np.linalg.norm(lines_rect[:, :2], axis=1).reshape((-1, 1))

npprint(pnts_dist)
npprint(pnts_rect)

fig, ax = plt.subplots()
ax.scatter(pnts_dist[:, 0], pnts_dist[:, 1])
ax.scatter(vp_dist[0], vp_dist[1])
ax.scatter(pnts2[:, 0], pnts2[:, 1])
[abline(-l[0] / l[1], -l[2] / l[1]) for l in lines]
plt.xlim(-1000, 1000)
plt.ylim(-1000, 1000)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

fig, ax = plt.subplots()
ax.scatter(pnts_rect[:, 0], pnts_rect[:, 1])
ax.scatter(pnts2_rect[:, 0], pnts2_rect[:, 1])
[abline(-l[0] / l[1], -l[2] / l[1]) for l in lines_rect]
plt.xlim(-1000, 1000)
plt.ylim(-1000, 1000)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()