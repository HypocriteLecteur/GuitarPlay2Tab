import numpy as np

def angle_diff(ang1, ang2):
    diff = (ang1 - ang2) % np.pi
    if diff > np.pi/2:
        return np.pi - diff
    else:
        return diff