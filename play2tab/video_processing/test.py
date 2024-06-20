import numpy as np
import cv2

line_template = np.array([   0.,   14.,   44.,   71.,   94.,  130.,  157.,  190.,  226.,
                          269.,  310.,  352.,  397.,  445.,  496.,  550.,  607.,
                          662.,  726.,  799.,  871.,  947., 1023.])

line_template_homogeneous = np.zeros((line_template.shape[0], 3))
line_template_homogeneous[:, 0] = 1
line_template_homogeneous[:, 2] = -line_template

line_detected = np.array([251.0, 272.0, 292.0, 320.0, 346.0, 372.0, 397.0, 430.0, 457.0, 489.0, 
                 525.0, 560.0, 592.0, 633.0, 672.0, 708.0, 746.0, 793.0, 842.0, 882.0, 937.0, 986.0, 1038.0])
line_detected_template_homogeneous = np.zeros((line_detected.shape[0], 3))
line_detected_template_homogeneous[:, 0] = 1
line_detected_template_homogeneous[:, 2] = -line_detected

transform = np.linalg.pinv(line_template_homogeneous) @ line_detected_template_homogeneous
dists = -(line_template_homogeneous @ transform)[:, 2]
print('')