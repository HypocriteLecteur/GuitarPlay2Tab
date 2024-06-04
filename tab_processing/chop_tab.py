import cv2
import numpy as np
import glob
import os

def chop_measures(dir):
    lines_path = glob.glob(dir + '\\output\\' + r'*_*.png')
    lines_path.sort()

    num_measures_in_line = []
    for line_path in lines_path:
        img = cv2.imread(line_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = cv2.bitwise_not(img)

        lines = cv2.HoughLines(mask, 1, np.pi/180, 100, None)
        if lines is None:
            continue

        lines = lines[abs(lines[:, 0, 1]) < 0.01, :, :]  # filter non-vertical line
        rhos = lines[:, 0, 0].astype(int)
        rhos.sort()

        diff = 1/np.diff((rhos - rhos[0]) / (rhos[-1] - rhos[0]))
        
        num_measures_in_line.append(round(np.median(diff[diff <= 10])))
    return num_measures_in_line

def chop_lines(dir):
    images_path = glob.glob(dir + r'\*.png')
    images_path.sort()

    for page, image_path in enumerate(images_path):
        img = cv2.imread(image_path)
        cdst = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = cv2.bitwise_not(img)

        lines = cv2.HoughLines(mask, 1, np.pi/180, 500, None)
        if lines is None:
            continue

        lines = lines[abs(lines[:, 0, 1] - np.pi / 2) < 0.01, :, :]  # filter non-horizontal line
        rhos = lines[:, 0, 0].astype(int)
        rhos.sort()

        # for rho in rhos:
        #     pt1 = (int(1000), int(rho))
        #     pt2 = (int(-1000), int(rho))
        #     cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.imshow('', cdst), cv2.waitKey()

        split = np.squeeze(np.argwhere(np.diff(rhos) > 20))
        split = split[1::2]
        rhos_start = np.hstack((0, rhos[split]+10))
        rhos_end = np.hstack((rhos[split], rhos[-1]))
        # for rho in rhos_start:
        #     pt1 = (int(1000), int(rho))
        #     pt2 = (int(-1000), int(rho))
        #     cv2.line(cdst, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        # for rho in rhos_end:
        #     pt1 = (int(1000), int(rho))
        #     pt2 = (int(-1000), int(rho))
        #     cv2.line(cdst, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.imshow('', cdst), cv2.waitKey()

        # construct bounding box
        row_s2e = np.vstack((rhos_start, rhos_end)).T

        # bounding box refinement (start row)
        for s2e in row_s2e:
            while True:
                if np.sum(mask[s2e[0], :]) != 0:
                    break
                else:
                    s2e[0] = s2e[0] + 1
            # print(np.sum(mask[s2e[0], :]))

        # refine bounding box: add slack and confine to image shape
        slack = 10
        row_s2e[:, 0] = np.maximum(row_s2e[:, 0] - slack, np.zeros_like(row_s2e[:, 0]))
        row_s2e[:, 1] = np.minimum(row_s2e[:, 1] + slack, (img.shape[0]-1) * np.ones_like(row_s2e[:, 1]))

        if not os.path.exists(f'{dir}\\output'):
            os.makedirs(f'{dir}\\output')
        for i, s2e in enumerate(row_s2e):
            cv2.imwrite(f'{dir}\\output\\{page}_{i}.png', img[s2e[0]:s2e[1]+1, :])
            # cv2.imshow("", img[s2e[0]:s2e[1]+1, :]), cv2.waitKey()
    return

def chop_tab(dir):
    chop_lines(dir)
    num_measures_in_line = chop_measures(dir)
    np.save(dir + '\\num_measures_in_line.npy', num_measures_in_line)

if __name__ == '__main__':
    dir = 'test\\tab_pages'
    chop_tab(dir)