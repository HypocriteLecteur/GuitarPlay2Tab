import cv2
import numpy as np

def angle_diff(ang1, ang2):
    diff = (ang1 - ang2) % np.pi
    if diff > np.pi/2:
        return np.pi - diff
    else:
        return diff

def nothing(x):
    pass

class ColorThresholder:
    def __init__(self):
        cv2.namedWindow('color thresholder')

        cv2.createTrackbar('HMin','color thresholder',0,179,nothing) # Hue is from 0-179 for Opencv
        cv2.createTrackbar('SMin','color thresholder',0,255,nothing)
        cv2.createTrackbar('VMin','color thresholder',0,255,nothing)
        cv2.createTrackbar('HMax','color thresholder',0,179,nothing)
        cv2.createTrackbar('SMax','color thresholder',0,255,nothing)
        cv2.createTrackbar('VMax','color thresholder',0,255,nothing)

        # Set default value for MAX HSV trackbars.
        cv2.setTrackbarPos('HMax', 'color thresholder', 179)
        cv2.setTrackbarPos('SMax', 'color thresholder', 255)
        cv2.setTrackbarPos('VMax', 'color thresholder', 255)

        self.hMin = self.sMin = self.vMin = self.hMax = self.sMax = self.vMax = 0
    
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

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask= mask)

        cv2.imshow('color thresholder', output)