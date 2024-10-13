from detector import DetectorInterface
from tracker import TrackerInterface

import cv2
import numpy as np
from utils.visualize import draw_fretboard

from utils.utils_math import mask_out_oriented_bb

from typing import Tuple
from cv2.typing import MatLike

class FretboardDetector:
    '''
    FretboardDetector operates on two stages: initiali
    zation and tracking.

    While not initialized, it will try to detect and locate the fretboard.
    After a successful detection, a template for the fretboard would be made before it switch to the tracking stage.

    During tracking, the tracker will search for the fretboard in an ROI determined by its pose in the previous frame.
    If the fretboard is lost, switch back to initialization stage.
    '''

    def __init__(self, detector: DetectorInterface, tracker: TrackerInterface, frame_shape: Tuple[int, int]) -> None:
        self.is_initialized = False

        self.backSub = cv2.createBackgroundSubtractorMOG2()

        self.detector = detector
        self.tracker = tracker

        self.fretboard = None

        self.video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'H264'), 25, frame_shape)

        self.counter = 0

    def detect(self, frame: MatLike) -> None:
        if not self.is_initialized:
            self.is_initialized, self.fretboard = self.detector.detect(frame)

            if self.is_initialized:
                frame_bg = mask_out_oriented_bb(frame, self.fretboard.oriented_bb)
                self.backSub.apply(frame_bg)
            
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            # self.colorthresholder = ColorThresholder(self.screensize)
        else:
            fgmask = self.backSub.apply(frame, None, learningRate=0)

            self.is_initialized, self.fretboard = self.tracker.track(frame, fgmask, self.fretboard)
            if not self.is_initialized:
                self.is_initialized, self.fretboard = self.detector.detect(frame) 
            
            if self.is_initialized:
                frame_bg = mask_out_oriented_bb(frame, self.fretboard.oriented_bb)
                self.backSub.apply(frame_bg, None, learningRate=-1)
            # cv2.imshow('bg_model', self.backSub.getBackgroundImage())
            cv2.waitKey(1)

            # self.videoplayback.put(final_cdst)
            # self.videoplayback.imshow()
        
        final_cdst = frame.copy()
        if self.fretboard is not None:
            draw_fretboard(final_cdst, self.fretboard)
        cv2.putText(final_cdst, f'{self.counter}', (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

        self.video.write(final_cdst)
        self.counter = self.counter + 1

if __name__ == '__main__':
    cap = cv2.VideoCapture('test\\guitar3.mp4')
    frame_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    loop_video = False

    from detector import Detector
    from tracker import Tracker
    fd = FretboardDetector(detector=Detector(), tracker=Tracker(), frame_shape=frame_shape)

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            else:
                break
        fd.detect(frame)
    fd.video.release()