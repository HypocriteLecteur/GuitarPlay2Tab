from video_processing.detector import DetectorInterface
from video_processing.tracker import TrackerInterface

import cv2
import numpy as np
from video_processing.utils.visualize import draw_fretboard

from video_processing.utils.utils_math import mask_out_oriented_bb, crop_from_oriented_bb, transform_points

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

    def __init__(self, detector: DetectorInterface, tracker: TrackerInterface, hand_detector: DetectorInterface, frame_shape: Tuple[int, int]) -> None:
        self.is_initialized = False

        self.backSub = cv2.createBackgroundSubtractorMOG2()

        self.detector = detector
        self.tracker = tracker
        self.hand_detector = hand_detector

        self.fretboard = None

        self.video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'H264'), 20, frame_shape)

        self.counter = 0

    def detect(self, frame: MatLike) -> None:
        if not self.is_initialized:
            resized, scale = resize_with_aspect(frame, (540, 960))
            bbox = cv2.selectROI("Select ROI", resized, fromCenter=False, showCrosshair=True)
            bbox = tuple([int(x / scale) for x in bbox])
            self.is_initialized, self.fretboard = self.detector.detect(frame, bbox)
            if self.is_initialized:
                self.tracker.create_template(frame, self.fretboard)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
            # self.colorthresholder = ColorThresholder(self.screensize)
        else:
            self.is_initialized, self.fretboard = self.tracker.track(frame)
            if not self.is_initialized:
                self.is_initialized, self.fretboard = self.detector.detect(frame) 
            cv2.waitKey(1)

            # self.videoplayback.put(final_cdst)
            # self.videoplayback.imshow()
        
        final_cdst = frame.copy()
        if self.fretboard is not None:
            draw_fretboard(final_cdst, self.fretboard)

            hand_oriented_bb = (self.fretboard.oriented_bb[0], 
                                (self.fretboard.oriented_bb[1][0]+100, self.fretboard.oriented_bb[1][1]),
                                self.fretboard.oriented_bb[2])
            cropped_frame, crop_transform = crop_from_oriented_bb(frame, hand_oriented_bb)

            hands = self.hand_detector.detect(cropped_frame, crop_transform=crop_transform)
            if hands is not None:
                for i in range(hands.shape[0]):
                    for landmark in hands[i]:
                        cv2.circle(final_cdst, (landmark[0], landmark[1]), 5, (0, 0, 255), 3)
                    
        cv2.putText(final_cdst, f'{self.counter}', (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

        final_cdst, _ = resize_with_aspect(final_cdst, (540, 960))

        cv2.imshow('final_cdst', final_cdst)
        self.video.write(final_cdst)
        self.counter = self.counter + 1

def resize_with_aspect(frame, max_size):
    h, w = frame.shape[:2]
    scale = min(max_size[1] / w, max_size[0] / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale

if __name__ == '__main__':
    cap = cv2.VideoCapture('test\\guitar4.mp4')
    frame_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    loop_video = False

    from video_processing.detector import Detector, HandDetector
    from video_processing.tracker import TrackerLightGlue
    fd = FretboardDetector(detector=Detector(), tracker=TrackerLightGlue(), hand_detector=HandDetector(), frame_shape=frame_shape)

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