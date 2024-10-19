from detector import DetectorInterface
from tracker import TrackerInterface

import cv2
import numpy as np
from utils.visualize import draw_fretboard

from utils.utils_math import mask_out_oriented_bb, crop_from_oriented_bb, transform_points

from typing import Tuple
from cv2.typing import MatLike

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence=0.5)

def media_pipe_hand(frame, hands):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.flip(frame, 1)
    frame.flags.writeable = False
    results = hands.process(frame)
    frame.flags.writeable = True
    return results

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

            hand_oriented_bb = (self.fretboard.oriented_bb[0], 
                                (self.fretboard.oriented_bb[1][0]+100, self.fretboard.oriented_bb[1][1]),
                                self.fretboard.oriented_bb[2])
            cropped_frame, crop_transform = crop_from_oriented_bb(frame, hand_oriented_bb)

            results = media_pipe_hand(cropped_frame, hands)
            
            w = cropped_frame.shape[1]
            h = cropped_frame.shape[0]
            if results.multi_hand_landmarks:
                landmarks = []
                for idx in [4, 8, 12, 16, 20]:
                    relative_x = int(results.multi_hand_landmarks[0].landmark[idx].x * w)
                    relative_y = int(results.multi_hand_landmarks[0].landmark[idx].y * h)
                    landmarks.append([relative_x, relative_y])
                    # cv2.circle(cropped_frame, (relative_x, relative_y), 5, (0, 0, 255), 3)
                landmarks = np.array(landmarks)
                landmarks = transform_points(landmarks, np.linalg.inv(crop_transform)).astype(int)
                for landmark in landmarks:
                    cv2.circle(final_cdst, (landmark[0], landmark[1]), 5, (0, 0, 255), 3)
                    
        cv2.putText(final_cdst, f'{self.counter}', (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
        cv2.imshow('final_cdst', final_cdst)
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