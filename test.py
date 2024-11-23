# from ultralytics import YOLO
# import torch
import cv2

if __name__ == '__main__':
    # Load a model
    # model = YOLO(r"yolo11n-pose.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # results = model.train(resume=True, save_period=2)

    # model = YOLO('hand-keypoints.yaml', task='pose').load('runs\weights\last.pt') # build from YAML and transfer weights
    # model.train(data="datasets\hand-keypoints\data.yaml", epochs=100, save_period=2)

    import mediapipe as mp
    import numpy as np
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



    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    

        results = media_pipe_hand(frame, hands)
        
        w = frame.shape[1]
        h = frame.shape[0]
        if results.multi_hand_landmarks:
            landmarks = []
            for idx in [4, 8, 12, 16, 20]:
                relative_x = int(results.multi_hand_landmarks[0].landmark[idx].x * w)
                relative_y = int(results.multi_hand_landmarks[0].landmark[idx].y * h)
                landmarks.append([relative_x, relative_y])
                # cv2.circle(cropped_frame, (relative_x, relative_y), 5, (0, 0, 255), 3)
            landmarks = np.array(landmarks)
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            # landmarks = transform_points(landmarks, np.linalg.inv(crop_transform)).astype(int)
            for landmark in landmarks:
                cv2.circle(frame, (landmark[0], landmark[1]), 5, (0, 0, 255), 3)


        # results = model.predict(source=frame)
        # for r in results:
        #     keypoints = r.keypoints.xy.int().cpu().numpy()
        #     if keypoints.shape[0] > 1:
        #         for hand in keypoints:
        #             for kp in hand.squeeze():
        #                 cv2.circle(frame, (kp[0], kp[1]), 3, (0, 0, 255))
        #     else:
        #         for kp in keypoints.squeeze():
        #             cv2.circle(frame, (kp[0], kp[1]), 3, (0, 0, 255))
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        counter = counter + 1