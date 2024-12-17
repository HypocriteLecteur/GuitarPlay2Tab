import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QScrollArea
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QRect, QRunnable, QObject, QThreadPool
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from pathlib import Path
import threading

import numpy as np

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
model_hands = mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence=0.5)

class WorkerSignals(QObject):
    progress = Signal(object)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()  # QtCore.Slot
    def run(self):
        self.fn(*self.args, **self.kwargs)

def media_pipe_hand(frame, model_hands):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.flip(frame, 1)
    frame.flags.writeable = False
    results = model_hands.process(frame)
    frame.flags.writeable = True
    return results

class ColoredSlider(QSlider):
    def __init__(self, orientation, *args, **kwargs):
        super().__init__(orientation, *args, **kwargs)
        self.colored_marks = {}

    def set_colored_marks(self, marks):
        """
        Set the colored marks on the slider.
        marks: dict where keys are frame numbers (slider positions), and values are QColor objects
        """
        self.colored_marks = marks
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        for position, color in self.colored_marks.items():
            x = self._position_to_slider_x(position)
            painter.setPen(QPen(color, 2))
            painter.drawLine(x, self.rect().top(), x, self.rect().top()+2)
        painter.end()

    def _position_to_slider_x(self, position):
        """
        Convert a slider position to the x-coordinate on the slider track.
        """
        range_min, range_max = self.minimum(), self.maximum()
        slider_pos = (position - range_min) / (range_max - range_min) if range_max > range_min else 0
        return int(slider_pos * (self.width()))

class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Window")
        self.setGeometry(200, 200, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll_area)
        layout.setContentsMargins(0, 0, 0, 0)
    
    def display(self, frame):
        frame = np.repeat(frame, 6, axis=0)
        if frame.ndim == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(self.pixmap)

class TabWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tab")
        self.setGeometry(200, 200, 600, 400)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll_area)
        layout.setContentsMargins(0, 0, 0, 0)
    
    def create_canvas(self, total_sec, length_per_sec):
        self.length_per_sec = length_per_sec
        width, height = int(total_sec*length_per_sec), 250
        self.image = QImage(width, height, QImage.Format.Format_RGB32)
        self.image.fill(QColor(0, 0, 0))

        self.painter = QPainter(self.image)
        self.painter.setPen(QColor(255, 255, 255))

        # draw tab line
        self.line_begin = 50
        self.line_spacing = height // 7
        for i in range(1, 7):
            y_position = i * self.line_spacing
            self.painter.drawLine(self.line_begin, y_position, width, y_position)
        
        # write EADGBE
        font = QFont("Arial", 12)
        self.painter.setFont(font)
        self.painter.setPen(QColor(255, 255, 255))

        self.text_rect_size = (20, 20)
        self.offset = (10, 10)

        for i, t in enumerate(['E', 'B', 'G', 'D', 'A', 'E']):
            text_rect = QRect(25, (i+1)*self.line_spacing-self.offset[1], *self.text_rect_size)
            self.painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, t)

        self.image_label.setPixmap(QPixmap.fromImage(self.image))
    
    def midi_to_tab(self, midi, detected_frames, fps):
        for note in midi.instruments[0].notes:
            # the open sixth string: E2 - 40 (6, note.pitch - 40)
            # the open fifth string: A2 - 45 (5, note.pitch - 45)
            # the open fourth string: D3 - 50 (4, note.pitch - 50)
            # the open third string: G3 - 55 (3, note.pitch - 55)
            # the open second string: B3 - 59 (2, note.pitch - 59)
            # the open first string: E4 - 64 (1, note.pitch - 64)

            frame_num = round(note.start * fps)
            pos_candidates = np.array([
                [1, note.pitch - 64],
                [2, note.pitch - 59],
                [3, note.pitch - 55],
                [4, note.pitch - 50],
                [5, note.pitch - 45],
                [6, note.pitch - 40]
            ])

            if detected_frames[frame_num][0].frets[0, 0] < detected_frames[frame_num][0].frets[-1, 0]:
                frets = detected_frames[frame_num][0].frets[-1::-1]
            else:
                frets = detected_frames[frame_num][0].frets
            # 8 12 16 20 tips of the index, middle, ring, pinky
            fingertips_fret = []
            for i in [8, 12, 16, 20]:
                fingertip_fret_ = np.argmin(np.abs(frets[:, 0] - np.cos(frets[:, 1])*detected_frames[frame_num][1][0][i, 0] \
                    - np.sin(frets[:, 1])*detected_frames[frame_num][1][0][i, 1]))
                fingertips_fret.append([fingertip_fret_, fingertip_fret_+1])
            fingertips_fret = np.array(fingertips_fret)

            intersection = list(set(np.unique(fingertips_fret.reshape((-1,)))) & set(pos_candidates[:, 1]))
            if len(intersection) == 1:
                idx = np.argwhere(pos_candidates[:, 1] == intersection[0])[0, 0]

                rect_center = [round(note.start * self.length_per_sec) + self.line_begin, pos_candidates[idx, 0] * self.line_spacing]
                text_rect = QRect(rect_center[0]-self.offset[0], rect_center[1]-self.offset[1], *self.text_rect_size)
                self.painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(pos_candidates[idx, 1]))
            else:
                if len(intersection) > 1:
                    pass
                    # raise NotImplementedError
                else:
                    pass
                    # raise NotImplementedError
        self.image_label.setPixmap(QPixmap.fromImage(self.image))

class VideoPlayer(QMainWindow):
    def __init__(self, detector, tracker):
        super().__init__()
        self.setup_ui()
        self.cap = None
        self.current_frame = 0
        self.original_size, self.scaled_size = None, (720, 1080)
        self.x_scale, self.y_scale = 1, 1
        self.drawing, self.drawing_enabled = False, False
        self.start_pos, self.end_pos = None, None
        self.final_box = None

        self.frame = None
        self.detector = detector
        self.tracker = tracker

        self.detected_frames = dict()

        self.model_output, self.midi_data, self.note_events = None, None, None
        
        self.tab = None

        self.threadpool = QThreadPool()

        self.cap_lock = threading.Lock()

    def setup_ui(self):
        self.setWindowTitle("PyQt6 Video Player")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget, main_layout = QWidget(), QHBoxLayout()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(main_layout)

        sidebar_layout = QVBoxLayout()
        main_layout.addLayout(sidebar_layout)
        for name, func in [("Draw BoundingBox", self.enable_drawing), 
                           ("Detect", self.detect_objects), 
                           ("Track", self.track_objects),
                           ("AMT", self.audio_to_midi),
                           ("save", self.save_data),
                           ("load", self.load_data),
                           ("show_midi", self.show_midi),
                           ("show_tab", self.show_tab),
                           ("midi_to_tab", self.midi_to_tab)]:
            button = QPushButton(name)
            button.clicked.connect(func)
            sidebar_layout.addWidget(button)
        sidebar_layout.addStretch()

        video_layout, self.video_label = QVBoxLayout(), QLabel("Load a video to display frames here")
        main_layout.addLayout(video_layout)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label)

        # Frame number display and slider
        slider_layout = QHBoxLayout()
        self.frame_number_label = QLabel(f"Frame: {0:03d}")
        self.slider = ColoredSlider(Qt.Orientation.Horizontal)
        # self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_moved)

        slider_layout.addWidget(self.frame_number_label)
        slider_layout.addWidget(self.slider)
        video_layout.addLayout(slider_layout)

        load_button = QPushButton("Load Video")
        load_button.clicked.connect(self.load_video)
        video_layout.addWidget(load_button)

        self.video_label.mousePressEvent = self.start_draw
        self.video_label.mouseMoveEvent = self.update_draw
        self.video_label.mouseReleaseEvent = self.end_draw
        self.keyPressEvent = self.finalize_box

    def load_video(self):
        # filename, _ = QFileDialog().getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        filename = str(Path('.').absolute() / 'test' / 'guitar4.mp4')
        if filename:
            self.filename = filename
            self.cap = cv2.VideoCapture(filename)
            if not self.cap.isOpened():
                self.video_label.setText("Failed to open video file.")
                return

            self.slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            self.slider.setEnabled(True)
            self.display_frame(0)

    def display_frame(self, frame_number):
        if self.cap:
            self.cap_lock.acquire()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            self.cap_lock.release()
            if not ret: return

            self.original_size = frame.shape[1], frame.shape[0]
            if len(self.detected_frames) == 0 or frame_number not in self.detected_frames:
                self.frame = frame.copy()
                frame = self.resize_with_aspect(frame, self.scaled_size)

                self.x_scale, self.y_scale = self.original_size[0] / frame.shape[1], self.original_size[1] / frame.shape[0]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_image))
                self.current_frame_frame = frame
            else:
                self.gui_draw_fretboard(frame, self.detected_frames[frame_number][0], self.detected_frames[frame_number][1])

            # Update frame number label
            self.frame_number_label.setText(f"Frame: {frame_number:03d}")

    def resize_with_aspect(self, frame, max_size):
        h, w = frame.shape[:2]
        scale = min(max_size[1] / w, max_size[0] / h)
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def slider_moved(self, position):
        self.display_frame(position)

    def enable_drawing(self):
        self.drawing_enabled = True

    def start_draw(self, event):
        if self.drawing_enabled:
            self.drawing, self.start_pos = True, (event.position().x(), event.position().y())

    def update_draw(self, event):
        if self.drawing:
            self.end_pos = (event.position().x(), event.position().y())
            self.bounding_box = (self.start_pos, self.end_pos)
            self.draw_bounding_box()

    def end_draw(self, event):
        if self.drawing:
            self.drawing = False
            self.end_pos = (event.position().x(), event.position().y())
            self.bounding_box = (self.start_pos, self.end_pos)
            self.draw_bounding_box()

    def draw_bounding_box(self):
        if self.current_frame_frame is None:
            return

        frame_copy = cv2.cvtColor(self.current_frame_frame.copy(), cv2.COLOR_BGR2RGB)
        q_image = QImage(frame_copy.data, frame_copy.shape[1], frame_copy.shape[0], frame_copy.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.GlobalColor.green, 2))
        
        (start, end) = self.bounding_box
        x, y, w, h = int(min(start[0], end[0])), int(min(start[1], end[1])), abs(int(end[0] - start[0])), abs(int(end[1] - start[1]))
        painter.drawRect(QRect(x, y, w, h))
        
        painter.end()
        self.video_label.setPixmap(pixmap)

    def finalize_box(self, event):
        if event.key() == Qt.Key.Key_Return and self.bounding_box:
            (start, end) = self.bounding_box
            x, y, w, h = int(min(start[0], end[0])), int(min(start[1], end[1])), abs(int(end[0] - start[0])), abs(int(end[1] - start[1]))
            original_box = (int(x * self.x_scale), int(y * self.y_scale), int(w * self.x_scale), int(h * self.y_scale))
            self.final_box = original_box
            self.drawing_enabled = False
    
    def gui_draw_fretboard(self, frame, fretboard, hands=None):
        frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        draw_fretboard(frame_copy, fretboard)
        if hands is not None:
            for i in range(hands.shape[0]):
                for landmark in hands[i]:
                    cv2.circle(frame_copy, (landmark[0], landmark[1]), 5, (0, 0, 255), 3)
        frame_copy = self.resize_with_aspect(frame_copy, self.scaled_size)

        self.x_scale, self.y_scale = self.original_size[0] / frame_copy.shape[1], self.original_size[1] / frame_copy.shape[0]

        q_image = QImage(frame_copy.data, frame_copy.shape[1], frame_copy.shape[0], frame_copy.strides[0], QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def detect_objects(self):
        if self.frame is not None:
            if self.final_box:
                is_initialized, fretboard = self.detector.detect(self.frame, bb=self.final_box, is_visualize=False)
            else:
                is_initialized, fretboard = self.detector.detect(self.frame, is_visualize=False)
            if is_initialized:
                hand_oriented_bb = (fretboard.oriented_bb[0], 
                                    (fretboard.oriented_bb[1][0]+200, fretboard.oriented_bb[1][1]),
                                    fretboard.oriented_bb[2])
                cropped_frame, crop_transform = crop_from_oriented_bb(self.frame, hand_oriented_bb)
                results = media_pipe_hand(cropped_frame, model_hands)
                if results.multi_hand_landmarks:
                    h, w = cropped_frame.shape[0], cropped_frame.shape[1]
                    hands = [[] for _ in range(len(results.multi_hand_landmarks))]
                    for i, hand in enumerate(results.multi_hand_landmarks):
                        for idx in range(21):
                            hands[i].append([int(hand.landmark[idx].x * w), int(hand.landmark[idx].y * h)])
                    hands = np.array(hands)
                    for i in range(len(results.multi_hand_landmarks)):
                        hands[i, ...] = transform_points(hands[i, ...], np.linalg.inv(crop_transform)).astype(int)
                    self.gui_draw_fretboard(self.frame, fretboard, hands)
                    self.detected_frames[self.current_frame] = (fretboard, hands)
                else:
                    self.gui_draw_fretboard(self.frame, fretboard)
                    self.detected_frames[self.current_frame] = (fretboard, None)
        else:
            pass
        
        if len(self.detected_frames) > 0:
            colored_frames = {k: QColor("green") for k in self.detected_frames}
            self.slider.set_colored_marks(colored_frames)

    def track_objects_(self, progress_callback):
        begin_frame = min(list(self.detected_frames.keys()))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, begin_frame)
        ret, frame = self.cap.read()
        self.tracker.create_template(frame, self.detected_frames[begin_frame][0])
        begin_frame = begin_frame + 1
        
        track_flag = True
        while track_flag:
            self.cap_lock.acquire()
            ret, frame = self.cap.read()
            self.cap_lock.release()
            if not ret:
                break

            track_flag, fretboard = self.tracker.track(frame)
            if track_flag:
                hand_oriented_bb = (fretboard.oriented_bb[0], 
                                    (fretboard.oriented_bb[1][0]+200, fretboard.oriented_bb[1][1]),
                                    fretboard.oriented_bb[2])
                cropped_frame, crop_transform = crop_from_oriented_bb(frame, hand_oriented_bb)
                results = media_pipe_hand(cropped_frame, model_hands) 
                if results.multi_hand_landmarks:
                    h, w = cropped_frame.shape[0], cropped_frame.shape[1]
                    hands = [[] for _ in range(len(results.multi_hand_landmarks))]
                    for i, hand in enumerate(results.multi_hand_landmarks):
                        for idx in range(21):
                            hands[i].append([int(hand.landmark[idx].x * w), int(hand.landmark[idx].y * h)])
                    hands = np.array(hands)
                    for i in range(len(results.multi_hand_landmarks)):
                        hands[i, ...] = transform_points(hands[i, ...], np.linalg.inv(crop_transform)).astype(int)
                    result = (fretboard, hands)
                else:
                    result = (fretboard, None)
            progress_callback.emit((begin_frame, result))
            begin_frame = begin_frame + 1

    def track_objects(self):
        if len(self.detected_frames) == 0:
            return
        if len(self.detected_frames) > 1:
            raise NotImplementedError
        else:
            # self.track_objects_()
            worker = Worker(self.track_objects_) # Any other args, kwargs are passed to the run function
            def progress_fn(msg):
                self.detected_frames[msg[0]] = msg[1]
                colored_frames = {k: QColor("green") for k in self.detected_frames}
                self.slider.set_colored_marks(colored_frames)
            worker.signals.progress.connect(progress_fn)

            self.threadpool.start(worker)
    
    def audio_to_midi(self):
        from basic_pitch_torch.inference import predict

        model_path = str(Path('.').absolute() / 'basic_pitch_torch' / 'assets' / 'basic_pitch_pytorch_icassp_2022.pth')

        # note_events: A list of note event tuples (start_time_s, end_time_s, pitch_midi, amplitude)
        self.model_output, self.midi_data, self.note_events = predict(self.filename, model_path=model_path)
    
    def show_midi(self):
        piano_roll = self.midi_data.get_piano_roll(fs=50)
        if np.max(piano_roll) > 255:
            raise NotImplementedError
        piano_roll = piano_roll.astype(np.uint8)
        self.piano_roll = ImageWindow()
        self.piano_roll.display(piano_roll)
        self.piano_roll.show()
    
    def show_tab(self):
        if self.tab is not None:
            self.tab.show()
    
    def midi_to_tab(self):
        self.tab = TabWindow()
        total_sec = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
        length_per_sec = 100
        self.tab.create_canvas(total_sec, length_per_sec)
        self.show_tab()
        self.tab.midi_to_tab(self.midi_data, self.detected_frames, self.cap.get(cv2.CAP_PROP_FPS))
    
    def save_data(self):
        import pickle
        # file, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "All Files (*)")
        filename = str(Path('.').absolute() / 'test' / 'guitar4.pickle')
        with open(filename, "wb") as f:
            pickle.dump([self.detected_frames, self.model_output, self.midi_data, self.note_events], f)
    
    def load_data(self):
        import pickle
        # filename, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "All Files (*)")
        filename = str(Path('.').absolute() / 'test' / 'guitar4.pickle')
        with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
            self.detected_frames, self.model_output, self.midi_data, self.note_events = pickle.load(f)
        
        filename = str(Path('.').absolute() / 'test' / 'guitar4.mp4')
        self.filename = filename
        self.cap = cv2.VideoCapture(filename)
        if not self.cap.isOpened():
            self.video_label.setText("Failed to open video file.")
            return

        self.slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        self.slider.setEnabled(True)
        self.display_frame(0)

if __name__ == "__main__":
    import sys
    import os

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

    from video_processing.utils.utils_math import crop_from_oriented_bb, transform_points

    from video_processing.detector import Detector
    from video_processing.tracker import TrackerLightGlue
    from video_processing.utils.visualize import draw_fretboard
    
    app = QApplication(sys.argv)
    player = VideoPlayer(detector=Detector(), tracker=TrackerLightGlue())

    player.show()
    sys.exit(app.exec())