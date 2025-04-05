from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QHBoxLayout
from PyQt6.QtWidgets import QWidget, QPushButton, QFileDialog, QScrollArea, QInputDialog, QLineEdit
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QRect, QRunnable, QObject, QThreadPool
from PyQt6.QtCore import pyqtSlot as Slot

import gui_utils as gui_utils
import cv2
import numpy as np

# Dealing with Python relative import issues
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from video_processing.detector import Detector, HandDetector, DetectorStatus
from video_processing.tracker import TrackerLightGlue, Tracker

import video_processing.utils.utils_math as utils
import video_processing.utils.visualize as visualize
from video_processing.fretboard import Fretboard

from pathlib import Path
from tqdm import tqdm
import threading


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

class ColoredSlider(QSlider):
    def __init__(self, orientation, *args, **kwargs):
        super().__init__(orientation, *args, **kwargs)
        self.colored_marks = {}

    def set_colored_marks(self, key, value):
        """
        Set the colored marks on the slider.
        marks: dict where keys are frame numbers (slider positions), and values are QColor objects
        """
        self.colored_marks[key] = value
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

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()  # QtCore.Slot
    def run(self):
        self.fn(*self.args, **self.kwargs)

class VideoPlayer(QMainWindow):
    def __init__(self, detector, tracker, backup_tracker, hand_detector):
        super().__init__()
        
        self.setup_ui()

        self.is_images_cache = False # only works for short video
        self.frames_cache = None
        self.dir = None
        self.total_frames, self.frame_width, self.frame_height = None, None, None
        self.video_fps = None

        self.current_frame = None
        self.max_display_size = (720, 1080)
        self.drawing, self.drawing_enabled = False, False
        self.start_pos, self.end_pos = None, None
        self.final_box = None
        self.fretboard_boundaries = []
        self.frets = []

        self.detector = detector
        self.tracker = tracker
        self.backup_tracker = backup_tracker
        self.hand_detector = hand_detector
        self.yolo_model = None

        self.detected_frames = dict()
        self.outlier_frames = []
        self.model_output, self.midi_data = None, None

        self.bounding_box, self.line_endpnts = None, None
    
    def setup_ui(self):
        self.setWindowTitle("GuitarPlay2TAB Annotation GUI")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget, main_layout = QWidget(), QHBoxLayout()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(main_layout)

        sidebar_layout = QVBoxLayout()
        main_layout.addLayout(sidebar_layout)
        for name, func in [ 
                           ("Detect", self.detect_objects),
                           ("Detect YOLO", self.detect_fretboard_with_yolo),
                           ("Detect hand", self.detect_hand),
                           ("Track Fretboard", self.track_fretboard),
                           ("Track Fretboard YOLO", self.track_fretboard_yolo),
                           ("Track hand", self.track_hand),
                           ("Outlier Detection", self.detect_outliers),
                           ("Refine Inliers", self.refine_inliers),
                           ('Retrack Outliers', self.retrack_outliers),
                           ('Single Inspect Track', self.inspect_track),
                           ('Batch Inspect Track', self.batch_inspect_track),
                           ('Batch Delete', self.batch_delete),
                           ("AMT", self.audio_to_midi),
                           ("show_midi", self.show_midi),
                           ("show_tab", self.show_tab),
                           ("midi_to_tab", self.midi_to_tab)
                           ]:
            button = QPushButton(name)
            button.clicked.connect(func)
            sidebar_layout.addWidget(button)
        sidebar_layout.addStretch()

        video_layout = QVBoxLayout()
        self.video_label = QLabel("Load a video to display frames here")
        main_layout.addLayout(video_layout)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label)

        slider_layout = QHBoxLayout()
        self.frame_number_label = QLabel(f"Frame: {0:03d}")
        self.slider = ColoredSlider(Qt.Orientation.Horizontal)
        # self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.display_frame)

        slider_layout.addWidget(self.frame_number_label)
        slider_layout.addWidget(self.slider)
        video_layout.addLayout(slider_layout)

        save_load_layout = QHBoxLayout()
        video_layout.addLayout(save_load_layout)
        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_data)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_data)
        save_load_layout.addWidget(load_button)
        save_load_layout.addWidget(save_button)
        save_load_layout.addStretch()

        self.video_label.mousePressEvent = self.start_draw
        self.video_label.mouseMoveEvent = self.update_draw
        self.video_label.mouseReleaseEvent = self.end_draw

        self.status = self.statusBar()
        self.status.showMessage("Press '1' to draw bounding-box, '2' to draw fretboard boundaries.")

    def save_data(self):
        import pickle
        
        if self.dir is None:
            return
        if self.midi_data is not None:
            self.midi_data.write(str(self.dir / 'audio.mid'))
        filename = str(self.dir / 'vision.pickle')
        with open(filename, "wb") as f:
            pickle.dump([self.detected_frames, self.midi_data], f)
        
        gui_utils.format_and_store_json(self.detected_frames, self.total_frames, 
                                        self.frame_width, self.frame_height, 
                                        self.dir)
    
    def load_data(self):
        import glob
        import pickle
        
        dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir == '':
            return
        self.dir = Path(dir)
        # self.dir = Path("D:\\GitHub\\GuitarPlay2Tab\\video\\video5")
        print(f"Working on dir: {self.dir}")

        filename = glob.glob(str(self.dir / '*.pickle*'))
        if filename:
            with open(filename[0], 'rb') as f:
                self.detected_frames, self.midi_data = pickle.load(f)
            self.slider.colored_marks = {k: QColor("green") for k in self.detected_frames}
        
        filename = glob.glob(str(self.dir / '*.mp4'))
        if not filename:
            print(f"Video {filename} not found.")
            return
        cap = cv2.VideoCapture(filename[0])
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not os.path.isdir(str(self.dir / 'images')):
            os.mkdir(str(self.dir / 'images'))
            if not cap.isOpened():
                self.video_label.setText("Failed to open video file.")
                return
            self.convert_video_to_images(cap)
        
        if self.is_images_cache:
            cap = cv2.VideoCapture(filename[0])
            self.cache_video(cap)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.setEnabled(True)
        self.slider.setValue(0)
        self.display_frame(0)
    
    def cache_video(self, cap):
        self.frames_cache = []
        for i in tqdm(range(self.total_frames)):
            success, frame = cap.read()
            if not success:
                return
            self.frames_cache.append(frame)
        cap.release()

    def convert_video_to_images(self, cap):
        for i in tqdm(range(self.total_frames)):
            success, frame = cap.read()
            if not success:
                return
            cv2.imwrite(str(self.dir / 'images' / f'{i}.jpg'), frame)
        cap.release()
    
    def read_frame(self, frame_number):
        if self.is_images_cache:
            frame = self.frames_cache[frame_number]
        else:
            frame = cv2.imread(str(self.dir / 'images' / f'{frame_number}.jpg'))
        return frame

    def display_frame(self, frame_number):
        if self.dir is None:
            return
        self.current_frame = self.read_frame(frame_number)

        display_frame = self.current_frame.copy()
        if len(self.detected_frames) > 0 and frame_number in self.detected_frames:
            fretboard = self.detected_frames[frame_number][0]
            hands = self.detected_frames[frame_number][1]
            if fretboard is not None:
                visualize.draw_fretboard(display_frame, fretboard)
            if hands is not None:
                visualize.draw_hands(display_frame, hands)
        
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        self.resized_frame, self.display_factor = gui_utils.resize_with_aspect(frame_rgb, self.max_display_size)

        q_image = QImage(self.resized_frame.data, self.resized_frame.shape[1], 
                        self.resized_frame.shape[0], self.resized_frame.strides[0], 
                        QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

        self.frame_number_label.setText(f"Frame: {frame_number:03d}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_1:
            self.drawing_enabled = True
            self.drawing_mode = 'bb'
            self.bounding_box = None
        
        if event.key() == Qt.Key.Key_2:
            self.drawing_enabled = True
            self.drawing_mode = 'lines'
            self.fretboard_boundaries = []
        
        if event.key() == Qt.Key.Key_3:
            self.drawing_enabled = True
            self.drawing_mode = 'boundary_redraw'
        
        if event.key() == Qt.Key.Key_F:
            self.drawing_enabled = True
            self.drawing_mode = 'frets'
            self.frets = []

        if event.key() == Qt.Key.Key_Q:
            current_frame = self.slider.value()
            self.slider.setValue(current_frame-1)
        
        if event.key() == Qt.Key.Key_E:
            current_frame = self.slider.value()
            self.slider.setValue(current_frame+1)

        if event.key() == Qt.Key.Key_A:
            if len(self.outlier_frames) == 0:
                return
            current_frame = self.slider.value()
            smaller_outlier_frames = sorted(i for i in self.outlier_frames if i < current_frame)
            if len(smaller_outlier_frames) == 0:
                return
            self.slider.setValue(smaller_outlier_frames[-1])

        if event.key() == Qt.Key.Key_D:
            if len(self.outlier_frames) == 0:
                return
            current_frame = self.slider.value()
            greater_outlier_frames = sorted(i for i in self.outlier_frames if i > current_frame)
            if len(greater_outlier_frames) == 0:
                return
            self.slider.setValue(greater_outlier_frames[0])
        
        if event.key() == Qt.Key.Key_B:
            current_frame = self.slider.value()
            self.detected_frames.pop(current_frame, None)
            self.display_frame(current_frame)
        
        if event.key() == Qt.Key.Key_N:
            current_frame = self.slider.value()
            if current_frame not in self.detected_frames:
                return
            self.detected_frames[current_frame] = (self.detected_frames[current_frame][0], None)
            self.display_frame(current_frame)
        
        if event.key() == Qt.Key.Key_Return:
            if self.drawing_mode == 'bb':
                (start, end) = self.bounding_box
                x, y = min(start[0], end[0]), min(start[1], end[1])
                w, h = abs(end[0] - start[0]), abs(end[1] - start[1])
                original_box = (int(x / self.display_factor), int(y / self.display_factor), 
                                int(w / self.display_factor), int(h / self.display_factor))
                self.final_box = original_box
                self.drawing_enabled = False
            
            if self.drawing_mode == 'lines':
                if len(self.fretboard_boundaries) == 0:
                    (start, end) = self.line_endpnts
                    self.fretboard_boundaries.append([start[0] / self.display_factor, start[1] / self.display_factor, \
                                                      end[0] / self.display_factor, end[1] / self.display_factor])
                elif len(self.fretboard_boundaries) == 1:
                    (start, end) = self.line_endpnts
                    self.fretboard_boundaries.append([start[0] / self.display_factor, start[1] / self.display_factor, \
                                                      end[0] / self.display_factor, end[1] / self.display_factor])
                    self.drawing_enabled = False

            if self.drawing_mode == 'frets':
                (start, end) = self.line_endpnts
                self.frets.append([start[0] / self.display_factor, start[1] / self.display_factor, \
                                   end[0] / self.display_factor, end[1] / self.display_factor])

            if self.drawing_mode == 'boundary_redraw':
                (start, end) = self.line_endpnts
                lineP_redraw = np.array([start[0] / self.display_factor, start[1] / self.display_factor, \
                                        end[0] / self.display_factor, end[1] / self.display_factor])
                current_frame = self.slider.value()
                utils.linesP_to_houghlines(lineP_redraw.reshape((-1, 1, 4)))
                if utils.lineP_line_dist(lineP_redraw, self.detected_frames[current_frame][0].strings[0]) < \
                utils.lineP_line_dist(lineP_redraw, self.detected_frames[current_frame][0].strings[-1]):
                    self.detected_frames[current_frame][0].strings[0] = utils.linesP_to_houghlines(lineP_redraw.reshape((-1, 1, 4)))[0]
                else:
                    self.detected_frames[current_frame][0].strings[-1] = utils.linesP_to_houghlines(lineP_redraw.reshape((-1, 1, 4)))[0]
                
                self.display_frame(current_frame)

    def start_draw(self, event):
        if self.drawing_enabled:
            self.drawing = True
            self.start_pos = (event.position().x(), event.position().y())
        
        if event.button() == Qt.MouseButton.RightButton:
            current_frame = self.slider.value()
            if current_frame not in self.detected_frames:
                return
            resize_mat = np.eye(3)
            resize_mat[0, 0] = self.display_factor
            resize_mat[1, 1] = self.display_factor
            resized_frets = utils.transform_houghlines(self.detected_frames[current_frame][0].frets, np.linalg.inv(resize_mat))
            dist_to_frets = np.abs(resized_frets[:, 0] - event.position().x()*np.cos(resized_frets[:, 1]) - event.position().y()*np.sin(resized_frets[:, 1]))
            fret_ind = np.argmin(dist_to_frets)
            if dist_to_frets[fret_ind] < 10:
                self.detected_frames[current_frame][0].frets[fret_ind, :]
                self.detected_frames[current_frame][0].frets = np.delete(self.detected_frames[current_frame][0].frets, fret_ind, 0)
                oriented_bb = utils.oriented_bb_from_frets_strings(self.detected_frames[current_frame][0].frets, self.detected_frames[current_frame][0].strings)
                oriented_bb = (oriented_bb[0], (oriented_bb[1][0] + 50, oriented_bb[1][1] + 100), oriented_bb[2])
                self.detected_frames[current_frame][0].oriented_bb = oriented_bb

    def update_draw(self, event):
        if self.drawing:
            self.end_pos = (event.position().x(), event.position().y())

            if self.drawing_mode == 'bb':
                self.bounding_box = (self.start_pos, self.end_pos)
                self.draw_bounding_box()
            elif self.drawing_mode == 'lines' or self.drawing_mode == 'frets' or self.drawing_mode == 'boundary_redraw':
                self.line_endpnts = (self.start_pos, self.end_pos)
                self.draw_line()

    def end_draw(self, event):
        if self.drawing:
            self.drawing = False
            self.end_pos = (event.position().x(), event.position().y())

            if self.drawing_mode == 'bb':
                self.bounding_box = (self.start_pos, self.end_pos)
                self.draw_bounding_box()
            elif self.drawing_mode == 'lines' or self.drawing_mode == 'frets' or self.drawing_mode == 'boundary_redraw':
                self.line_endpnts = (self.start_pos, self.end_pos)
                self.draw_line()

    def draw_bounding_box(self):
        if self.resized_frame is None:
            return

        frame_copy = self.resized_frame.copy()
        q_image = QImage(frame_copy.data, frame_copy.shape[1], frame_copy.shape[0], frame_copy.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.GlobalColor.green, 2))
        
        (start, end) = self.bounding_box
        x, y = int(min(start[0], end[0])), int(min(start[1], end[1]))
        w, h = int(abs(end[0] - start[0])), int(abs(end[1] - start[1]))
        painter.drawRect(QRect(x, y, w, h))
        
        painter.end()
        self.video_label.setPixmap(pixmap)

    def draw_line(self):
        if self.resized_frame is None:
            return
        
        frame_copy = self.resized_frame.copy()
        q_image = QImage(frame_copy.data, frame_copy.shape[1], frame_copy.shape[0], frame_copy.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.GlobalColor.green, 2))
        painter.drawLine(int(self.line_endpnts[0][0]), int(self.line_endpnts[0][1]), int(self.line_endpnts[1][0]), int(self.line_endpnts[1][1]))
        if len(self.fretboard_boundaries) > 0:
            for line in self.fretboard_boundaries:
                painter.drawLine(int(line[0] * self.display_factor), int(line[1] * self.display_factor), 
                                int(line[2] * self.display_factor), int(line[3] * self.display_factor))
        if len(self.frets) > 0:
            for line in self.frets:
                painter.drawLine(int(line[0] * self.display_factor), int(line[1] * self.display_factor), 
                                int(line[2] * self.display_factor), int(line[3] * self.display_factor))
        painter.end()
        self.video_label.setPixmap(pixmap)
    
    # -------------------------------------------------------------------------
    # Implementation for the buttons
    def detect_objects(self):
        if self.current_frame is None:
            return
        if self.final_box:
            status, fretboard = self.detector.detect(self.current_frame, bb=self.final_box, is_visualize=False)
        elif len(self.frets) > 0:
            if len(self.fretboard_boundaries) == 0:
                current_frame = self.slider.value()
                if current_frame in self.detected_frames:
                    frets = utils.linesP_to_houghlines(np.expand_dims(np.array(self.frets), axis=1))
                    concat_frets = np.vstack((frets, self.detected_frames[current_frame][0].frets))
                    concat_frets = concat_frets[np.argsort(concat_frets[:, 0])[::-1]]
                    oriented_bb = utils.oriented_bb_from_frets_strings(concat_frets, self.detected_frames[current_frame][0].strings)
                    oriented_bb = (oriented_bb[0], (oriented_bb[1][0] + 50, oriented_bb[1][1] + 100), oriented_bb[2])
                    fretboard = Fretboard(concat_frets, self.detected_frames[current_frame][0].strings, oriented_bb)
                    status = DetectorStatus.success
                else:
                    return
            else:
                frets = utils.linesP_to_houghlines(np.expand_dims(np.array(self.frets), axis=1))
                strings = utils.linesP_to_houghlines(np.expand_dims(np.array(self.fretboard_boundaries), axis=1))
                oriented_bb = utils.oriented_bb_from_frets_strings(frets, strings)
                oriented_bb = (oriented_bb[0], (oriented_bb[1][0] + 50, oriented_bb[1][1] + 100), oriented_bb[2])
                fretboard = Fretboard(frets, strings, oriented_bb)
                status = DetectorStatus.success
        elif len(self.fretboard_boundaries) == 2:
            print(f"Detecting with user-defined fretboard_boundaries: {self.fretboard_boundaries}")
            status, fretboard = self.detector.detect(self.current_frame, fretboard_boundaries=self.fretboard_boundaries, is_visualize=False)
        else:
            status, fretboard = self.detector.detect(self.current_frame, is_visualize=False)
        print(status)
        if fretboard is not None:
            hands = self.hand_detector.detect(self.current_frame, fretboard=fretboard)
            current_frame = self.slider.value()
            if hands is not None:
                self.detected_frames[current_frame] = (fretboard, hands)
            else:
                self.detected_frames[current_frame] = (fretboard, None)
            self.slider.set_colored_marks(current_frame, QColor("green"))
            self.display_frame(current_frame)

    def detect_hand(self):
        if self.current_frame is None:
            return
        if self.final_box is None:
            hands = self.hand_detector.detect(self.current_frame)
        else:
            hands = self.hand_detector.detect(self.current_frame, bb=self.final_box)
        current_frame = self.slider.value()
        if current_frame in self.detected_frames:
            tmp = self.detected_frames[current_frame]
            if hands is not None:
                self.detected_frames[current_frame] = (tmp[0], hands)
        else:
            if hands is not None:
                self.detected_frames[current_frame] = (None, hands)
        self.display_frame(current_frame)

    def detect_fretboard_with_yolo(self):
        if self.current_frame is None:
            return
        if self.yolo_model is None:
            from ultralytics import YOLO
            model_path = "D:\\GitHub\\GuitarPlay2Tab\\GuitarPlay2Tab\\runs\\pose\\train\\weights\\last.pt"
            self.yolo_model = YOLO(model_path)
        results = self.yolo_model.predict(source=self.current_frame, show=False, save=False)
        
        kps = results[0].keypoints.to('cpu').data.numpy()[0]
        fretboard = utils.fretboard_from_yolo(kps)
        current_frame = self.slider.value()
        if current_frame in self.detected_frames:
            hands = self.detected_frames[current_frame][1]
        else:
            hands = None
        self.detected_frames[current_frame] = (fretboard, hands)
        self.display_frame(current_frame)

    def track_fretboard_(self):
        begin_frame = self.slider.value()
        frame = self.read_frame(begin_frame)
        self.tracker.create_template(frame, self.detected_frames[begin_frame])

        text, okPressed = QInputDialog.getText(self, "Get Text", "Enter end frame:")
        if okPressed:
            end_frame = int(text)
        else:
            end_frame = self.total_frames
        
        # begin_frame = max(list(self.detected_frames.keys()))+1
        begin_frame = begin_frame + 1
        
        self.track_flag = True
        try:
            # from cProfile import Profile
            # import pstats
            # with Profile() as profile:
                # while self.track_flag:
            for i in tqdm(range(begin_frame, end_frame)):
                frame = self.read_frame(i)

                # self.track_flag, fretboard = self.backup_tracker.track(frame, self.detected_frames[i-1][0], is_track_strings=True, is_visualize=False)
                self.track_flag, fretboard = self.tracker.track(frame, self.detected_frames[i-1][0], is_visualize=False)
                if self.track_flag:
                    # hand_oriented_bb = (fretboard.oriented_bb[0], 
                    #                     (fretboard.oriented_bb[1][0]+250, fretboard.oriented_bb[1][1]),
                    #                     fretboard.oriented_bb[2])
                    # cropped_frame, crop_transform = crop_from_oriented_bb(frame, hand_oriented_bb)
                    
                    # prev_hands = self.detected_frames[begin_frame-1][1]
                    # hands = self.hand_detector.detect(cropped_frame, fretboard=fretboard, crop_transform=crop_transform, prev_hands=prev_hands)
                    if i in self.detected_frames:
                        hands = self.detected_frames[i][1]
                    else:
                        hands = None
                    result = (fretboard, hands)
                else:
                    break
                self.detected_frames[i] = result
                self.slider.setValue(begin_frame)
                self.slider.set_colored_marks(i, QColor("green"))
            # results = pstats.Stats(profile)
            # results.sort_stats(pstats.SortKey.TIME)
            # results.dump_stats('results.prof')
        except Exception as e:
            print(e)

    def track_fretboard(self):
        if len(self.detected_frames) == 0:
            return
        # self.worker = Worker(self.track_objects_) # Any other args, kwargs are passed to the run function
        # self.threadpool.start(self.worker)
        self.track_fretboard_()
    
    def track_fretboard_yolo_(self):
        if self.yolo_model is None:
            from ultralytics import YOLO
            model_path = "D:\\GitHub\\GuitarPlay2Tab\\GuitarPlay2Tab\\runs\\pose\\train\\weights\\last.pt"
            self.yolo_model = YOLO(model_path)
        self.track_flag = True
        try:
            begin_frame = self.slider.value() + 1
            text, okPressed = QInputDialog.getText(self, "Get Text", "Enter end frame:")
            if okPressed:
                end_frame = int(text)
            else:
                end_frame = self.total_frames
            for i in tqdm(range(begin_frame, end_frame)):
                frame = self.read_frame(i)
                results = self.yolo_model.predict(source=frame, show=False, save=False, verbose=False)
                kps = results[0].keypoints.to('cpu').data.numpy()[0]
                fretboard = utils.fretboard_from_yolo(kps)
                self.detected_frames[i] = (fretboard, None)
        except Exception as e:
            print(e)

    def track_fretboard_yolo(self):
        self.track_fretboard_yolo_()
    
    def track_hand_(self):
        try:
            begin_frame = self.slider.value() + 1
            text, okPressed = QInputDialog.getText(self, "Get Text", "Enter end frame:")
            if okPressed:
                end_frame = int(text)
            else:
                end_frame = self.total_frames
            for i in tqdm(range(begin_frame, end_frame)):
                frame = self.read_frame(i)
                if self.final_box is not None:
                    hands = self.hand_detector.detect(frame, bb=self.final_box)
                else:
                    if i-1 not in self.detected_frames:
                        hands = self.hand_detector.detect(frame)
                    else:
                        prev_hands = self.detected_frames[i-1][1]
                        if prev_hands is None:
                            if i not in self.detected_frames:
                                hands = self.hand_detector.detect(frame)
                            else:
                                now_fretboard = self.detected_frames[i][0]
                                hands = self.hand_detector.detect(frame, fretboard=now_fretboard)
                        else:
                            hands = self.hand_detector.detect(frame, prev_hands=prev_hands)
                if hands is None:
                    break
                if i not in self.detected_frames:
                    self.detected_frames[i] = (None, hands)
                else:
                    self.detected_frames[i] = (self.detected_frames[i][0], hands)
        except Exception as e:
            print(e)

    def track_hand(self):
        self.track_hand_()
    
    def refine_inliers(self):
        begin_frame = min(list(self.detected_frames.keys())) + 1
        try:
            for i in tqdm(range(begin_frame, self.total_frames)):
                if i in self.outlier_frames or i-1 in self.outlier_frames:
                    continue
                if i-1 not in self.detected_frames or i not in self.detected_frames:
                    continue
                frame = self.read_frame(i)

                fretboard = self.detected_frames[i][0].copy()
                fretboard.oriented_bb = self.detected_frames[i-1][0].oriented_bb
                track_flag, fretboard = self.backup_tracker.track(frame, fretboard, is_track_strings=False, is_visualize=False)
                if not track_flag:
                    continue
                self.detected_frames[i] = (fretboard, self.detected_frames[i][1])
        except Exception as e:
            print(e)

    def batch_inspect_track(self):
        try:
            begin_frame = self.slider.value()
            frame = self.read_frame(begin_frame)
            self.tracker.create_template(frame, self.detected_frames[begin_frame])

            text, okPressed = QInputDialog.getText(self, "Get Text", "Enter end frame:")
            if okPressed:
                end_frame = int(text)
            else:
                end_frame = self.total_frames
            begin_frame = self.slider.value()+1
            for i in tqdm(range(begin_frame, end_frame)):
                begin_frame = i
                frame = self.read_frame(begin_frame)
                if begin_frame-1 not in self.detected_frames:
                    return
                fretboard = self.detected_frames[begin_frame][0].copy()
                fretboard.oriented_bb = self.detected_frames[begin_frame-1][0].oriented_bb
                fretboard.frets = self.detected_frames[begin_frame-1][0].frets
                track_flag, fretboard = self.backup_tracker.track(frame, fretboard, is_track_strings=False, is_visualize=True)

                if track_flag:
                    # hand_oriented_bb = (fretboard.oriented_bb[0], 
                    #                     (fretboard.oriented_bb[1][0]+200, fretboard.oriented_bb[1][1]),
                    #                     fretboard.oriented_bb[2])
                    # cropped_frame, crop_transform = crop_from_oriented_bb(frame, hand_oriented_bb)
                    
                    # prev_hands = self.detected_frames[begin_frame-1][1]
                    # hands = self.hand_detector.detect(cropped_frame, fretboard=fretboard, crop_transform=crop_transform, prev_hands=prev_hands)
                    hands = None
                    if hands is not None:
                        result = (fretboard, hands)
                    else:
                        result = (fretboard, None)
                    self.detected_frames[begin_frame] = result
                    self.display_frame(begin_frame)
        except Exception as e:
            print(e)

    def inspect_track(self):
        try:
            begin_frame = min(list(self.detected_frames.keys()))
            frame = self.read_frame(begin_frame)
            self.tracker.create_template(frame, self.detected_frames[begin_frame])
            begin_frame = self.slider.value()
            frame = self.read_frame(begin_frame)
            if begin_frame-1 not in self.detected_frames:
                return
            fretboard = self.detected_frames[begin_frame][0].copy()
            fretboard.oriented_bb = self.detected_frames[begin_frame-1][0].oriented_bb
            fretboard.frets = self.detected_frames[begin_frame-1][0].frets
            track_flag, fretboard = self.backup_tracker.track(frame, fretboard, is_track_strings=False, is_visualize=True)

            if track_flag:
                # hand_oriented_bb = (fretboard.oriented_bb[0], 
                #                     (fretboard.oriented_bb[1][0]+200, fretboard.oriented_bb[1][1]),
                #                     fretboard.oriented_bb[2])
                # cropped_frame, crop_transform = crop_from_oriented_bb(frame, hand_oriented_bb)
                
                # prev_hands = self.detected_frames[begin_frame-1][1]
                # hands = self.hand_detector.detect(cropped_frame, fretboard=fretboard, crop_transform=crop_transform, prev_hands=prev_hands)
                if begin_frame in self.detected_frames:
                    hands = self.detected_frames[begin_frame][1]
                else:
                    hands = None
                result = (fretboard, hands)
                self.detected_frames[begin_frame] = result
                self.display_frame(begin_frame)
        except Exception as e:
            print(e)

    def check_outlier(self, key):
        if key-1 not in self.detected_frames:
            return False
        fretboard_prev = self.detected_frames[key-1][0]
        fretboard_now = self.detected_frames[key][0]
        angle_diff = utils.angle_diff_np(fretboard_now.frets[:, 1], fretboard_prev.frets[:, 1])*180/np.pi
        now_pnt1 = utils.line_line_intersection(fretboard_now.frets[0], fretboard_now.strings[0])
        now_pnt2 = utils.line_line_intersection(fretboard_now.frets[-1], fretboard_now.strings[0])
        # now_scale = np.linalg.norm(now_pnt1-now_pnt2)
        prev_pnt1 = utils.line_line_intersection(fretboard_prev.frets[0], fretboard_prev.strings[0])
        prev_pnt2 = utils.line_line_intersection(fretboard_prev.frets[-1], fretboard_prev.strings[0])
        # prev_scale = np.linalg.norm(prev_pnt1-prev_pnt2)
        # scale_diff = np.abs(now_scale - prev_scale) / prev_scale
        pos_diff = np.max((np.linalg.norm(now_pnt1-prev_pnt1), np.linalg.norm(now_pnt2-prev_pnt2)))
        if np.max(angle_diff) > 7 or pos_diff > 15:
            return True
        else:
            return False

    def detect_outliers(self):
        self.outlier_frames = []
        self.slider.colored_marks = {k: QColor("green") for k in self.detected_frames}

        keys = list(self.detected_frames.keys())
        for key in keys[1:]:
            is_outlier = self.check_outlier(key)
            if is_outlier:
                self.outlier_frames.append(key)
                self.slider.set_colored_marks(key, QColor('Red'))
        print(f'a total of {len(self.outlier_frames)} outliers')
    
    def retrack_outliers(self):
        while len(self.outlier_frames) > 0:
            key = self.outlier_frames.pop(0)
            self.slider.setValue(key)

            begin_frame = self.slider.value()
            frame = self.read_frame(begin_frame)
            if begin_frame-1 not in self.detected_frames:
                continue
            fretboard = self.detected_frames[begin_frame-1][0].copy()
            fretboard.strings = self.detected_frames[begin_frame][0].strings
            track_flag, fretboard = self.backup_tracker.track(frame, fretboard, is_track_strings=False, is_visualize=False)
            if not track_flag:
                continue
            # track_flag, fretboard = self.backup_tracker.track(frame, self.detected_frames[begin_frame-1][0], is_visualize=True)
            self.detected_frames[begin_frame][0].frets = fretboard.frets
            self.detected_frames[begin_frame][0].oriented_bb = fretboard.oriented_bb
            # if self.check_outlier(begin_frame+1):
            #     self.outlier_frames.insert(0, begin_frame+1)
        self.detect_outliers()
    
    def batch_delete(self):
        text, okPressed = QInputDialog.getText(self, "Get Text", "Enter range:")
        if okPressed:
            twonumbers = text.split('-')
            if twonumbers[0] == '':
                start = 0
            else:
                start = int(twonumbers[0])
            if twonumbers[1] == '':
                end = self.total_frames
            else:
                end = int(twonumbers[1])
            for i in range(start, end):
                self.detected_frames.pop(i, None)


    def audio_to_midi(self):
        from basic_pitch_torch.inference import predict

        model_path = str(Path('.').absolute() / 'basic_pitch_torch' / 'assets' / 'basic_pitch_pytorch_icassp_2022.pth')

        # note_events: A list of note event tuples (start_time_s, end_time_s, pitch_midi, amplitude)
        import glob
        self.model_output, self.midi_data, _ = predict(glob.glob(str(self.dir / '*.mp4'))[0], model_path=model_path)
        print('Finished Processing')
    
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
        assert('broken implementation')
        self.tab = TabWindow()
        total_sec = self.total_frames / self.video_fps
        length_per_sec = 100
        self.tab.create_canvas(total_sec, length_per_sec)
        self.show_tab()
        self.tab.midi_to_tab(self.midi_data, self.detected_frames, self.video_fps)
    

if __name__ == "__main__":
    # Dealing with Python relative import issues
    import sys
    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    
    app = QApplication(sys.argv)
    player = VideoPlayer(detector=Detector(), tracker=TrackerLightGlue(), 
                         backup_tracker=Tracker(), hand_detector=HandDetector())

    player.show()
    sys.exit(app.exec())