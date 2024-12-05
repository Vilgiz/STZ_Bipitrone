import sys
from PyQt6.QtCore import QTimer, Qt, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider
import cv2
import json
from image import Image
from robot import Robot
from camera import Camera
import threading
import asyncio


class FrameProcessor(QObject):
    # Signals to emit processed frame and data
    frame_processed = pyqtSignal(QImage, int)
    robot_message_ready = pyqtSignal(str)

    def __init__(self, camera, parameters):
        super().__init__()
        self.camera = camera
        self.parameters = parameters
        self.running = True
        self.num_of_frame = 0

    def process_frames(self):
        while self.running:
            frame = self.camera.get_image()
            frame = 255 - frame
            image = Image(frame)

            # Set parameters
            image.brightness_factor = self.parameters['brigh']
            image.threshold_3 = self.parameters['threshold_3']
            image.threshold_2 = self.parameters['threshold_2']
            image.dilate = self.parameters['dilate']
            image.blur = self.parameters['blur']

            # Image processing
            frame = image.transform_zone(frame)
            frame = image.image_correction(frame)
            frame, coordinates, orientation = image.detect_contours(frame)
            frame = image.draw_contours(frame)
            frame = cv2.resize(frame, None, fx=2, fy=2,
                               interpolation=cv2.INTER_AREA)
            height_frame, width_frame = frame.shape

            # Convert frame to QImage
            bytes_per_line = width_frame
            q_image = QImage(frame.data, width_frame, height_frame,
                             bytes_per_line, QImage.Format.Format_Grayscale8)

            # Emit the processed frame
            self.frame_processed.emit(q_image, len(coordinates))
            self.num_of_frame += 1

            # Robot communication
            if coordinates and self.num_of_frame == 20:
                self.num_of_frame = 0
                message = ";".join(("move", str(coordinates[0][1]), str(
                    coordinates[0][0]), str(orientation[0])))
                self.robot_message_ready.emit(message)

    def stop(self):
        self.running = False


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera = Camera()

        self.robot = Robot('127.0.0.1', 48569)

        self.width_frame = None
        self.height_frame = None

        self.start_flag = True

        self.__init_main_window()
        self.__init_layouts()
        self.__init_widgets()
        self.__init_style()
        self.__init_sizes()
        self.__addition_widgets()
        self.__setting_layers()
        self.__settings()

        # Start frame processor thread
        self.start_frame_processing()

    def __init_main_window(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle("Video")
        self.resize(600, 600)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 150);")

    def __init_widgets(self):
        self.video_label = QLabel()

        self.brigh_fac_slider = QSlider(Qt.Orientation.Horizontal)
        self.sat_fac_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_2_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_3_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.dilate_slider = QSlider(Qt.Orientation.Horizontal)

        self.brigh_fac_label = QLabel("Настройка яркости")
        self.sat_fac_label = QLabel("Настройка насыщения")
        self.threshold_2_label = QLabel(
            "Настройка чувствительность обнаружения № 2")
        self.threshold_3_label = QLabel(
            "Настройка чувствительность обнаружения № 3")
        self.blur_label = QLabel("Настройка размытия")
        self.dilate_label = QLabel("Настройка заполнения")

        self.detect_detail_label = QLabel("Обнаружено деталей:")

    def __init_layouts(self):
        self.main_layout = QHBoxLayout(self.central_widget)
        self.video_and_info_layout = QHBoxLayout()
        self.video_layout = QVBoxLayout()
        self.info_layout = QVBoxLayout()
        self.vid_info_and_dop_layout = QVBoxLayout()
        self.dop_layout = QHBoxLayout()
        self.calibration_layout = QVBoxLayout()

    def __addition_widgets(self):
        self.video_layout.addWidget(self.video_label)

        self.calibration_layout.addWidget(self.brigh_fac_label)
        self.calibration_layout.addWidget(self.brigh_fac_slider)
        self.calibration_layout.addWidget(self.sat_fac_label)
        self.calibration_layout.addWidget(self.sat_fac_slider)
        self.calibration_layout.addWidget(self.threshold_2_label)
        self.calibration_layout.addWidget(self.threshold_2_slider)
        self.calibration_layout.addWidget(self.threshold_3_label)
        self.calibration_layout.addWidget(self.threshold_3_slider)
        self.calibration_layout.addWidget(self.blur_label)
        self.calibration_layout.addWidget(self.blur_slider)
        self.calibration_layout.addWidget(self.dilate_label)
        self.calibration_layout.addWidget(self.dilate_slider)

        self.info_layout.addWidget(self.detect_detail_label)

    def __setting_layers(self):
        self.video_and_info_layout.addLayout(self.video_layout)
        self.video_and_info_layout.addLayout(self.info_layout)
        self.vid_info_and_dop_layout.addLayout(self.video_and_info_layout)
        self.vid_info_and_dop_layout.addLayout(self.dop_layout)
        self.main_layout.addLayout(self.vid_info_and_dop_layout)
        self.main_layout.addLayout(self.calibration_layout)

    def __settings(self):
        self.slider_label_mapping = {
            self.brigh_fac_slider: self.brigh_fac_label,
            self.sat_fac_slider: self.sat_fac_label,
            self.threshold_2_slider: self.threshold_2_label,
            self.threshold_3_slider: self.threshold_3_label,
            self.blur_slider: self.blur_label,
            self.dilate_slider: self.dilate_label
        }

        for slider, label in self.slider_label_mapping.items():
            slider.setMinimum(0)
            slider.setMaximum(255)
            slider.valueChanged.connect(self.on_slider_value_changed)

        if self.start_flag:
            with open('video_parametrs.json', 'r') as json_file:
                try:
                    data = json.load(json_file)
                except json.JSONDecodeError:
                    raise ValueError("Ошибка при чтении файла с данными.")

        self.parameters = {
            'brigh': data.get('brigh', []),
            'sat': data.get('sat', []),
            'threshold_3': data.get('threshold_3', []),
            'threshold_2': data.get('threshold_2', []),
            'blur': data.get('blur', []),
            'dilate': data.get('dilate', [])
        }

        self.brigh_fac_slider.setValue(int(self.parameters['brigh'] * 255 / 3))
        self.sat_fac_slider.setValue(self.parameters['sat'])
        self.threshold_3_slider.setValue(self.parameters['threshold_3'])
        self.threshold_2_slider.setValue(self.parameters['threshold_2'])
        self.blur_slider.setValue(self.parameters['blur'])
        self.dilate_slider.setValue(self.parameters['dilate'])
        self.start_flag = False

    def __init_style(self):
        pass

    def __init_sizes(self):
        frame = self.camera.get_image()
        frame = 255 - frame
        image = Image(frame)
        frame = image.transform_zone(frame)
        frame = image.transform_chees(frame)
        frame = image.image_correction(frame)
        self.height_frame, self.width_frame = frame.shape
        self.video_label.setMinimumSize(
            int(self.width_frame * 2), int(self.height_frame * 2))

    def start_frame_processing(self):
        # Create a thread and a worker object
        self.thread = QThread()
        self.worker = FrameProcessor(self.camera, self.parameters)
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.process_frames)
        self.worker.frame_processed.connect(self.update_frame)
        self.worker.robot_message_ready.connect(self.robot_communication)

        # Start the thread
        self.thread.start()

    def update_frame(self, q_image, num_details):
        # Update the video label
        self.video_label.setPixmap(QPixmap.fromImage(q_image))
        self.detect_detail_label.setText(f"Обнаружено деталей: {num_details}")

    def robot_communication(self, message):
        print(message)
        self.robot.send_message(message)

    def closeEvent(self, event):
        # Stop the worker and thread properly
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.robot.close_socket()
        event.accept()

    def on_slider_value_changed(self, value):
        if self.start_flag:
            return
        sender_slider = self.sender()

        corresponding_label = self.slider_label_mapping.get(sender_slider)

        if sender_slider == self.brigh_fac_slider:
            self.parameters['brigh'] = value / 255 * 3
            description = "Яркость"
        elif sender_slider == self.threshold_3_slider:
            self.parameters['threshold_3'] = value
            description = "Настройка чувствительность обнаружения № 3"
        elif sender_slider == self.threshold_2_slider:
            self.parameters['threshold_2'] = value + 1
            description = "Настройка чувствительность обнаружения № 2"
        elif sender_slider == self.blur_slider:
            self.parameters['blur'] = value
            description = "Настройка размытия"
        elif sender_slider == self.dilate_slider:
            self.parameters['dilate'] = value
            description = "Настройка заполнения"

        if corresponding_label is not None:
            corresponding_label.setText(f"{description}: {value}")

        # Save parameters to JSON
        with open('video_parametrs.json', 'w') as json_file:
            json.dump(self.parameters, json_file)

        # Update worker parameters
        self.worker.parameters = self.parameters


def run_server(player):
    asyncio.run(player.robot.start_server())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()

    player.show()

    server_thread = threading.Thread(target=run_server, args=(player,))
    server_thread.start()

    sys.exit(app.exec())
