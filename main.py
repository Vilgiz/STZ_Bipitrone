import cv2
import numpy
import cvzone
import numpy as np
import json
import asyncio
from Camera_std import Camera

asyncio

# async def


class vision_billet:

    def __init__(self) -> None:

        self.brightness_factor = 3000
        self.saturation_factor = 2000

        self.circles_min_array = []
        self.circles_max_array = []

        self.counters = []
        self.coordinates = []

        self.threshold_1 = 295
        self.threshold_2 = 1
        self.minRadius = 20
        self.maxRadius = 30

        self.activate = False

    def prepare_frames(self, frame):

        self.frame = frame
        self.original = frame
        self.WHITE_frame = cv2.imread("Prepared_Image/white.jpg")
        self.RGB_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.GRAY_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.HSV_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.painted = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        height_W, width_W, _ = self.WHITE_frame.shape
        height, width, _ = self.frame.shape
        width = (width/width_W)
        height = (height/height_W)
        self.WHITE_frame = cv2.resize(
            self.WHITE_frame, None, fx=width, fy=height)

    def detect_contours(self, frame):
        GRAY_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        WHITE_frame = cv2.imread("Prepared_Image/white.jpg")

        edges = cv2.Canny(GRAY_frame, self.threshold_1, self.threshold_2)
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
        cv2.drawContours(WHITE_frame, contours, -1,
                         (0, 0, 255), thickness=11)

        WHITE_frame = cv2.GaussianBlur(WHITE_frame, (7, 7), 0)
        cv2.imshow("ere_2", WHITE_frame)
        WHITE_frame = cv2.cvtColor(WHITE_frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(WHITE_frame, self.threshold_1, self.threshold_2)
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(WHITE_frame, contours, -1,
                         (255, 255, 255), thickness=cv2.FILLED)

        for contour in contours:

            M = cv2.moments(contour)

            # Находим центр тяжести
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            area = cv2.contourArea(contour)
            if area < 5000 or area > 10000:
                continue

            cv2.circle(self.painted, (cX, cY), 3, (0, 0, 255), -1)
            self.counters.append(contour)

            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            self.painted = cv2.drawContours(
                self.painted, [contour], -1, 255, 2)

            self.coordinates.append([cX, cY])

        print(f"Обнаружено деталей: {len(self.coordinates)}")
        print(f"Координаты: {self.coordinates}")

        cv2.imshow("ere_3", WHITE_frame)
        cv2.imshow("FINAL", frame)

    def __check_blillet(self):
        pass

    def color_correction(self, frame):

        brightened_image = cv2.convertScaleAbs(
            frame, alpha=self.brightness_factor, beta=0)

        hsv_image = cv2.cvtColor(brightened_image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = hsv_image[:, :, 1] * self.saturation_factor
        self.saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return self.saturated_image

    def show_video(self, resize_coff):
        ImgList = [self.WHITE_frame,
                   self.painted]  # TODO Параметр калибровки - ImgList
        # ImgList = [self.painted]
        # TODO Параметр калибровки - cols, scale
        stackedImg = cvzone.stackImages(ImgList, cols=2, scale=1)

        height, width, _ = self.frame.shape
        width = int(width/resize_coff)
        height = int(height/resize_coff)

        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', width, height)
        cv2.imshow("Video", stackedImg)

    def activate_sliders(self):
        cv2.createTrackbar('threshold_1', 'Video', 1, 1000, self.__pass)
        cv2.createTrackbar('threshold_2', 'Video', 1, 100, self.__pass)

        # cv2.createTrackbar('minRadius', 'Video', 1, 100, self.__pass)
        # cv2.createTrackbar('maxRadius', 'Video', 1, 100, self.__pass)

        cv2.createTrackbar('brightness_factor', 'Video', 1, 3000, self.__pass)
        cv2.createTrackbar('saturation_factor', 'Video', 1, 2000, self.__pass)

        self.activate = True

    def get_sliders(self):
        if self.activate is True:
            threshold_1 = cv2.getTrackbarPos('threshold_1', 'Video')
            threshold_2 = cv2.getTrackbarPos('threshold_2', 'Video')

            # minRadius = cv2.getTrackbarPos('minRadius', 'Video')
            # maxRadius = cv2.getTrackbarPos('maxRadius', 'Video')

            brightness_factor = cv2.getTrackbarPos(
                'brightness_factor', 'Video')
            saturation_factor = cv2.getTrackbarPos(
                'saturation_factor', 'Video')
            self.__function_sliders(
                threshold_1, threshold_2, brightness_factor, saturation_factor)  # minRadius, maxRadius)

    # minRadius, maxRadius):
    def __function_sliders(self, threshold_1, threshold_2, brightness_factor, saturation_factor):
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2

        # self.maxRadius = maxRadius
        # self.minRadius = minRadius

        self.brightness_factor = brightness_factor / 1000
        self.saturation_factor = saturation_factor / 1000

    def __pass(self, df):
        pass

    def tranform(self, frame):
        with open('transformation_data.json', 'r') as json_file:

            data = json.load(json_file)

        M = np.array(data['M'])
        maxWidth = data['maxWidth']
        maxHeight = data['maxHeight']
        frame = cv2.warpPerspective(
            frame, M, (maxWidth, maxHeight))

        with open('calibration_result.json', 'r') as json_file:
            data = json.load(json_file)

        camera_matrix = np.array(data['camera_matrix'])
        dist_coefficients = np.array(data['dist_coefficients'])

        frame = cv2.undistort(
            frame, camera_matrix, dist_coefficients)
        return frame


if __name__ == '__main__':

    vision = vision_billet()
    cam = cv2.VideoCapture("output_4.mp4")

    while True:
        ret, frame = cam.read()

        if ret is not None:
            # frame = cv2.imread("sudoku.png")

            vision.coordinates = []

            frame = vision.color_correction(frame)

            vision.prepare_frames(frame)
            vision.detect_contours(frame)

            vision.show_video(2)
            vision.get_sliders()

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('k'):
                vision.activate_sliders()
