import json
import cv2
import numpy as np
from part import Part


class Image:

    def __init__(self, frame):
        self.brightness_factor = 2.0
        self.threshold_3 = 17
        self.threshold_2 = 65
        self.blur = 1
        self.dilate = 9

        self.coordinates = []
        self.counters = []

    def transform_zone(self, frame: np.ndarray) -> np.ndarray:
        """
        Преобразует изображение с использованием матрицы трансформации.

        Args:
            frame (np.ndarray): Входное изображение в формате NumPy.

        Returns:
            np.ndarray: Преобразованное изображение.
        """
        with open('transformation_data.json', 'r') as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError:
                raise ValueError(
                    "Ошибка при чтении файла с данными трансформации.")

        M = np.array(data.get('M', []))
        maxWidth, maxHeight = data.get('maxWidth', 0), data.get('maxHeight', 0)

        return cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

    def transform_chees(self, frame: np.ndarray) -> np.ndarray:
        """
        Исправляет искажения в изображении, основываясь на данных калибровки.

        Args:
            frame (np.ndarray): Входное изображение в формате NumPy.

        Returns:
            np.ndarray: Исправленное изображение.
        """
        with open('calibration_result.json', 'r') as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError:
                raise ValueError(
                    "Ошибка при чтении файла с результатами калибровки.")

        camera_matrix = np.array(data.get('camera_matrix', []))
        dist_coefficients = np.array(data.get('dist_coefficients', []))

        return cv2.undistort(frame, camera_matrix, dist_coefficients)

    def image_correction(self, frame):
        frame = cv2.convertScaleAbs(
            frame, alpha=self.brightness_factor, beta=0)

        if self.blur % 2 == 0:
            self.blur = self.blur + 1
        if self.threshold_2 % 2 == 0:
            self.threshold_2 = self.threshold_2 + 1
        if self.threshold_3 == 1:
            self.threshold_3 = self.threshold_3 + 2

        frame = cv2.medianBlur(frame, 3)

        return frame

    def detect_contours(self, frame):
        # if frame is None or len(frame.shape) != 3:
        #     print("Ошибка: некорректное изображение!")
        #     print(len(frame.shape))
        #     return frame, [], []

        self.centers = []
        self.angels = []
        # GRAY_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        H, W = frame.shape
        width, height = int(W), int(H)
        WHITE_frame = np.ones((height, width, 3), np.uint8) * 0
        olny_white = WHITE_frame.copy()

        thead_2 = cv2.adaptiveThreshold(
            frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, self.threshold_2, self.threshold_3)

        thead_2 = cv2.medianBlur(thead_2, self.blur)
        thead_2 = cv2.equalizeHist(thead_2)

        kernel = np.ones((self.dilate, self.dilate), np.uint8)
        thead_2 = cv2.dilate(thead_2, kernel, iterations=1)

        thead_2 = cv2.erode(thead_2, kernel, iterations=1)
        thead_2 = cv2.morphologyEx(thead_2, cv2.MORPH_CLOSE, kernel)

        self.contours_3, hierarchy = cv2.findContours(
            thead_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(WHITE_frame, self.contours_3, -1, (255, 255, 255), 1)

        self.parts = []
        if self.contours_3 != ():
            for contour in self.contours_3:
                M = cv2.moments(contour)

                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                area = cv2.contourArea(contour)
                if 50 < area < 400 and cY > 25:

                    self.centers.append((cX, cY))
                    self.counters.append(contour)

                    cv2.drawContours(olny_white, [contour], -1, (0, 255, 0), 1)

                    roi, angle = self.orientation_detection(
                        olny_white, contour)

                    if angle == "above":
                        cv2.circle(olny_white, (cX, cY), 2, (0, 0, 255), -1)
                    elif angle == "under":
                        cv2.circle(olny_white, (cX, cY), 2, (255, 0, 0), -1)

                    # cv2.putText(frame, f"Area: {area}", (cX - 20, cY - 20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                    self.angels.append(angle)
                    self.coordinates.append([cX, cY])

                    self.part_type_definition(
                        cX, cY, angle, area, number=len(self.coordinates))

                    cv2.imshow("ROI", roi)
        # print(len(self.parts))
        cv2.imshow('result', WHITE_frame)
        cv2.imshow("Video", thead_2)
        cv2.imshow("Video_2", frame)
        cv2.imshow("result_contour", olny_white)

        return frame, self.coordinates, self.angels

    def draw_contours(self, frame):
        cv2.drawContours(frame, self.contours_3, -1, (0, 255, 0), 1)
        for center, contour, angle in zip(self.coordinates, self.counters, self.angels):
            if angle == "above":
                cv2.circle(frame, tuple(center), 2, (0, 0, 255), -1)
            elif angle == "under":
                cv2.circle(frame, tuple(center), 2, (255, 0, 0), -1)
        return frame

    def prepare_frames(self, frame):
        self.frame = frame
        self.RGB_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.GRAY_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.HSV_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.painted = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def part_type_definition(self, cX, cY, angle, area, number):
        number_type = "0"
        if 230 < area < 265:
            number_type = "4_5"
        elif 110 < area < 180:
            number_type = "3_4"
        elif 60 < area < 100:
            number_type = "1"

        part = Part(cX, cY, angle, area, number, number_type)
        self.parts.append(part)

    def orientation_detection(self, frame, contour):
        dist_result = [0, 0, 0]
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0 and h > 0:
            cv2.rectangle(frame, (x-5, y-5),
                          (x+w+5, y+h+5), (255, 0, 0), 1)
            if x <= 0:
                x = 5
            if y <= 0:
                y = 5
            roi = frame[y-5:y+5+h, x-5:x+w+5]
            height, width, _ = roi.shape
            if roi is None or roi.size == 0:
                roi = np.ones((800, 400, 3), dtype=np.uint8) * 255
            else:
                roi = cv2.resize(roi, (int(width*10), int(height*10)))
            height, width, _ = roi.shape
            coord = []

            for y in range(0, height, 15):
                for x in range(0, width, 1):
                    color = roi[y, x]
                    if color[1] > 240:
                        coord.append([x, y])
                        if len(coord) == 2:
                            dist = coord[1][0] - coord[0][0]
                            if coord[1][1] != coord[0][1]:
                                coord = []
                            elif dist < 50:
                                coord.pop(1)
                            else:
                                cv2.line(
                                    roi, coord[0], coord[1], (255, 255, 255), 1)
                                cv2.circle(
                                    roi, coord[0], 2, (0, 0, 255), -1)
                                cv2.circle(
                                    roi, coord[1], 2, (0, 0, 255), -1)
                                if dist > dist_result[0]:
                                    dist_result = [dist, coord[0], coord[1]]
                                coord = []
        if dist_result[0] != 0:
            cv2.line(roi, dist_result[1], dist_result[2], (255, 0, 255), 3)
            cv2.putText(
                roi, f"distance: {dist_result[0]}", (dist_result[1][0]-20, dist_result[1][1]-20), 1, 1, (255, 0, 255), 2)
            cv2.line(roi, (0, int(height/2)),
                     (width, int(height/2)), (255, 255, 255), 2)
            if dist_result[1][1] < int(height/2):
                angel = "under"
            else:
                angel = "above"
                return roi, angel
        angel = "under"
        return roi, angel
