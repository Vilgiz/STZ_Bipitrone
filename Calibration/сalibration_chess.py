import cv2
import numpy as np
from pypylon import pylon
import json

class ChessboardCalibrator:
    def __init__(self, param_file_path):
        # Загрузка параметров из файла
        with open(param_file_path, 'r') as file:
            self.calibration_params = json.load(file)

        self.pattern_size = tuple(self.calibration_params['pattern_size'])
        self.criteria = self.calibration_params['criteria']
        self.term_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self.criteria.get('max_iter', 30),
            self.criteria.get('epsilon', 0.001)
        )

        self.calibrated = False
        self.object_points = []
        self.image_points = []
        self.camera_matrix = None
        self.dist_coefficients = None

    def calibrate_image(self, input_image):
        # Если еще не проведена калибровка, пробуем найти углы шахматной доски
        if not self.calibrated:
            ret, corners = cv2.findChessboardCorners(
                input_image, self.pattern_size, None)

            if ret:
                # Улучшение точности углов
                corners2 = cv2.cornerSubPix(
                    input_image, corners, (11, 11), (-1, -1), self.term_criteria)

                # Добавление углов в массив
                self.object_points.append(
                    np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32))
                self.object_points[-1][:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
                self.image_points.append(corners2)

                # Отрисовка углов на изображении
                cv2.drawChessboardCorners(input_image, self.pattern_size, corners2, ret)

                print("Найдены углы шахматной доски.")

                # Если собраны данные для калибровки, проводим калибровку
                if len(self.object_points) == 30:  # Используем достаточное количество изображений
                    self.calibrate(input_image.shape[::-1])  # Передаем размеры в правильном порядке
            else:
                print("Не удалось найти шахматную доску на изображении.")
        else:
            # Если калибровка проведена, применяем параметры к изображению без доски
            undistorted_image = cv2.undistort(input_image, self.camera_matrix, self.dist_coefficients)
            cv2.imshow("Undistorted Image", undistorted_image)

    def calibrate(self, image_size):
        print("Проводим калибровку...")

        # Преобразование списка точек в numpy массивы
        object_points = np.array(self.object_points)
        image_points = np.array(self.image_points)

        # Калибровка камеры
        ret, self.camera_matrix, self.dist_coefficients, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None)

        if ret:
            print("Калибровка завершена успешно.")
            self.calibrated = True

            # Сохранение параметров в JSON
            calibration_result = {
                'camera_matrix': self.camera_matrix.tolist(),
                'dist_coefficients': self.dist_coefficients.tolist()
            }

            with open('calibration_result.json', 'w') as result_file:
                json.dump(calibration_result, result_file, indent=4)
        else:
            print("Не удалось провести калибровку.")

if __name__ == "__main__":
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    calibrator = ChessboardCalibrator('calibration_chees_parametrs.json')

    if camera.IsOpen():
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                frame = grabResult.Array
                with open('transformation_data.json', 'r') as json_file:

                    data = json.load(json_file)

                M = np.array(data['M'])
                maxWidth = data['maxWidth']
                maxHeight = data['maxHeight']
                frame = cv2.warpPerspective(
                frame, M, (maxWidth, maxHeight))
                calibrator.calibrate_image(frame)

                cv2.imshow("Calibration Image", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        camera.StopGrabbing()
        cv2.destroyAllWindows()
    else:
        print("Камера не открыта.")
