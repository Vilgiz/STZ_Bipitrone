import cv2
import numpy as np
from typing import Optional, Any
from hik_camera.hik_camera import HikCamera
import json


class Marker:
    def __init__(self, id, center, corners):
        self.id = id
        self.center = center
        [self.topLeft, self.topRight, self.bottomRight, self.bottomLeft] = corners


class Camera:
    """
    Класс для работы с IP-камерой Hikvision.
    """

    def __init__(self, ip: Optional[str] = None) -> None:
        """
        Инициализация объекта Camera.
        """
        self.ip = ip if ip else self._get_first_camera_ip()
        if not self.ip:
            raise RuntimeError("Не удалось найти доступные камеры.")

        self.camera = HikCamera(ip=self.ip)
        self._configure_camera()

    def _get_first_camera_ip(self) -> Optional[str]:
        """
        Получает IP-адрес первой доступной камеры.
        """
        ips = HikCamera.get_all_ips()
        if ips:
            print(f"Найдены камеры: {ips}")
            return ips[0]
        print("Камеры не найдены.")
        return None

    def _configure_camera(self) -> None:
        """
        Настраивает параметры камеры.
        """
        with self.camera:
            self.camera["GainAuto"] = "Off"
            self.camera["Gain"] = 0
        print(f"Камера с IP {self.ip} настроена.")

    def get_image(self) -> Optional[Any]:
        """
        Получает изображение с камеры.
        """
        try:
            with self.camera:
                frame = self.camera.robust_get_frame()
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Ошибка получения изображения: {e}")
            return None

    def show(self, frame: Any) -> None:
        """
        Отображает изображение.
        """
        if frame is not None:
            height, width = frame.shape[:2]
            new_dim = (width // 3, height // 3)
            resized_frame = cv2.resize(frame, new_dim)
            cv2.imshow("Camera Feed", resized_frame)

    def end(self) -> None:
        """
        Завершает работу с камерой и закрывает окна отображения.
        """
        cv2.destroyAllWindows()


class ImageProcessor:
    def detectArucoMarkers(self, image):
        markers = {}

        arucoDictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50)

        arucoParameters = cv2.aruco.DetectorParameters()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
            image, arucoDictionary, parameters=arucoParameters)

        if ids is None:
            print("ArUco маркеры не найдены!")
            return markers

        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = [int(topRight[0]), int(topRight[1])]
            bottomRight = [int(bottomRight[0]), int(bottomRight[1])]
            bottomLeft = [int(bottomLeft[0]), int(bottomLeft[1])]
            topLeft = [int(topLeft[0]), int(topLeft[1])]
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            print(f"[INFO] ArUco marker ID: {markerID}")
            markers[markerID] = Marker(
                markerID, [cX, cY], [topLeft, topRight, bottomRight, bottomLeft])

        return markers

    def cropImage(self, image, points):
        rect = np.array(points, dtype="float32")
        (tl, tr, br, bl) = rect

        dst = np.array([
            [0, 0],
            [279, 0],
            [279, 197],
            [0, 197]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)

        # Запись M, maxWidth и maxHeight в JSON
        data = {
            'M': M.tolist(),
            'maxWidth': 279,
            'maxHeight': 197
        }

        with open('transformation_data.json', 'w') as json_file:
            json.dump(data, json_file)

        warped = cv2.warpPerspective(image, M, (279, 197))
        return warped


if __name__ == "__main__":
    try:
        camera = Camera()
        ip = ImageProcessor()

        while True:
            # Получаем изображение
            frame = camera.get_image()

            if frame is not None:
                markers = ip.detectArucoMarkers(frame)

                if len(markers) >= 4:
                    cropped_image = ip.cropImage(frame, [
                        markers[0].topLeft,
                        markers[1].topRight,
                        markers[2].bottomRight,
                        markers[3].bottomLeft])

                    cv2.imshow("Cropped Image", cv2.cvtColor(
                        cropped_image, cv2.COLOR_RGB2BGR))
                else:
                    print("Недостаточно маркеров для обрезки изображения!")

                camera.show(frame)

            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.end()

    except RuntimeError as e:
        print(f"Ошибка: {e}")
