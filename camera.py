from typing import Optional, Any
from hik_camera.hik_camera import HikCamera
import cv2


class Camera:
    """
    Класс для работы с IP-камерой Hikvision.

    Attributes:
        camera (HikCamera): Объект камеры для захвата изображений.
        ip (str): IP-адрес подключенной камеры.
    """

    def __init__(self, ip: Optional[str] = None) -> None:
        """
        Инициализация объекта Camera.

        Args:
            ip (Optional[str]): IP-адрес камеры. Если не указан, автоматически выбирается первая доступная камера.
        """
        self.ip = ip if ip else self._get_first_camera_ip()
        if not self.ip:
            raise RuntimeError("Не удалось найти доступные камеры.")

        self.camera = HikCamera(ip=self.ip)
        self._configure_camera()

    def _get_first_camera_ip(self) -> Optional[str]:
        """
        Получает IP-адрес первой доступной камеры.

        Returns:
            Optional[str]: IP-адрес первой камеры или None, если камеры не найдены.
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
        # try:
        #     # Отключить автоматическую экспозицию
        #     self.camera["ExposureAuto"] = "Off"
        #     print("Автоматическая экспозиция отключена.")

        #     # Проверить доступные параметры
        #     available_params = self.camera.get_all_parameters()
        #     if "ExposureTime" not in available_params:
        #         print("Параметр ExposureTime недоступен для этой камеры.")
        #         return

        #     # Попытаться получить текущее значение экспозиции
        #     current_exposure = self.camera["ExposureTime"]
        #     print(f"Текущее время экспозиции: {current_exposure}")

        #     # Установить новое значение экспозиции
        #     self.camera["ExposureTime"] = 8000
        #     print(f"Новое время экспозиции: {self.camera['ExposureTime']}")
        # except AssertionError as e:
        #     print(f"Ошибка настройки камеры: {e}")

    def get_image(self) -> Optional[Any]:
        """
        Получает изображение с камеры.

        Returns:
            Optional[Any]: Изображение в формате NumPy массива или None, если произошла ошибка.
        """
        try:
            with self.camera:
                frame = self.camera.robust_get_frame()
                if len(frame.shape) == 2:
                    GRAY_frame = frame
                else:
                    GRAY_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                inverted_GRAY_frame = cv2.bitwise_not(GRAY_frame)
                return inverted_GRAY_frame
        except Exception as e:
            print(f"Ошибка получения изображения: {e}")
            return None

    def show(self, frame: Any) -> None:
        """
        Отображает изображение.

        Args:
            frame (Any): Изображение в формате NumPy массива.

        Returns:
            None
        """
        if frame is not None:
            height, width = frame.shape[:2]
            new_dim = (width // 3, height // 3)
            resized_frame = cv2.resize(frame, new_dim)
            cv2.imshow("Camera Feed", resized_frame)

    def end(self) -> None:
        """
        Завершает работу с камерой и закрывает окна отображения.

        Returns:
            None
        """
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Инициализация камеры
    try:
        camera = Camera()

        while True:
            # Получаем изображение
            frame = camera.get_image()

            # Отображаем изображение
            if frame is not None:
                camera.show(frame)

            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.end()

    except RuntimeError as e:
        print(f"Ошибка: {e}")
