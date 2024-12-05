import cv2
from typing import Optional, Any


class Camera:
    """
    Класс для работы с видео или изображением как с камерой.

    Attributes:
        source (cv2.VideoCapture | str): Объект для захвата кадров из видео или путь к изображению.
        is_image (bool): Флаг, указывающий, является ли источник изображением.
    """

    def __init__(self, source_path: str) -> None:
        """
        Инициализация объекта Camera.

        Args:
            source_path (str): Путь к видеофайлу или изображению.
        """
        self.source_path = source_path
        self.is_image = source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

        if self.is_image:
            # Если это изображение, загружаем его
            self.image = cv2.imread(self.source_path)
            if self.image is None:
                raise RuntimeError(
                    f"Не удалось загрузить изображение: {self.source_path}")
        else:
            # Если это видео, открываем через VideoCapture
            self.camera = cv2.VideoCapture(self.source_path)
            if not self.camera.isOpened():
                raise RuntimeError(
                    f"Не удалось открыть видеофайл: {self.source_path}")

    def get_image(self) -> Optional[Any]:
        """
        Получает следующий кадр из видео или возвращает изображение.

        Returns:
            Optional[Any]: Кадр в формате NumPy массива или None, если видео закончилось.
        """
        if self.is_image:
            # Если это изображение, возвращаем его
            GRAY_frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            return GRAY_frame
        else:
            # Если это видео, захватываем следующий кадр
            try:
                ret, frame = self.camera.read()
                if ret:
                    GRAY_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    return GRAY_frame
                else:
                    print("Кадры в видеофайле закончились.")
                    return None
            except Exception as e:
                print(f"Ошибка получения кадра: {e}")
                return None

    def show(self, frame: Any) -> None:
        """
        Отображает кадр.

        Args:
            frame (Any): Кадр в формате NumPy массива.

        Returns:
            None
        """
        if frame is not None:
            cv2.imshow("Feed", frame)

    def end(self) -> None:
        """
        Завершает работу с видео или изображением и закрывает окна отображения.

        Returns:
            None
        """
        if not self.is_image:
            self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Указываем путь к видеофайлу или изображению
    # Замените на путь к видео или изображению (например, "example.jpg")
    source_path = "photo.png"

    try:
        # Инициализация "камеры", работающей с видеофайлом или изображением
        camera = Camera(source_path)

        while True:
            # Получаем кадр
            frame = camera.get_image()

            # Если кадр есть, отображаем его
            if frame is not None:
                camera.show(frame)
            else:
                break  # Завершаем, если кадры закончились

            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Завершение работы
        camera.end()

    except RuntimeError as e:
        print(f"Ошибка: {e}")
