import asyncio
import threading


class Robot:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.writer = None

    async def handle_client(self, reader, writer):
        self.writer = writer  # Сохраняем объект writer для отправки сообщений
        while True:
            try:
                data = await asyncio.wait_for(reader.read(1024), timeout=1.0)
            except asyncio.exceptions.TimeoutError:
                print("Таймаут ожидания данных")
                continue
            if not data:
                print("Клиент отключился")
                break
            message = data.decode()
            print(f"Получено сообщение от клиента: {message}")
            if message == "quit":
                break
            #response = self.process_message(message)
            #writer.write(response.encode())
            #await writer.drain()
        print("Закрытие соединения")
        writer.close()

    def process_message(self, message):
        # Обработка сообщения по вашему усмотрению
        return "Сообщение получено"

    def send_message(self, message):
        if self.writer is None:
            print("Ошибка: нет активного соединения")
            return
        try:
            self.writer.write(message.encode())
            self.writer.drain()
        except Exception as e:
            print(f"Ошибка отправки сообщения: {e}")

    async def start_server(self):
        while True:
            try:
                server = await asyncio.start_server(self.handle_client, self.host, self.port)

                print(f"Сервер слушает на {self.host}:{self.port}")

                async with server:
                    await server.serve_forever()
            except Exception as e:
                print(f"Ошибка сервера: {e}")
                print("Повторная попытка запуска сервера через 5 секунд...")
                await asyncio.sleep(5)


def run_server():
    asyncio.run(Robot('192.168.0.20', 48569).start_server())


if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
