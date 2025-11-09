from numpy import concatenate, float32
from sounddevice import InputStream
from pynput.keyboard import Key, Listener


class Recorder:
    """
    Записывает аудио при зажатой клавише SPACE. Запись передается в callback функцию
    в виде ndarray. Для завершения работы нажать Esc.
    """
    def __init__(self, callback, samplerate=16000, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self.is_recording = False
        self.data = []
        self.stream = None
        self.callback = callback
        self.listen()

    def listen(self):
        print(" • Нажмите и удерживайте ПРОБЕЛ чтобы говорить")
        print(" • ESC для выхода")
        with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def on_press(self, key):
        if key == Key.space and not self.is_recording:
            self.start_recording()

    def on_release(self, key):
        if key == Key.space and self.is_recording:
            self.stop_recording()
        return False if key == Key.esc else None

    def start_recording(self):
        self.is_recording = True
        self.data = []
        self.stream = InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=lambda *args: self.data.append(args[0].copy()) if self.is_recording else None
        )
        self.stream.start()
        print("⏺️ Запись идет...")

    def stop_recording(self):
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        print(f"⏹️ Запись остановлена")
        try:
            audio_data = concatenate(self.data, axis=0)
        except ValueError:
            return
        self.callback(audio_data.flatten().astype(float32))
