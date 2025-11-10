import sounddevice as sd
import torch
import time


# константы голосов, поддерживаемых в silero
class SPEAKER:
    AIDAR   = "aidar"
    BAYA    = "baya"
    KSENIYA = "kseniya"
    XENIA   = "xenia"
    RANDOM  = "random"

# константы девайсов для работы torch
class DEVICE:
    CPU    = "cpu"
    CUDA   = "cuda"
    VULKAN = "vulkan"
    OPENGL = "opengl"
    OPENCL = "opencl"


class TTS:
    def __init__(
            self, speaker: str = SPEAKER.XENIA,
            device: str = DEVICE.CPU,
            samplerate: int = 48000
    ):
        # подгружаем модель
        self.__MODEL__, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v5_ru",
            trust_repo=True
        )
        self.__MODEL__.to(torch.device(device))

        self.__SPEAKER__ = speaker
        self.__SAMPLERATE__ = samplerate

    def say(self, text: str):
        # генерируем аудио из текста
        audio = self.__MODEL__.apply_tts(
            text=str(text),
            speaker=self.__SPEAKER__,
            sample_rate=self.__SAMPLERATE__,
            put_accent=True,
            put_yo=False
        )

        # проигрываем что получилось
        sd.play(audio, samplerate=self.__SAMPLERATE__)
        time.sleep((len(audio) / self.__SAMPLERATE__))
        sd.stop()

    def hello(self):
        self.say(text="Буду рада ответить на ваши вопросы")

    def default(self):
        self.say(text="К сожалению я не знаю ответа, вы можете найти больше информации на сайте ка́федры "
                      "ю, ю, эс, ти́, точка, ру, слэш, информатик https://uust.ru/informatic/")
