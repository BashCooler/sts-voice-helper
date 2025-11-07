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
            samplerate: int = 48_000
    ):
        # подгружаем модель
        self.__MODEL__, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="ru_v3"
        )
        self.__MODEL__.to(torch.device(device))

        self.__SPEAKER__ = speaker
        self.__SAMPLERATE__ = samplerate

    def text2speech(self, text: str):
        # генерируем аудио из текста
        audio = self.__MODEL__.apply_tts(
            text=text,
            speaker=self.__SPEAKER__,
            sample_rate=self.__SAMPLERATE__,
            put_accent=True,
            put_yo=True
        )

        # проигрываем что получилось
        sd.play(audio, samplerate=self.__SAMPLERATE__)
        time.sleep((len(audio) / self.__SAMPLERATE__))
        sd.stop()


if __name__ == "__main__":
    print(SPEAKER.XENIA)
    tts = TTS()
    tts.text2speech(text="Привет, меня зовут Саша и я диктор канала мастерская настроения")
