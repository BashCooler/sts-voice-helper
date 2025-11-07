import whisper
from numpy import ndarray
from recorder import Recorder

# Модели загружаются в (user)/.cache/whisper
# Model - Parameters: tiny - 39 M, base - 74 M, small - 244 M, medium - 769 M, large - 1150 M, turbo - 809 M
model = whisper.load_model('tiny')


def process_audio(audio_data: ndarray):
    text = whisper.transcribe(audio=audio_data, model=model)['text']
    print(text)


def main():
    Recorder(callback=process_audio)


if __name__ == "__main__":
    main()
