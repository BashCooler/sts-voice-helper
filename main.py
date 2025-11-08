import speech
import json
import whisper
from numpy import ndarray
from recorder import Recorder
from fuzzywuzzy import fuzz


class Assistant:
    """
    Основной модуль голосового помощника
    """
    def __init__(self, model_name: str):
        self.model = whisper.load_model(model_name)
        self.QA = {}
        self.tts = speech.TTS()

    def process(self, audio_data: ndarray):
        text = whisper.transcribe(audio=audio_data, model=self.model)['text']
        print(f"> {text}")
        for q, a in self.QA.items():
            if fuzz.ratio(text, q) >= 70:
                self.tts.say(a)

    def main(self):
        self.QA = load_json('question.json')
        self.tts.hello()
        Recorder(callback=self.process)
        return


def load_json(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        get_link = lambda item: ' ' + item.get('link') if item.get('link') is not None else ''
        qa = {item['question']: f"{item['answer']}{get_link(item)}" for item in data['questions']}
        return qa


if __name__ == "__main__":
    assistant = Assistant(model_name='tiny')
    assistant.main()
