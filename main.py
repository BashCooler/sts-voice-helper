import speech
import json
import whisper
from numpy import ndarray
from recorder import Recorder
from fuzzywuzzy import fuzz
from paraphraser import Paraphraser


class Assistant:
    """
    Основной модуль голосового помощника

    `model_name: str` - имя модели

    `paraphrase: bool` - пробовать привести формулировку пользователя к одному из известных вопросов

    Модели загружаются в (user)/.cache/whisper

    Доступные модели: `tiny` - 39 M, `base` - 74 M, `small` - 244 M, `medium` - 769 M, `large` - 1150 M, `turbo` - 809 M
    """
    def __init__(self, model_name: str, paraphrase: bool = False):
        self.model = whisper.load_model(model_name)
        self.tts = speech.TTS()
        self.paraphrase = paraphrase
        self.paraphraser = Paraphraser()
        self.QA = load_json('question.json')
        # Выполняем при инициализации
        self.paraphraser.setup_questions(list(self.QA.keys()))
        self.tts.hello()
        Recorder(callback=self.process)

    def process(self, audio_data: ndarray):
        text = whisper.transcribe(audio=audio_data, model=self.model)['text']
        # Привести к одному из вопросов в базе
        if self.paraphrase:
            text = self.paraphraser.find_best_match(text)
        # Если такой вопрос есть озвучить ответ
        print(f"> {text}")
        for q, a in self.QA.items():
            if fuzz.partial_ratio(text, q) >= 70:
                self.tts.say(a)


def load_json(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        get_link = lambda item: ' ' + item.get('link') if item.get('link') is not None else ''
        qa = {item['question']: f"{item['answer']}{get_link(item)}" for item in data['questions']}
        return qa


if __name__ == "__main__":
    assistant = Assistant(model_name='tiny', paraphrase=True)
