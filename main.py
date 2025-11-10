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
    def __init__(self, model_name: str):
        self.model = whisper.load_model(model_name)
        self.tts = speech.TTS()
        self.paraphraser = Paraphraser(model_path=r'./models/paraphrase-multilingual-mpnet-base-v2')
        self.QA = load_json('question.json')
        self.questions = list(self.QA.keys())
        # Выполняем при инициализации
        self.paraphraser.setup_questions(list(self.QA.keys()))
        self.tts.hello()
        Recorder(callback=self.process)

    def process(self, audio_data: ndarray):
        # Транскрибировать аудио
        prompt = "Текст на русском языке, ожидаются слова: СТС, квалификация, бакалавриат, специалитет"
        text_1 = whisper.transcribe(audio=audio_data, model=self.model, initial_prompt=prompt)['text']
        print(f"Распознан текст     : {text_1}")
        # Привести к одному из вопросов в базе
        text_2 = self.paraphraser.find_best_match(text_1)
        print(f"Обработанный текст  : {text_2}")
        # Если такой вопрос есть, озвучить ответ
        max_ratio = 0
        question = None
        for q, a in self.QA.items():
            # Если языковой моделью найдено совпадение, озвучить ответ
            if text_2 is q or fuzz.partial_token_sort_ratio(text_1, q) >= 60:
                self.tts.say(a)
                print(f"Ответ               : {a}\n")
                return
        self.tts.default()
        return


def load_json(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        qa = {item['question']: item['answer'] for item in data['questions']}
        return qa


if __name__ == "__main__":
    assistant = Assistant(model_name='base')
