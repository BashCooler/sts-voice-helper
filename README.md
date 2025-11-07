# sts-voice-helper

Голосовой помошник, консультирующий по направлению СТС

## Зависимости
- Python 3.13.7
- Библиотеки

  ```
  pip install numpy sounddevice pynput openai-whisper torch
  ```

## Whisper
Модели загружаются в `(user)/.cache/whisper`.

Доступные модели:
1. tiny - 39 M,
2. base - 74 M,
3. small - 244 M,
4. medium - 769 M,
5. large - 1150 M,
6. turbo - 809 M

Спасибо OpenAI капец 😳

## Полезные статьи

- [Решение с silero](https://habr.com/ru/articles/864000/)
- [Решение с pyttsx3](https://habr.com/ru/articles/529590/)
- [Сравнение Vosk и Whisper](https://habr.com/ru/articles/814057/)
- [Решение с Whisper](https://habr.com/ru/articles/919720/)
