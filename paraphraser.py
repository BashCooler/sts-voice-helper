from sentence_transformers import SentenceTransformer, util
import numpy as np


class Paraphraser:
    """
    `model_path` - путь к локально установленной модели
    `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

    При отсутствии локальной копии модели она будет загружена из
    репозитория
    """
    def __init__(self, model_path: str = None):
        self.path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        self.model_path = model_path
        try:
            self.model = SentenceTransformer(self.model_path)
        except FileNotFoundError:
            self.model = SentenceTransformer(self.path)
        self.available_questions = []
        self.question_embeddings = None

    def setup_questions(self, questions: list):
        self.available_questions = questions
        self.question_embeddings = self.model.encode(questions, convert_to_tensor=True)

    def find_best_match(self, user_input: str, threshold: float = 0.6):
        user_embedding = self.model.encode([user_input], convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(user_embedding, self.question_embeddings)[0]
        best_match_idx = np.argmax(similarities.cpu().numpy())
        best_similarity = similarities[best_match_idx]

        if best_similarity > threshold:
            return self.available_questions[best_match_idx]
        return None
