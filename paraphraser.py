from sentence_transformers import SentenceTransformer, util
import numpy as np


class Paraphraser:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
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
