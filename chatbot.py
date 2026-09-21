"""Customer service chatbot: TF-IDF intent matching with a fallback reply."""
import json
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INTENTS_FILE = Path(__file__).parent / "intents.json"
THRESHOLD = 0.3  # minimum similarity before we trust a match
FALLBACK = "Sorry, I didn't get that. Could you rephrase, or type 'agent' to reach a human?"


def preprocess(text: str) -> str:
    """Lowercase and strip punctuation."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


class Chatbot:
    def __init__(self, intents_path=INTENTS_FILE, threshold=THRESHOLD):
        data = json.loads(Path(intents_path).read_text(encoding="utf-8"))
        self.intents = data["intents"]
        self.threshold = threshold

        # One row per example pattern, remembering which intent it belongs to
        self.patterns, self.labels = [], []
        for idx, intent in enumerate(self.intents):
            for pattern in intent["patterns"]:
                self.patterns.append(preprocess(pattern))
                self.labels.append(idx)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.patterns)

    def get_response(self, message: str) -> str:
        query = self.vectorizer.transform([preprocess(message)])
        scores = cosine_similarity(query, self.matrix)[0]
        best = scores.argmax()
        if scores[best] < self.threshold:
            return FALLBACK
        return random.choice(self.intents[self.labels[best]]["responses"])


def main():
    bot = Chatbot()
    print("Support Bot: Hi! Ask me about orders, shipping, returns or payments. (type 'quit' to exit)")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in {"quit", "exit", "bye"}:
            print("Support Bot: Thanks for chatting. Goodbye!")
            break
        print("Support Bot:", bot.get_response(user))


if __name__ == "__main__":
    main()