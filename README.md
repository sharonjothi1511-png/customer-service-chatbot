# Customer Service Chatbot

A simple NLP chatbot that answers basic customer queries (orders, shipping, returns, payments).
It matches the user's message to example phrases in `intents.json` using **TF-IDF vectors and cosine similarity**,
and falls back to a polite message when it isn't confident.

## Features
- Intent matching with TF-IDF (unigrams + bigrams)
- Confidence threshold with a fallback reply
- Easy to extend: add intents in `intents.json`, no code changes needed
- Unit tests with pytest

## Setup
```bash
git clone https://github.com/<sharonjothi1511-png>/customer-service-chatbot.git
cd customer-service-chatbot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python chatbot.py
```

## Test
```bash
pytest
```

## Example
```
You: where is my order
Support Bot: You can track your order under 'My Orders' using your order ID. Tracking updates within 24 hours of dispatch.

You: can I pay with UPI
Support Bot: We accept cards, UPI, net banking and cash on delivery. If a payment failed but money was deducted, it is refunded automatically in 3-5 days.
```

## How it works
1. `intents.json` holds each intent's example patterns and responses.
2. On startup, all patterns are cleaned (lowercased, punctuation removed) and converted to TF-IDF vectors.
3. A user message is vectorized the same way and compared to every pattern with cosine similarity.
4. The best-matching intent's response is returned, or a fallback if the score is below the threshold.

## Future improvements
- Add lemmatization/stemming with NLTK or spaCy
- Add entity extraction (e.g., pull order IDs from messages)
- Web UI with Flask or Streamlit
- Swap TF-IDF for sentence embeddings

## Author
sharon

## License
MIT