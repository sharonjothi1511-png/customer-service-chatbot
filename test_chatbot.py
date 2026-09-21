from chatbot import Chatbot, FALLBACK

bot = Chatbot()


def test_order_query():
    assert "track" in bot.get_response("where is my package").lower()


def test_returns_query():
    assert "return" in bot.get_response("I want to return something").lower()


def test_fallback():
    assert bot.get_response("what is the meaning of life xyzzy") == FALLBACK


def test_cancel_query():
    assert "cancel" in bot.get_response("I want to cancel my order").lower()