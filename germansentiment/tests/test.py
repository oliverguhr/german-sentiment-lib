from sentimentmodel import SentimentModel
import pytest


model = SentimentModel()

def test_multi_document_classification():
    texts = ["Mit keinem guten Ergebniss","Das war unfair", "Das ist gar nicht mal so gut",
            "Total awesome!","nicht so schlecht wie erwartet", "Das ist gar nicht mal so schlecht",
            "Der Test verlief positiv.","Sie fährt ein grünes Auto.", "Der Fall wurde an die Polzei übergeben."]
        
    result = model.predict_sentiment(texts)
    expected = ["negative","negative","negative","positive","positive","positive","neutral", "neutral", "neutral"]

    assert result == expected

def test_single_document_classification():
    text = ["Mit keinem guten Ergebniss"]
        
    result = model.predict_sentiment(text)
    expected = ["negative"]
    assert result == expected


def test_long_document_classification():
    text = ["Mit keinem guten Ergebniss" * 100]
        
    result = model.predict_sentiment(text)
    expected = ["negative"]
    assert result == expected
