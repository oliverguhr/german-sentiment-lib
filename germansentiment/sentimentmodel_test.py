from .sentimentmodel import SentimentModel
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

def test_output_class_probabilities():
    text = ["das ist super"]
        
    result, probabilities = model.predict_sentiment(text, output_probabilities = True)    
    assert result == ["positive"]
    assert len(probabilities[0]) == 3    

def test_output_multitiple_class_probabilities():
    text = ["Mit keinem guten Ergebniss", "Das ist toll"]
        
    result, probabilities = model.predict_sentiment(text, output_probabilities = True)    
    assert result == ["negative", "positive"]
    assert len(probabilities) == 2
    assert len(probabilities[0]) == 3    


def test_long_document_classification():
    text = ["Mit keinem guten Ergebniss" * 100]
        
    result = model.predict_sentiment(text)
    expected = ["negative"]
    assert result == expected
