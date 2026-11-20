import pytest
from analyzer import FeedbackAnalyzer

@pytest.fixture
def model():
    texts = [
        "product broken damaged defective", "device stopped working cracked product",
        "friendly helpful employee polite staff", "staff patiently helped great service",
        "package arrived late delivery delayed shipment", "courier late order not arrived",
        "nice store selection shopping experience", "reasonable prices clean shop overall",
    ]
    labels = ["Product Issue", "Product Issue", "Staff Praise", "Staff Praise", "Delivery Issue", "Delivery Issue", "General Feedback", "General Feedback"]
    return FeedbackAnalyzer().train(texts, labels)

def test_predicts_known_categories(model):
    assert model.predict("my product is broken and defective")[0] == "Product Issue"
    assert model.predict("courier delivery was late")[0] == "Delivery Issue"

def test_confidence_is_probability(model):
    label, confidence = model.predict("helpful polite staff")
    assert label in {"Product Issue", "Staff Praise", "Delivery Issue", "General Feedback"}
    assert 0 <= confidence <= 1

def test_blank_input_raises(model):
    with pytest.raises(ValueError):
        model.predict("   ")

def test_untrained_model_raises():
    with pytest.raises(RuntimeError):
        FeedbackAnalyzer().predict("hello")
