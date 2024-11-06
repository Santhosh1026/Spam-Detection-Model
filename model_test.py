import pytest
import joblib
import pandas as pd

# Load the model
model = joblib.load('C://Users//lucky//SpamDetectionModel//models//spam_detection_model.pkl')

# Test data samples
test_text_spam = "Congratulations! You've won a prize."
test_text_ham = "Hello, could we meet tomorrow?"

def test_spam_prediction():
    prediction = model.predict([test_text_spam])[0]
    assert prediction == 1, "Failed to predict spam"

def test_ham_prediction():
    prediction = model.predict([test_text_ham])[0]
    assert prediction == 0, "Failed to predict ham"
