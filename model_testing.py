import pickle
import joblib  # Correct way to import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open('spam_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Sample text for testing
test_text = ["Congratulations, you won a free iPhone!"]

# Transform the text using the loaded vectorizer
test_text_tfidf = vectorizer.transform(test_text)

# Predict using the trained model
prediction = model.predict(test_text_tfidf)

# Print the result
if prediction[0] == 1:
    print("This is a spam message.")
else:
    print("This is not a spam message.")
