import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Define a function to test the model
def test_spam_detection_model():
    # Define your sample data
    array = np.array(['Hello, could we meet tomorrow?', 'Win a free prize now!'])

    # Vectorize the input data using TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(array)  # Transform the text data into feature vectors

    # Define labels with at least two classes (e.g., 0 for non-spam, 1 for spam)
    y = [0, 1]  # 0 is non-spam, 1 is spam

    # Initialize the Logistic Regression model
    model = LogisticRegression()

    # Fit the model on the transformed data (X) and the target labels (y)
    model.fit(X, y)

    # Perform a prediction to check if the setup works
    prediction = model.predict(X)
    
    # Assert that predictions are correct
    assert prediction[0] == 0, f"Expected 0, but got {prediction[0]} for the non-spam message."
    assert prediction[1] == 1, f"Expected 1, but got {prediction[1]} for the spam message."

    print("Predictions:", prediction)

# If you want pytest to automatically discover this test, ensure that the file name starts with 'test_'
