import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function for text preprocessing
def preprocess_text(text):
    # Lowercasing, removing stopwords and lemmatizing
    text = text.lower()  # Convert text to lowercase
    words = text.split()  # Split text into words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(words)

# Load the dataset (ensure the correct file path)
df = pd.read_csv('C://Users//lucky//SpamDetectionModel//data//spam_ham_dataset.csv')  # Ensure you provide the correct path

# Check the first few rows to understand the structure of the dataset
print(df.head())

# Preprocess the text data
df['processed_text'] = df['text'].apply(preprocess_text)

# Feature matrix and target vector
X = df['processed_text'].values
y = df['label_num'].values  # Using 'label_num' for target (0 for ham, 1 for spam)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data using TfidfVectorizer
vectorizer = TfidfVectorizer()

# Transform the training data into numerical vectors
X_train_vect = vectorizer.fit_transform(X_train)

# Apply SMOTE to handle class imbalance on the vectorized data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vect, y_train)

# Train the model (using RandomForestClassifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Vectorize and predict on the test data
X_test_transformed = vectorizer.transform(X_test)
y_pred = model.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and vectorizer to a file
with open('spam_detection_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    pickle.dump(vectorizer, model_file)  # Save the vectorizer as well

print("Model training completed and saved to spam_detection_model.pkl")

# Import necessary libraries for prediction
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess the text (same as used during training)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Tokenize the text
    words = text.split()
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Load the trained model and vectorizer
with open('spam_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)  # Load the trained model
    vectorizer = pickle.load(model_file)  # Load the vectorizer

# Example text to predict (you can replace this with any text)
test_message = "Congratulations! You've won a $1000 gift card. Call us now!"

# Preprocess and vectorize the text
processed_message = preprocess_text(test_message)
message_vect = vectorizer.transform([processed_message])

# Predict whether the message is spam or not
prediction = model.predict(message_vect)

# Output the result
if prediction[0] == 1:
    print("This is a spam message.")
else:
    print("This is a ham (non-spam) message.")
