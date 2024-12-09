import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle  # For saving/loading model and vectorizer
from CleaningData import clean_text

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load your cleaned data
file_path = 'D:/TweetsAfterCleaning.csv'  # Update this path
data = pd.read_csv(file_path)



# Split data into features and labels
X = data['text']  # Independent variable (cleaned text)
y = data['sentiment']  # Dependent variable (sentiment)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Vectorize the text data
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Save the model and TF-IDF vectorizer
with open('your_model.pkl', 'wb') as model_file:
    pickle.dump(lr_model, model_file)
    
with open('your_tfidf_vectorizer.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

# Evaluate the model
y_pred = lr_model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



############ Function to predict sentiment from user input ##########
def predict_sentiment(user_input):
    # Preprocess the input text using your clean_text method
    cleaned_UserText = clean_text(user_input)

    # Transform the input text using TF-IDF
    input_tfidf = tfidf.transform([cleaned_UserText])

    # Load the trained model
    with open('your_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Predict sentiment using the model
    sentiment_prediction = model.predict(input_tfidf)

    # Map the prediction to a sentiment label based on numerical predictions
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    # If your model outputs a numerical label (0, 1, or 2), you can directly map it
    if isinstance(sentiment_prediction[0], int):
        sentiment_label = sentiment_map[sentiment_prediction[0]]
    else:
        # If the model outputs a string label (e.g., 'negative', 'neutral', 'positive')
        sentiment_label = sentiment_prediction[0]
    
    return sentiment_label


# Interactive input for user testing
while True:
    user_input = input("Enter a tweet or sentence to analyze sentiment (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    
    # Predict and display sentiment
    sentiment = predict_sentiment(user_input)
    print(f"Predicted Sentiment: {sentiment}")
