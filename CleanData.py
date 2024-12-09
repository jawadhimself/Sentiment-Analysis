import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the data
file_path = 'D:/DATA SET/DATA SET/twitter 2/newTweets.csv'  # Update this to the actual file path if different
data = pd.read_csv(file_path)

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Checking number of null values
print(data.isnull().sum())

#Dropping it
data.dropna(inplace=True)

#Checking after droping null values
print(data.isnull().sum())

#Check duplicates
print(data.duplicated().sum())
data = data.drop_duplicates(subset=['text'])
#Check after deleting duplicates
print(data.duplicated().sum())


# Define a function to clean text
def clean_text(text):
    try:
        # Convert to lowercase
        text = text.lower()

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Join tokens back into a single string
        return ' '.join(tokens)
    except Exception as e:
        return text  # Return the original text if any error occurs


# Apply cleaning to the 'text' column
data['text'] = data['text'].apply(clean_text)

# Save the cleaned data to a new CSV file
output_path = 'D:TweetsAfterCleaning.csv' # Update the path
data.to_csv(output_path, index=False)

print(f"Cleaned data saved to {output_path}")
