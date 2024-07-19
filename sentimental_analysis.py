import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK data if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Function to preprocess data
def preprocess_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    else:
        return ''  # Return empty string or handle NaN appropriately

# Load dataset with appropriate encoding
try:
    df = pd.read_csv('train.csv', encoding='latin1')  # Try different encoding if UTF-8 fails
except UnicodeDecodeError:
    print("Error: Unable to decode using specified encoding.")
    # Handle encoding error as needed

# Display dataset info if successfully loaded
if 'df' in locals():
    print("Dataset loaded, displaying dataset info")
    print(df.info())

    # Apply text preprocessing to 'selected_text' column
    print("Applying text preprocessing to 'selected_text', this may take a while...")
    df['selected_text'] = df['selected_text'].apply(preprocess_text)
    df['sentiment'] = df['sentiment'].map({'negative': 0, 'neutral': 2, 'positive': 4})
    
    print('Splitting the data')
    X_train, X_test, y_train, y_test = train_test_split(df['selected_text'], df['sentiment'], test_size=0.2, random_state=42)

    print('Vectorizing the text data...')
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print('Training the model')
    model = LogisticRegression(multi_class='ovr')
    model.fit(X_train_tfidf, y_train)

    print('Evaluating the model')
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Function to evaluate new sentences
    def evaluate_sentiment(sentence):
        processed_sentence = preprocess_text(sentence)
        transformed_sentence = vectorizer.transform([processed_sentence])
        prediction = model.predict(transformed_sentence)
        sentiment = 'negative' if prediction[0] == 0 else 'neutral' if prediction[0] == 2 else 'positive'
        return sentiment

    print("\nYou can now input sentences to evaluate their sentiment. Type 'exit' to quit.")
    while True:
        user_input = input("Enter a sentence: ")
        if user_input.lower() == 'exit':
            break
        sentiment = evaluate_sentiment(user_input)
        print(f'Sentiment: {sentiment}')

else:
    print("Error: DataFrame 'df' not defined. Check your data loading process and file path.")
