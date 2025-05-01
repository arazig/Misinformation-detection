import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Load and preprocess data
def load_and_preprocess_data(fake_path, true_path):
    print("[INFO] Loading data...")
    time.sleep(0.5)
    fake_data = pd.read_csv(fake_path)
    true_data = pd.read_csv(true_path)

    print("[INFO] Preprocessing data...")
    time.sleep(0.5)
    fake_data['label'] = 0  # Fake news
    true_data['label'] = 1  # True news

    data = pd.concat([fake_data, true_data], ignore_index=True)
    data['cleaned_text'] = data['text'].apply(preprocess_text)

    print("[INFO] Data loaded and preprocessed successfully.")
    return data

# Vectorization functions
def vectorize_data(data, method='tfidf', max_features=5000):
    print(f"[INFO] Vectorizing data using {method.upper()}...")
    time.sleep(0.5)
    if method == 'bow':
        vectorizer = CountVectorizer(max_features=max_features)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        raise ValueError("Unsupported vectorization method. Use 'bow' or 'tfidf'.")

    features = vectorizer.fit_transform(data['cleaned_text'])
    print(f"[INFO] Data vectorized using {method.upper()} successfully.")
    return features, vectorizer

# Train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []

    for model_name, model in models.items():
        print(f"[INFO] Training {model_name}...")
        time.sleep(0.5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[INFO] {model_name} trained successfully with accuracy: {accuracy:.5f}")
        results.append({'Model': model_name, 'Accuracy': accuracy})

    return pd.DataFrame(results)

# Main function to run the project
def run_project(fake_path, true_path):
    print("[INFO] Starting the project pipeline...")
    time.sleep(0.5)

    # Load and preprocess data
    data = load_and_preprocess_data(fake_path, true_path)

    # Split data
    print("[INFO] Splitting data into training and testing sets...")
    time.sleep(0.5)
    X = data['cleaned_text']
    y = data['label']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize data
    vectorization_methods = ['bow', 'tfidf']
    all_results = []

    for method in vectorization_methods:
        print(f"[INFO] Processing vectorization method: {method.upper()}...")
        time.sleep(0.5)
        X_train, vectorizer = vectorize_data(pd.DataFrame({'cleaned_text': X_train_raw}), method=method)
        X_test = vectorizer.transform(X_test_raw)

        # Train and evaluate models
        print(f"[INFO] Training and evaluating models for {method.upper()}...")
        time.sleep(0.5)
        results = train_and_evaluate(X_train, X_test, y_train, y_test)
        results['Vectorizer'] = method
        all_results.append(results)

    # Combine results
    print("[INFO] Combining results from all vectorization methods...")
    time.sleep(0.5)
    final_results = pd.concat(all_results, ignore_index=True)
    print("[INFO] Project pipeline completed successfully.")
    return final_results

# Example usage
if __name__ == "__main__":
    fake_path = 'data/Fake.csv'
    true_path = 'data/True.csv'
    results = run_project(fake_path, true_path)
    print(results)