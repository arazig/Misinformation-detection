import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
import time

from src.deep_lstm_model import prepare_word_embedding_features, train_and_evaluate_lstm, predict_on_new_data

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

# Word2Vec vectorization - Version corrigée
def vectorize_with_word2vec(data, vector_size=100):
    print("[INFO] Vectorizing data using Word2Vec...")
    time.sleep(0.5)
    
    # Tokenize the text
    tokenized_texts = [text.split() for text in data['cleaned_text']]
    
    # Train Word2Vec model
    word2vec_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4
    )
    
    # Create document vectors by averaging word vectors
    document_vectors = []
    for tokens in tokenized_texts:
        vectors = []
        for token in tokens:
            if token in word2vec_model.wv:
                vectors.append(word2vec_model.wv[token])
        if len(vectors) > 0:
            doc_vector = np.mean(vectors, axis=0)
        else:
            doc_vector = np.zeros(vector_size)
        document_vectors.append(doc_vector)
    
    X = np.array(document_vectors)
    print("[INFO] Data vectorized using Word2Vec successfully.")
    return X, word2vec_model

# BERT vectorization
def vectorize_with_bert(data, max_len=128):  # Réduit pour économiser de la mémoire
    print("[INFO] Vectorizing data using BERT...")
    time.sleep(0.5)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # Tokenize with truncation and padding
    inputs = tokenizer(
        data['cleaned_text'].tolist(),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    # Get embeddings
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # Use mean pooling of last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    print("[INFO] Data vectorized using BERT successfully.")
    return embeddings, tokenizer

# Train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Si les données ont 3 dimensions (comme dans l'ancienne version Word2Vec), les aplatir
    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

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
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            print(f"[INFO] {model_name} trained successfully with accuracy: {accuracy:.5f}")
            results.append({
                'Model': model_name, 
                'Accuracy': accuracy,
                'Classification Report': report
            })
        except Exception as e:
            print(f"[ERROR] Failed to train {model_name}: {str(e)}")
            results.append({
                'Model': model_name, 
                'Accuracy': None,
                'Classification Report': str(e)
            })

    return pd.DataFrame(results)



# Charger et prétraiter les nouvelles données
def load_and_preprocess_new_data(new_data_path):
    print("[INFO] Loading new data...")
    time.sleep(0.5)
    new_data = pd.read_csv(new_data_path)
    new_data['cleaned_text'] = new_data['Statement'].apply(preprocess_text)
    print("[INFO] New data loaded and preprocessed successfully.")
    return new_data

# Mettre à jour la fonction principale pour inclure les nouvelles données
def run_project_with_new_data(fake_path, true_path, new_data_path):
    print("[INFO] Starting the project pipeline with new data...")
    time.sleep(0.5)

    # Charger et prétraiter les données d'entraînement et de test
    data = load_and_preprocess_data(fake_path, true_path)

    # Charger et prétraiter les nouvelles données
    new_data = load_and_preprocess_new_data(new_data_path)

    # Diviser les données en ensembles d'entraînement et de test
    print("[INFO] Splitting data into training and testing sets...")
    time.sleep(0.5)
    X = data['cleaned_text']
    y = data['label']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorisation et entraînement
    vectorization_methods = ['bow']#, 'tfidf', 'word2vec']  # Ajoutez 'bert' si nécessaire
    all_results = []

    for method in vectorization_methods:
        print(f"\n[INFO] Processing vectorization method: {method.upper()}...")
        time.sleep(0.5)

        try:
            if method in ['bow', 'tfidf']:
                X_train, vectorizer = vectorize_data(pd.DataFrame({'cleaned_text': X_train_raw}), method=method)
                X_test = vectorizer.transform(X_test_raw)
                X_new = vectorizer.transform(new_data['cleaned_text'])
            elif method == 'word2vec':
                X_train, word2vec_model = vectorize_with_word2vec(pd.DataFrame({'cleaned_text': X_train_raw}))
                X_test, _ = vectorize_with_word2vec(pd.DataFrame({'cleaned_text': X_test_raw}))
                X_new, _ = vectorize_with_word2vec(pd.DataFrame({'cleaned_text': new_data}))
            elif method == 'bert':
                X_train, _ = vectorize_with_bert(pd.DataFrame({'cleaned_text': X_train_raw}))
                X_test, _ = vectorize_with_bert(pd.DataFrame({'cleaned_text': X_test_raw}))
                X_new, _ = vectorize_with_bert(pd.DataFrame({'cleaned_text': new_data}))

            # Entraîner et évaluer les modèles
            print(f"[INFO] Training and evaluating models for {method.upper()}...")
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Naive Bayes': MultinomialNB(),
                'SVM': LinearSVC(),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
            }

            for model_name, model in models.items():
                print(f"[INFO] Training {model_name} with {method.upper()}...")
                model.fit(X_train, y_train)

                # Accuracy sur le jeu de test
                y_test_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                # Accuracy sur les nouvelles données
                y_new_pred = model.predict(X_new)
                new_data_accuracy = accuracy_score(np.abs(new_data['label']-1), y_new_pred)

                print(f"[INFO] {model_name} with {method.upper()} - Test Accuracy: {test_accuracy:.5f}, New Data Accuracy: {new_data_accuracy:.5f}")

                # Ajouter les résultats
                all_results.append({
                    'Vectorizer': method,
                    'Model': model_name,
                    'Test Accuracy': test_accuracy,
                    'New Data Accuracy': new_data_accuracy
                })

        except Exception as e:
            print(f"[ERROR] Failed to process {method}: {str(e)}")
            all_results.append({
                'Vectorizer': method,
                'Model': None,
                'Test Accuracy': None,
                'New Data Accuracy': None,
                'Error': str(e)
            })


    # Entraîner et évaluer le modèle LSTM
    # --- Entraînement sur les données originales ---
    print("\n[INFO] Running deep learning LSTM model on Word2Vec features...")
    _, word2vec_model = vectorize_with_word2vec(pd.DataFrame({'cleaned_text': X_train_raw}))


    tokenized_texts = [text.split() for text in data['cleaned_text']]
    word_embedding_features = prepare_word_embedding_features(tokenized_texts, word2vec_model, max_len=100)
    labels = data['label'].values

    lstm_accuracy, lstm_model, lstm_device = train_and_evaluate_lstm(
        word_embedding_features,
        labels,
        input_size=word_embedding_features.shape[2],
        num_epochs=5
    )

    # --- Prédiction sur les nouvelles données ---
    tokenized_new = [text.split() for text in new_data['cleaned_text']]
    word_embedding_features_new = prepare_word_embedding_features(tokenized_new, word2vec_model, max_len=100)

    new_preds = predict_on_new_data(lstm_model, lstm_device, word_embedding_features_new)

    # Évaluer si tu as les vraies étiquettes :
    if 'label' in new_data.columns:
        new_labels = np.abs(new_data['label'].values -1)
        accuracy_new_data = (new_preds == new_labels).mean()
    else:
        accuracy_new_data = None  # ou tu peux juste afficher les prédictions


    # Convertir les résultats en DataFrame
    final_results = pd.DataFrame(all_results)
    
        # --- Ajout au tableau final ---
    final_results = pd.concat([final_results, pd.DataFrame([{
        'Vectorizer': 'word2vec',
        'Model': 'LSTM',
        'Test Accuracy': lstm_accuracy / 100,
        'New Data Accuracy': accuracy_new_data
    }])], ignore_index=True)

    print("\n[INFO] Project pipeline completed successfully.")
    return final_results




# Exemple d'utilisation
if __name__ == "__main__":
    fake_path = 'data/Fake.csv'
    true_path = 'data/True.csv'
    new_data_path = 'data/corpus_combined_dataset.csv'
    results = run_project_with_new_data(fake_path, true_path, new_data_path)
    print("\nFinal Results:")
    print(results[['Vectorizer', 'Model', 'Test Accuracy', 'New Data Accuracy']])


# # Example usage
# if __name__ == "__main__":
#     fake_path = 'data/Fake.csv'
#     true_path = 'data/True.csv'
#     results = run_project(fake_path, true_path)
#     print("\nFinal Results:")
#     print(results[['Vectorizer', 'Model', 'Accuracy']])