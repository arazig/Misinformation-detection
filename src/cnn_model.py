import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

'''# CNN Model for Fake News Detection

Explications : 

- prepare_data_for_cnn : Convertit les textes en séquences numériques et les remplit pour qu'ils aient une longueur fixe.
- build_cnn_model : Définit un modèle CNN avec une couche d'embedding, une convolution 1D, un pooling global, et des couches fully connected.
- train_and_evaluate_cnn_model : Entraîne le modèle et affiche un rapport de classification.

'''
# Préparer les données pour le CNN
def prepare_data_for_cnn(texts, labels, max_words=10000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences, labels, tokenizer

# Construire le modèle CNN
def build_cnn_model(max_words=10000, max_len=100, embedding_dim=50):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Pour une classification binaire
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Entraîner et évaluer le modèle CNN
def train_and_evaluate_cnn_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=10):
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return history