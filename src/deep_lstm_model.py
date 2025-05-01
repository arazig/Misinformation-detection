# deep_lstm_model.py

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

class WordEmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze(1)

def train_and_evaluate_lstm(X, y, input_size, hidden_size=128, num_layers=1, num_epochs=10, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = WordEmbeddingDataset(X_train, y_train)
    test_dataset = WordEmbeddingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMBinaryClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs >= 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total
    print(f"[LSTM Model] Accuracy: {accuracy:.2f}%")
    
    return accuracy, model, device

def prepare_word_embedding_features(tokenized_texts, embedding_model, max_len=100):
    embedding_dim = embedding_model.vector_size
    X = np.zeros((len(tokenized_texts), max_len, embedding_dim))
    for i, tokens in enumerate(tokenized_texts):
        for j, token in enumerate(tokens[:max_len]):
            if token in embedding_model.wv:
                X[i, j] = embedding_model.wv[token]
    return X

def predict_on_new_data(model, device, X_new, threshold=0.5, batch_size=32):
    model.eval()
    new_dataset = WordEmbeddingDataset(X_new, np.zeros(len(X_new)))  # dummy labels
    new_loader = DataLoader(new_dataset, batch_size=batch_size)

    all_preds = []
    with torch.no_grad():
        for X_batch, _ in new_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = (outputs >= threshold).float().cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)