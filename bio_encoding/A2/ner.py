import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
import re
import os

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.LongTensor(self.x[index]), torch.LongTensor(self.y[index])


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets


def load_glove_embeddings(file_path, word_to_index, embedding_dim):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings[word] = coefs

    num_words = len(word_to_index) + 1  # add 1 for the padding token
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, index in word_to_index.items():
        if (embedding := embeddings.get(word)) is not None:
            embedding_matrix[index] = embedding
        else:
            embedding_matrix[index] = embeddings.get("<unk>", np.zeros(embedding_dim))

    return embedding_matrix


def load_word2vec_embeddings(file_path, word_to_index, embedding_dim):
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    embedding_matrix = np.zeros((len(word_to_index) + 1, embedding_dim))

    for word, index in word_to_index.items():
        embedding_matrix[index] = (
            model[word] if word in model else np.zeros(embedding_dim)
        )

    return embedding_matrix


def load_fasttext_embeddings(embedding_file, word_index, embedding_dim):
    embeddings_index = {}
    with open(embedding_file, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


class Rnn(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size, embedding_dim):
        super(Rnn, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), padding_idx=0
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output


class Lstm(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size, embedding_dim):
        super(Lstm, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


class Gru(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size, embedding_dim):
        super(Gru, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), padding_idx=0
        )
        self.gru = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.gru(x)
        output = self.fc(output)
        return output


T1_LABEL_MAP = {
    "O": 0,
    "B_COURT": 1,
    "I_COURT": 2,
    "B_PETITIONER": 3,
    "I_PETITIONER": 4,
    "B_RESPONDENT": 5,
    "I_RESPONDENT": 6,
    "B_JUDGE": 7,
    "I_JUDGE": 8,
    "B_DATE": 9,
    "I_DATE": 10,
    "B_ORG": 11,
    "I_ORG": 12,
    "B_GPE": 13,
    "I_GPE": 14,
    "B_STATUTE": 15,
    "I_STATUTE": 16,
    "B_PROVISION": 17,
    "I_PROVISION": 18,
    "B_PRECEDENT": 19,
    "I_PRECEDENT": 20,
    "B_CASE_NUMBER": 21,
    "I_CASE_NUMBER": 22,
    "B_WITNESS": 23,
    "I_WITNESS": 24,
    "B_OTHER_PERSON": 25,
    "I_OTHER_PERSON": 26,
}

def ner(model_path):
    segmented_model_path = re.split("[/_.]", model_path)
    task, model, word_embeddings = segmented_model_path[-4:-1]

    if model == "biLSTM":
        return

    print(f"Task: {task}, Model: {model}, Word embeddings: {word_embeddings}")

    with open(f"./data/{task}_train_data.json", "r") as file:
        processed_train_data = json.load(file)
    
    with open(f"./data/{task}_test_data.json", "r") as file:
        processed_test_data = json.load(file)

    texts_test = [entry["text"] for entry in processed_test_data.values()]
    labels_test = [entry["labels"] for entry in processed_test_data.values()]

    word_to_index = {}
    word_to_index["<unk>"] = 0
    index = 1
    for entry in processed_train_data.values():
        tokens = entry["text"].split()
        for token in tokens:
            if token not in word_to_index:
                word_to_index[token] = index
                index += 1

    label_to_index = {"O": 0, "B": 1, "I": 2} if task == "t2" else T1_LABEL_MAP

    x_test = [
        [word_to_index.get(token, word_to_index["<unk>"]) for token in text.split()]
        for text in texts_test
    ]
    y_test = [[label_to_index[label] for label in entry] for entry in labels_test]

    test_dataset = CustomDataset(x_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=30, shuffle=False, collate_fn=collate_fn
    )

    model = torch.load(model_path)
    model.eval()

    all_preds_test = []
    all_targets_test = []

    with torch.no_grad():
        for inputs_test, targets_test in test_loader:
            outputs_test = model(inputs_test)
            preds_test = torch.argmax(outputs_test, dim=2).cpu().numpy()
            targets_test = targets_test.cpu().numpy()

            all_preds_test.extend(preds_test)
            all_targets_test.extend(targets_test)

    all_preds_test = np.concatenate(all_preds_test, axis=0)
    all_targets_test = np.concatenate(all_targets_test, axis=0)

    test_accuracy = accuracy_score(all_targets_test, all_preds_test)
    test_macro_f1 = f1_score(all_targets_test, all_preds_test, average="macro")

    print(f"Accuracy: {test_accuracy:.4f}, Macro F1: {test_macro_f1:.4f}")
    classification_report_test = classification_report(
        all_targets_test, all_preds_test, target_names=label_to_index.keys()
    )
    print("Classification Report for Test Data:\n", classification_report_test)
    print("\n\n")

if __name__ == "__main__":
    models = os.listdir("models")
    for model in models:
        ner(f"./models/{model}")  
