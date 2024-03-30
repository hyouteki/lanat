import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class SimilarityDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Check if all required columns are present
        if 'score' not in row or pd.isna(row['score']) or 'sentence1' not in row or pd.isna(row['sentence1']) or 'sentence2' not in row or pd.isna(row['sentence2']):
            # Skip this row if any required field is missing
            return None

        score = float(row['score'])  # Convert score to float
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']

        # Check if the combined length of the two sentences exceeds max_len
        combined_length = len(sentence1.split()) + len(sentence2.split())
        if combined_length > self.max_len:
            # Handle the case where the combined length exceeds max_len
            # Truncate the longer sentence to fit within max_len
            max_len_sentence1 = self.max_len // 2
            max_len_sentence2 = self.max_len - max_len_sentence1

            sentence1_tokens = sentence1.split()[:max_len_sentence1]
            sentence2_tokens = sentence2.split()[:max_len_sentence2]

            sentence1 = ' '.join(sentence1_tokens)
            sentence2 = ' '.join(sentence2_tokens)

        inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        token_type_ids = inputs['token_type_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return input_ids, token_type_ids, attention_mask, torch.tensor(score, dtype=torch.double)

train_data = pd.read_csv('train.csv', sep='\t', names=['score', 'sentence1', 'sentence2'])
val_data = pd.read_csv('dev.csv', sep='\t', names=['score', 'sentence1', 'sentence2'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128

train_dataset = SimilarityDataset(train_data, tokenizer, max_len)
val_dataset = SimilarityDataset(val_data, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

class SimilarityModel(nn.Module):
    def __init__(self):
        super(SimilarityModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.to(torch.double)  # Convert BERT model to double precision
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 1, dtype=torch.double)  # Create linear layer with double precision

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.fc(dropout_output)
        return logits.squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimilarityModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

epochs = 3
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for input_ids, token_type_ids, attention_mask, scores in train_loader:
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        scores = scores.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, token_type_ids, attention_mask)
        loss = torch.mean((outputs - scores) ** 2)  # Mean Squared Error
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    # Evaluation on validation set
    model.eval()
    total_val_loss = 0
    all_outputs = []
    all_scores = []
    with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, scores in val_loader:
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            scores = scores.to(device)

            outputs = model(input_ids, token_type_ids, attention_mask)
            loss = torch.mean((outputs - scores) ** 2)  # Mean Squared Error
            total_val_loss += loss.item()

            all_outputs.extend(outputs.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    val_losses.append(total_val_loss / len(val_loader))

    # Calculate Pearson correlation
    pearson_corr = np.corrcoef(all_outputs, all_scores)[0, 1]
    print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}, Pearson Corr: {pearson_corr:.4f}')

# Plot the loss curves
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), train_losses, label='Training Loss')
plt.plot(range(epochs), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()

# Analyze and explain the plots
print("Loss Plots Analysis:")
print("The training and validation loss curves show the model's performance during training.")
print("The training loss decreases steadily, indicating that the model is learning and improving on the training set.")
print("The validation loss also decreases, but it may fluctuate or plateau, indicating the model's generalization ability.")
print("Ideally, both training and validation losses should decrease and converge, with a small gap between them.")
print("A large gap between the two curves may indicate overfitting or underfitting.")
print("The Pearson correlation coefficient measures the linear correlation between the predicted and actual scores.")
print("A higher value (closer to 1) indicates a stronger positive correlation and better model performance.")
