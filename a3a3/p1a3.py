import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# Check if GPU is available and set device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load the dataset
train_data = pd.read_csv('train.csv', sep='\t')
print(train_data.columns)

val_data = pd.read_csv('dev.csv', sep='\t')
print(val_data.columns)
# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom dataset class
class SimilarityDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.sentences1 = []
        self.sentences2 = []
        self.scores = []

        for _, row in data.iterrows():
            if row['sentence1'] and row['sentence2'] and row['score']:
                self.sentences1.append(str(row['sentence1']))
                self.sentences2.append(str(row['sentence2']))
                self.scores.append(row['score'])

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, idx):
        sentence1 = self.sentences1[idx]
        sentence2 = self.sentences2[idx]
        score = self.scores[idx]
        encoding = tokenizer.encode_plus(sentence1, sentence2, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)  # Remove the extra dimension
        attention_mask = encoding['attention_mask'].squeeze(0)  # Remove the extra dimension
        return input_ids, attention_mask, score

# Create datasets and data loaders
train_dataset = SimilarityDataset(train_data)
val_dataset = SimilarityDataset(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Custom model class
class BertForSimilarity(torch.nn.Module):
    def __init__(self):
        super(BertForSimilarity, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(768, 1)
        self.double()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        output = self.fc(pooled_output)
        return output

# Initialize model, optimizer, and loss function
model = BertForSimilarity().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.MSELoss()

# Training function
def train(epochs):
    training_losses = []
    validation_losses = []
    training_pearson_corrs = []
    validation_pearson_corrs = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        # Training loop
        model.train()
        train_loss = 0.0
        train_pearson_corr = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask).squeeze(1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate Pearson correlation coefficient
            pred_scores = outputs.detach().cpu().numpy()
            true_scores = labels.detach().cpu().numpy()
            train_pearson_corr += np.corrcoef(pred_scores, true_scores)[0, 1]

        train_loss /= len(train_dataloader)
        train_pearson_corr /= len(train_dataloader)
        training_losses.append(train_loss)
        training_pearson_corrs.append(train_pearson_corr)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_pearson_corr = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                inputs, labels = batch
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device)
                outputs = model(input_ids, attention_mask).squeeze(1)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                # Calculate Pearson correlation coefficient
                pred_scores = outputs.detach().cpu().numpy()
                true_scores = labels.detach().cpu().numpy()
                val_pearson_corr += np.corrcoef(pred_scores, true_scores)[0, 1]

        val_loss /= len(val_dataloader)
        val_pearson_corr /= len(val_dataloader)
        validation_losses.append(val_loss)
        validation_pearson_corrs.append(val_pearson_corr)

        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Training Pearson Correlation: {train_pearson_corr:.4f}')
        print(f'Validation Pearson Correlation: {val_pearson_corr:.4f}')

    return training_losses, validation_losses, training_pearson_corrs, validation_pearson_corrs

# Train the model
epochs = 1
training_losses, validation_losses, training_pearson_corrs, validation_pearson_corrs = train(epochs)

torch.save(model, 'model.pt')

# Plot the loss and Pearson correlation
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), training_losses, label='Training Loss')
plt.plot(range(epochs), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(epochs), training_pearson_corrs, label='Training Pearson Correlation')
plt.plot(range(epochs), validation_pearson_corrs, label='Validation Pearson Correlation')
plt.xlabel('Epochs')
plt.ylabel('Pearson Correlation')
plt.title('Pearson Correlation Plot')
plt.legend()
plt.tight_layout()
plt.savefig("plot.jpg")
