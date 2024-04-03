import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

cuda_device = 0  # Set to the index of the GPU you want to use
torch.cuda.set_device(cuda_device)

train_data = pd.read_csv('train.csv', sep='\t', names=['score', 'sentence1', 'sentence2'])
val_data = pd.read_csv('dev.csv', sep='\t', names=['score', 'sentence1', 'sentence2'])

print("Train data shape before filtering:", train_data.shape)
print("Validation data shape before filtering:", val_data.shape)

train_data = train_data.dropna(subset=['sentence2'])
train_data = train_data[train_data['sentence2'].str.strip() != '']

val_data = val_data.dropna(subset=['sentence2'])
val_data = val_data[val_data['sentence2'].str.strip() != '']

print("Train data shape after filtering:", train_data.shape)
print("Validation data shape after filtering:", val_data.shape)

val_data = val_data[pd.to_numeric(val_data['score'], errors='coerce').notnull()]

print("Validation data shape after removing non-numerical scores:", val_data.shape)

model = SentenceTransformer('all-MiniLM-L6-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

val_sentences1 = val_data['sentence1'].tolist()
val_sentences2 = val_data['sentence2'].tolist()
val_embeddings1 = model.encode(val_sentences1, convert_to_tensor=True, device=device)
val_embeddings2 = model.encode(val_sentences2, convert_to_tensor=True, device=device)

cosine_similarity = torch.nn.functional.cosine_similarity(val_embeddings1, val_embeddings2)

scaled_cosine_similarity = cosine_similarity * 5.0  # Scale to the range of scores (0 to 5)

val_data['score'] = val_data['score'].astype(float)

pearson_corr = np.corrcoef(val_data['score'], scaled_cosine_similarity.cpu().numpy())[0, 1]

print("Pearson correlation coefficient:", pearson_corr)
