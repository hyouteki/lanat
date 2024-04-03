import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
#using gpu for faster computation
cuda_device = 0  
torch.cuda.set_device(cuda_device)
#importing requisite files, changed 'setence' to 'sentence'
train_data = pd.read_csv('train.csv', sep='\t', names=['score', 'sentence1', 'sentence2'])
val_data = pd.read_csv('dev.csv', sep='\t', names=['score', 'sentence1', 'sentence2'])
print("Training shape before:", train_data.shape)
print("Validation shape before:", val_data.shape)
train_data = train_data.dropna(subset=['sentence2'])
train_data = train_data[train_data['sentence2'].str.strip() != '']
val_data = val_data.dropna(subset=['sentence2'])
val_data = val_data[val_data['sentence2'].str.strip() != '']
print("Train shape after:", train_data.shape)
print("Validation shape after:", val_data.shape)
val_data = val_data[pd.to_numeric(val_data['score'], errors='coerce').notnull()]
print("Validation shape after removing invalid scores:", val_data.shape)
model = SentenceTransformer('all-MiniLM-L6-v2')
if torch.cuda.is_available():
  device=torch.device("cuda")
else:
  device=torch.device("cpu")
model.to(device)
val_sentences1 = val_data['sentence1'].tolist()
val_sentences2 = val_data['sentence2'].tolist()
val_embeddings1 = model.encode(val_sentences1, convert_to_tensor=True, device=device)
val_embeddings2 = model.encode(val_sentences2, convert_to_tensor=True, device=device)
cosine_similarity = torch.nn.functional.cosine_similarity(val_embeddings1, val_embeddings2)
# "must appropriately scale the cosine similarity to the score column's scale"
scaled_cosine_similarity = cosine_similarity*5.0 
val_data['score'] = val_data['score'].astype(float)
pearson_corr = np.corrcoef(val_data['score'], scaled_cosine_similarity.cpu().numpy())[0, 1]
print("PCC:", pearson_corr)
