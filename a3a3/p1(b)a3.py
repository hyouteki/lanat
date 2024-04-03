import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import matplotlib.pyplot as plt
#gpu
cuda_device=0  
torch.cuda.set_device(cuda_device)
#changed 'setence' to 'sentence'
trainingdata=pd.read_csv('train.csv', sep='\t', names=['score','sentence1','sentence2'])
validata=pd.read_csv('dev.csv',sep='\t',names=['score','sentence1','sentence2'])
print("Training shape before:",trainingdata.shape)
print("Validation shape before:",validata.shape)
trainingdata=trainingdata.dropna(subset=['sentence2'])
trainingdata=trainingdata[trainingdata['sentence2'].str.strip()!='']
validata=validata.dropna(subset=['sentence2'])
validata=validata[validata['sentence2'].str.strip()!='']
print("Training Data",trainingdata)
print("Train shape after:",trainingdata.shape)
print("Validation Data",validata)
print("Validation shape after:",validata.shape)
validata = validata[pd.to_numeric(validata['score'], errors='coerce').notnull()]
print("Validation shape after removing invalid scores:", validata.shape)
model=SentenceTransformer('all-MiniLM-L6-v2')
if torch.cuda.is_available():
  device=torch.device("cuda")
else:
  device=torch.device("cpu")
model.to(device)
valsentence1=validata['sentence1'].tolist()
valsentence2=validata['sentence2'].tolist()
valembedding1=model.encode(valsentence1,convert_to_tensor=True,device=device)
valembedding2=model.encode(valsentence2,convert_to_tensor=True,device=device)
cossimilarity=torch.nn.functional.cosine_similarity(valembedding1, valembedding2)
# "must appropriately scale the cosine similarity to the score column's scale"
scaledcossimilarity=cossimilarity*5.0 
validata['score']=validata['score'].astype(float)
pearson_corr=np.corrcoef(validata['score'], scaledcossimilarity.cpu().numpy())[0, 1]
print("PCC:",pearson_corr)
