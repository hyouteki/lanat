import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model = model.to(DEVICE)

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

TRAIN_SPLIT_SIZE = 20000
NUM_EPOCHS = 18
DATA_LOADER_BATCH_SIZE = 32

MAX_SRC_LEN = 512
MAX_TGT_LEN = 128

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
class WMT16Dataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.data = load_dataset('wmt16', SRC_LANGUAGE + "-" + TGT_LANGUAGE, split=self.split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        srcText = self.data[index]['translation'][SRC_LANGUAGE]
        tgtText = self.data[index]['translation'][TGT_LANGUAGE]
        return srcText, tgtText

from tqdm import tqdm
def trainEpoch(model, tokenizer, trainDataLoader):
    losses = 0
    model.train()
    for tgt, src in tqdm(trainDataLoader):
        encoding = tokenizer(
            [f"translate English to German: {sequence}" for sequence in src],
            padding="longest",
            max_length=MAX_SRC_LEN,
            truncation=True,
            return_tensors="pt",
        )
        targetEncoding = tokenizer(
            tgt,
            padding="longest",
            max_length=MAX_TGT_LEN,
            truncation=True,
            return_tensors="pt",
        )
        labels = targetEncoding.input_ids
        inputIds, attentionMasks = encoding.input_ids, encoding.attention_mask
        labels[labels == tokenizer.pad_token_id] = -100
        loss = model(input_ids=inputIds.to(DEVICE),
                     attention_mask=attentionMasks.to(DEVICE),
                     labels=labels.to(DEVICE)).loss
        loss.backward()
        losses += loss.item()
    return losses / len(list(trainDataLoader))


def valEpoch(model, tokenizer, valDataLoader):
    losses = 0
    model.eval()
    for tgt, src in tqdm(valDataLoader):
        encoding = tokenizer(
            [f"translate English to German: {sequence}" for sequence in src],
            padding="longest",
            max_length=MAX_SRC_LEN,
            truncation=True,
            return_tensors="pt",
        )
        targetEncoding = tokenizer(
            tgt,
            padding="longest",
            max_length=MAX_TGT_LEN,
            truncation=True,
            return_tensors="pt",
        )
        labels = targetEncoding.input_ids
        inputIds, attentionMasks = encoding.input_ids, encoding.attention_mask
        labels[labels == tokenizer.pad_token_id] = -100
        loss = model(input_ids=inputIds.to(DEVICE),
                     attention_mask=attentionMasks.to(DEVICE),
                     labels=labels.to(DEVICE)).loss
        loss.backward()
        losses += loss.item()
    return losses / len(list(valDataLoader))

trainDataset = WMT16Dataset(split=f"train[:{TRAIN_SPLIT_SIZE}]")
trainDataLoader = DataLoader(trainDataset, batch_size=DATA_LOADER_BATCH_SIZE, shuffle=False)

valDataset = WMT16Dataset(split=f"validation")
valDataLoader = DataLoader(valDataset, batch_size=DATA_LOADER_BATCH_SIZE, shuffle=False)

from timeit import default_timer as timer
with tqdm(total=NUM_EPOCHS, desc="Evaluating") as pbar:
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = trainEpoch(model, tokenizer, trainDataLoader)
        end_time = timer()
        val_loss = valEpoch(model, tokenizer, valDataLoader)
        print(f"Epoch: {epoch} Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, ",
              f"Epoch time = {(end_time - start_time):.3f}s")
        pbar.update(1)

torch.save(model, "FineTunedT5SmallModel.pt")
