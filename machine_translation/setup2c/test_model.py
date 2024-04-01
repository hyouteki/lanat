import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = torch.load("FineTunedT5SmallModel.pt")
model = model.to(DEVICE)

SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"

DATA_LOADER_BATCH_SIZE = 32

MAX_SRC_LEN = 512
MAX_TGT_LEN = 128

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
class WMT16Dataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("wmt16", f"{SRC_LANGUAGE}-{TGT_LANGUAGE}", split=self.split)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        srcText = self.data[index]["translation"][SRC_LANGUAGE]
        tgtText = self.data[index]["translation"][TGT_LANGUAGE]
        return srcText, tgtText

import evaluate
from tqdm import tqdm
from pprint import pprint
def evaluateModel(dataIter):
    srcSamples, tgtSamples = [], []
    for dataSample in dataIter:
        srcSamples.append(f"translate English to German: {dataSample[0]}")
        tgtSamples.append(dataSample[1])
    inputs = tokenizer(srcSamples, return_tensors="pt", padding=True)
    outputSequences = model.generate(
        input_ids=inputs["input_ids"].to(DEVICE),
        attention_mask=inputs["attention_mask"].to(DEVICE),
        do_sample=False,
    )
    translatedSamples = tokenizer.batch_decode(outputSequences, skip_special_tokens=True)
    for maxOrder in range(1, 5):
        print(f"\nBLEU{maxOrder}")
        print(evaluate.load("bleu").compute(predictions=translatedSamples, references=tgtSamples,
                                            max_order=maxOrder))
    print(f"METEOR: {evaluate.load('meteor').compute(predictions=translatedSamples, references=tgtSamples)}")
    bertScores = evaluate.load("bertscore").compute(predictions=translatedSamples,
                                                    references=tgtSamples, lang="de")
    bertPrecisions = bertScores["precision"]
    bertRecalls = bertScores["recall"]
    bertF1s = bertScores["f1"]
    print(f"BERT avg precision: {(sum(bertPrecisions)/len(bertPrecisions))}")
    print(f"BERT avg recall: {(sum(bertRecalls)/len(bertRecalls))}")
    print(f"BERT avg f1: {(sum(bertF1s)/len(bertF1s))}")

evaluateModel(WMT16Dataset(split="validation"))
evaluateModel(WMT16Dataset(split="test"))
