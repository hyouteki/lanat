from torch.utils.data import Dataset
from datasets import load_dataset
class WMT16Dataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("wmt16", "de-en", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        srcText = self.data[index]["translation"]["en"]
        tgtText = self.data[index]["translation"]["de"]
        return srcText, tgtText

from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

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
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
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
    bertPrecisions = bertScores['precision']
    bertRecalls = bertScores['recall']
    bertF1s = bertScores['f1']
    print(f"BERT avg precision: {(sum(bertPrecisions)/len(bertPrecisions))}")
    print(f"BERT avg recall: {(sum(bertRecalls)/len(bertRecalls))}")
    print(f"BERT avg f1: {(sum(bertF1s)/len(bertF1s))}")

evaluateModel(WMT16Dataset(split="validation"))
evaluateModel(WMT16Dataset(split="test"))
