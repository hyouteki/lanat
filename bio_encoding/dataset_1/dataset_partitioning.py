import json
import numpy as np
from sklearn.model_selection import train_test_split

LABELS_MAP = {
    "COURT": 0,
    "PETITIONER": 1,
    "RESPONDENT": 2,
    "JUDGE": 3,
    "DATE": 4,
    "ORG": 5,
    "GPE": 6,
    "STATUTE": 7,
    "PROVISION": 8,
    "PRECEDENT": 9,
    "CASE_NUMBER": 10,
    "WITNESS": 11,
    "OTHER_PERSON": 12,
}

LABEL_COUNT = 13

with open("NER_TRAIN_JUDGEMENT.json", "r") as file:
    data = json.load(file)

labels = []
for sample in data:
    annotations = sample["annotations"][0]["result"]
    label = np.zeros(LABEL_COUNT)
    if len(annotations) != 0:
        label[LABELS_MAP[annotations[0]["value"]["labels"][0]]] = 1
    labels.append(label)

train_data, val_data, train_label, val_label = train_test_split(
    data, labels, stratify=labels, shuffle=True, test_size=0.15
)

with open("train_dataset_stratified.json", "w") as file:
    json.dump(train_data, file, indent=4)  

with open("val_dataset_stratified.json", "w") as file:
    json.dump(val_data, file, indent=4)  