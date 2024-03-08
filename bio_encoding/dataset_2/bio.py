import json
import pprint

def bio_chunking(dataset):
    processed_data = {}
    for idx, entry in enumerate(dataset, start=1):
        text = entry["raw_words"]  
        labels = ["O"] * len(entry["words"])
        
        for aspect in entry["aspects"]:
            for i in range(aspect["from"], aspect["to"]):
                if i == aspect["from"]:
                    labels[i] = "B"
                else:
                    labels[i] = "I"
        

        processed_data[str(idx)] = {"text": text, "labels": labels}
    
    return processed_data

input_data = [
    {
        "raw_words": "I charge it at night and skip taking the cord with me because of the good battery life .",
        "words": ["I", "charge", "it", "at", "night", "and", "skip", "taking", "the", "cord", "with", "me", "because", "of", "the", "good", "battery", "life", "."],
        "aspects": [{"index": 0, "from": 16, "to": 18, "polarity": "POS", "term": ["battery", "life"]}],
        "opinions": [{"index": 0, "from": 15, "to": 16, "term": ["good"]}]
    },
]


f=open("Laptop_Review_Train.json", "r")
train_data = json.load(f)
processed_train_data = bio_chunking(train_data)

f=open("Laptop_Review_Test.json", "r")
test_data = json.load(f)
processed_test_data = bio_chunking(test_data)

f=open("Laptop_Review_Val.json", "r")
val_data = json.load(f)
processed_val_data = bio_chunking(val_data)

with open("processed_train_data.json", "w") as f:
    json.dump(processed_train_data, f)

with open("processed_test_data.json", "w") as f:
    json.dump(processed_test_data, f)

with open("processed_val_data.json", "w") as f:
    json.dump(processed_val_data, f)