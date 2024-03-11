import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
import skeleton
import main

def run(Model, model_path):
    model_state_dict = torch.load(model_path).state_dict()
    model = Model.load_state_dict(model_state_dict)
    
    with open("test_dataset.json", "r") as file:
        processed_test_data = json.load(file)

    texts_test = [entry["text"] for entry in processed_test_data.values()]
    labels_test = [entry["labels"] for entry in processed_test_data.values()]

    word_to_index = {}
    word_to_index["<unk>"] = 0
    index = 1
    for entry in processed_test_data.values():
        tokens = entry["text"].split()
        for token in tokens:
            if token not in word_to_index:
                word_to_index[token] = index
                index += 1

    label_to_index = {
        "O": 0,
        "B_COURT": 1,
        "I_COURT": 2,
        "B_PETITIONER": 3,
        "I_PETITIONER": 4,
        "B_RESPONDENT": 5,
        "I_RESPONDENT": 6,
        "B_JUDGE": 7,
        "I_JUDGE": 8,
        "B_DATE": 9,
        "I_DATE": 10,
        "B_ORG": 11,
        "I_ORG": 12,
        "B_GPE": 13,
        "I_GPE": 14,
        "B_STATUTE": 15,
        "I_STATUTE": 16,
        "B_PROVISION": 17,
        "I_PROVISION": 18,
        "B_PRECEDENT": 19,
        "I_PRECEDENT": 20,
        "B_CASE_NUMBER": 21,
        "I_CASE_NUMBER": 22,
        "B_WITNESS": 23,
        "I_WITNESS": 24,
        "B_OTHER_PERSON": 25,
        "I_OTHER_PERSON": 26,
    }

    x_test = [
        [word_to_index.get(token, word_to_index["<unk>"]) for token in text.split()]
        for text in texts_test
    ]
    y_test = [[label_to_index[label] for label in entry] for entry in labels_test]
    
    test_dataset = main.CustomDataset(x_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=30, shuffle=False, collate_fn=main.collate_fn
    )

    all_preds_test = []
    all_targets_test = []

    with torch.no_grad():
        for inputs_test, targets_test in test_loader:
            outputs_test = model(inputs_test)
            preds_test = torch.argmax(outputs_test, dim=2).cpu().numpy()
            targets_test = targets_test.cpu().numpy()

            all_preds_test.extend(preds_test)
            all_targets_test.extend(targets_test)

    all_preds_test = np.concatenate(all_preds_test, axis=0)
    all_targets_test = np.concatenate(all_targets_test, axis=0)

    test_accuracy = accuracy_score(all_targets_test, all_preds_test)
    test_macro_f1 = f1_score(all_targets_test, all_preds_test, average="macro")

    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Macro F1: {test_macro_f1:.4f}")
    classification_report_test = classification_report(
        all_targets_test, all_preds_test, target_names=label_to_index.keys()
    )
    print("Classification Report for Test Data:\n", classification_report_test)

if __name__ == "__main__":
    run(skeleton.Rnn, "./models/rnn_glove.pth")