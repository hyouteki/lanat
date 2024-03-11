import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
import skeleton
import main


def run(Model, load_embeddings, embedding_dim, embeddings_path, model_path):
    with open("processed_test_data.json", "r") as file:
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

    label_to_index = {"O": 0, "B": 1, "I": 2} 
    embedding_matrix = load_embeddings(embeddings_path, word_to_index, embedding_dim)

    x_test = [
        [word_to_index.get(token, word_to_index["<unk>"]) for token in text.split()]
        for text in texts_test
    ]
    y_test = [[label_to_index[label] for label in entry] for entry in labels_test]

    test_dataset = main.CustomDataset(x_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=30, shuffle=False, collate_fn=main.collate_fn
    )

    hidden_size = 100
    output_size = len(label_to_index)
    model = Model(
        embedding_matrix, hidden_size, output_size, embedding_dim
    ).load_state_dict(torch.load(model_path))

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
    run(
        skeleton.Rnn,
        skeleton.load_glove_embeddings,
        100,
        "../word_embeddings/glove.6B.100d.txt",
        "./models/rnn_glove.pth",
    )
