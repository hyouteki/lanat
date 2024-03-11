#take models <.pth> as input and run them

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

def run_model(model_path, name):
    with open("processed_test_data.json", "r") as file:
        processed_test_data = json.load(file)

    texts_test = [entry["text"] for entry in processed_test_data.values()]
    labels_test = [entry["labels"] for entry in processed_test_data.values()]

    # Your word_to_index, label_to_index, and embedding_matrix loading code here

    x_test = [[word_to_index.get(token, word_to_index["<unk>"]) for token in text.split()] for text in texts_test]
    y_test = [[label_to_index[label] for label in entry] for entry in labels_test]

    test_dataset = CustomDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMCRF(embedding_matrix, hidden_size=128, output_size=len(label_to_index), embedding_dim=your_embedding_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

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

    test_macro_f1 = f1_score(all_targets_test, all_preds_test, average="macro")

    print(f"Model: {name}")
    print(f"Final Test Macro F1: {test_macro_f1:.4f}")
    print("=" * 30)

if __name__ == "__main__":
    model_paths = ["models/gru_fasttext.pth", "models/gru_glove.pth","models/gru_word2vec.pth", "models/lstm_fasttext.pth"]  # Add your model paths
    model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]  

    for model_path, model_name in zip(model_paths, model_names):
        run_model(model_path, model_name)


