import json
import joblib
import numpy as np
import torch
import torch.nn as nn

from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader

from src.dataset import load_pubmed_rct

CLASS_MERGE = {"objective": "background"}


class SparseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.tocsr() if sparse.issparse(X) else X
        self.y = np.array(y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if sparse.issparse(x):
            x = x.toarray().ravel()
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def make_loader(X, y, batch_size=256):
    dataset = SparseDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, n_classes=5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    preds_total, y_total = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = torch.argmax(model(X), dim=1).cpu().numpy()
            preds_total.extend(preds)
            y_total.extend(y.numpy())
    return np.array(preds_total), np.array(y_total)


def main():
    print("Carregando dataset e vetorizador...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_pubmed_rct()
    vectorizer = joblib.load("models/best_tfidf_vectorizer.joblib")

    X_test = vectorizer.transform(test_texts)

    # LabelEncoder refeito só para reproduzir o mapeamento determinístico (ordem alfabética) —
    # não há treino acontecendo aqui, é o mesmo mapeamento que o treino original produziu.
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    y_test_enc = label_encoder.transform(test_labels)

    with open("optimized_params/mlp_best.json", "r") as f:
        best_params = json.load(f)

    n_layers = best_params["n_layers"]
    hidden_dims = [best_params[f"hidden_{i}"] for i in range(n_layers)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(
        input_dim=X_test.shape[1],
        hidden_dims=hidden_dims,
        dropout=best_params["dropout"],
        n_classes=len(label_encoder.classes_),
    ).to(device)
    model.load_state_dict(torch.load("models/final_mlp_best.pt", map_location=device))

    test_loader = make_loader(X_test, y_test_enc)
    preds_enc, true_enc = evaluate(model, test_loader, device)

    preds_labels = label_encoder.inverse_transform(preds_enc)
    true_labels = label_encoder.inverse_transform(true_enc)

    preds_collapsed = np.array([CLASS_MERGE.get(l, l) for l in preds_labels])
    true_collapsed = np.array([CLASS_MERGE.get(l, l) for l in true_labels])
    collapsed_classes = sorted(set(true_collapsed))

    test_f1 = f1_score(true_collapsed, preds_collapsed, average="macro")
    print("\nF1 Macro (colapsado):", round(test_f1, 4))
    print(classification_report(true_collapsed, preds_collapsed, labels=collapsed_classes))

    results = {
        "model": "mlp",
        "classes_treino": list(label_encoder.classes_),
        "classes_avaliacao": collapsed_classes,
        "class_merge": CLASS_MERGE,
        "f1_macro": float(test_f1),
        "accuracy": float((preds_collapsed == true_collapsed).mean()),
        "classification_report": classification_report(
            true_collapsed, preds_collapsed, labels=collapsed_classes, output_dict=True
        ),
    }

    with open("results/mlp_test_metrics_collapsed.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
