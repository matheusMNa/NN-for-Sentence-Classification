import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

PAD_IDX = 0

# Colapso pós-hoc, aplicado só na avaliação final de teste — não no treino/validação
CLASS_MERGE = {"objective": "background"}


# =========================
# DATA LOADING
# =========================
def load_lstm_data(data_dir="data/processed"):
    data_dir = Path(data_dir)

    with open(data_dir / "sequences_pubmed_rct20k.pkl", "rb") as f:
        seq_data = pickle.load(f)

    X_train, y_train = seq_data["train"], list(seq_data["y_train"])
    X_val, y_val = seq_data["val"], list(seq_data["y_val"])
    X_test, y_test = seq_data["test"], list(seq_data["y_test"])

    # descarte das sentenças de comprimento zero no treino (mesma decisão da HPO)
    lengths = np.array([len(s) for s in X_train])
    mask = lengths > 0
    X_train = [s for s, m in zip(X_train, mask) if m]
    y_train = [y for y, m in zip(y_train, mask) if m]

    E = np.load(data_dir / "embedding_matrix.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test, E


# =========================
# DATASET / DATALOADER
# =========================
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = np.array(y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), self.y[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


def make_loader(X, y, batch_size, shuffle=True, num_workers=2):
    dataset = SequenceDataset(X, y)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )


# =========================
# MODEL
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_tensor, hidden_size, num_layers, dropout,
                 freeze_embeddings, n_classes, pad_idx=PAD_IDX):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor, freeze=freeze_embeddings, padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_tensor.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h_final = self.dropout(h_final)
        return self.fc(h_final)


# =========================
# TRAIN / EVAL
# =========================
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for X, lengths, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X, lengths)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds_total, y_total = [], []

    with torch.no_grad():
        for X, lengths, y in loader:
            X = X.to(device)
            logits = model(X, lengths)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_total.extend(preds)
            y_total.extend(y.numpy())

    return np.array(preds_total), np.array(y_total)


# =========================
# MAIN
# =========================
def main():
    print("Carregando dados...")
    X_train, y_train, X_val, y_val, X_test, y_test, E = load_lstm_data()

    # LabelEncoder sobre as 5 classes originais — nenhum colapso até aqui
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    n_classes = len(label_encoder.classes_)
    print(f"Classes de treino ({n_classes}): {list(label_encoder.classes_)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    embedding_tensor = torch.tensor(E, dtype=torch.float32)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_enc),
        y=y_train_enc
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    with open("optimized_params/lstm_best.json", "r") as f:
        best_params = json.load(f)
    print("Hiperparâmetros carregados:", best_params)

    model = LSTMClassifier(
        embedding_tensor=embedding_tensor,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        freeze_embeddings=best_params["freeze_embeddings"],
        n_classes=n_classes,
    ).to(device)

    opt_name = best_params["optimizer"]
    lr = best_params["lr"]
    wd = best_params["weight_decay"]

    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)

    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

    batch_size = best_params["batch_size"]
    train_loader = make_loader(X_train, y_train_enc, batch_size)
    val_loader = make_loader(X_val, y_val_enc, batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test_enc, batch_size, shuffle=False)

    # -------- treino, com early stopping sobre F1 macro das 5 classes --------
    best_val_f1 = 0.0
    patience = 10
    epochs_without_improvement = 0
    n_epochs = 50

    print("Iniciando treino final...")

    for epoch in range(n_epochs):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_preds, val_true = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_true, val_preds, average="macro")  # 5 classes, sem colapso

        print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Val F1 Macro (5 classes): {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "models/final_lstm_best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping na epoch {epoch+1}.")
                break

    # -------- avaliação final no teste, com colapso pós-hoc --------
    print("Carregando melhor modelo para avaliação no teste...")
    model.load_state_dict(torch.load("models/final_lstm_best.pt"))

    test_preds_enc, test_true_enc = evaluate(model, test_loader, device)

    # decodifica de volta para nomes de classe e só então aplica o colapso
    test_preds_labels = label_encoder.inverse_transform(test_preds_enc)
    test_true_labels = label_encoder.inverse_transform(test_true_enc)

    test_preds_collapsed = np.array([CLASS_MERGE.get(l, l) for l in test_preds_labels])
    test_true_collapsed = np.array([CLASS_MERGE.get(l, l) for l in test_true_labels])

    collapsed_classes = sorted(set(test_true_collapsed))

    test_f1 = f1_score(test_true_collapsed, test_preds_collapsed, average="macro")

    print("\nResultados - Teste Final (com objective colapsado em background):")
    print("F1 Macro:", round(test_f1, 4))
    print("\nReport de Classificação:")
    print(classification_report(test_true_collapsed, test_preds_collapsed, labels=collapsed_classes))

    results = {
        "model": "lstm",
        "classes_treino": list(label_encoder.classes_),
        "classes_avaliacao": collapsed_classes,
        "class_merge": CLASS_MERGE,
        "best_params": best_params,
        "f1_macro": float(test_f1),
        "accuracy": float((test_preds_collapsed == test_true_collapsed).mean()),
        "classification_report": classification_report(
            test_true_collapsed, test_preds_collapsed, labels=collapsed_classes, output_dict=True
        ),
    }

    with open("results/lstm_test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
