import os
import json
import pickle
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

PAD_IDX = 0
N_CLASSES = 5


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

    # descarte das sentenças de comprimento zero no treino (decisão já tomada e verificada:
    # 3 sentenças, índices [109836, 129232, 136547], todas da classe 'results')
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


def make_loader(X, y, batch_size, shuffle=True):
    dataset = SequenceDataset(X, y)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )


# =========================
# MODEL
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_tensor, hidden_size, num_layers, dropout,
                 freeze_embeddings, n_classes=N_CLASSES, pad_idx=PAD_IDX):
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
            # nn.LSTM só aplica dropout entre camadas empilhadas; com num_layers=1
            # um valor != 0 gera um UserWarning inofensivo mas evitável.
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)  # *2: concatenação forward+backward

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        # h_n: (num_layers * 2, batch, hidden_size), ordenado [l0_fwd, l0_bwd, ..., lL_fwd, lL_bwd]
        # últimos dois índices = última camada, direções forward e backward
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

    return f1_score(y_total, preds_total, average="macro")


# =========================
# OPTUNA OBJECTIVE
# =========================
def objective(trial, X_train, y_train, X_val, y_val, embedding_tensor, device):

    # -------- arquitetura --------
    hidden_size = trial.suggest_int("hidden_size", 64, 512, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    freeze_embeddings = trial.suggest_categorical("freeze_embeddings", [True, False])

    # -------- otimização --------
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    # -------- modelo --------
    model = LSTMClassifier(
        embedding_tensor=embedding_tensor,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    ).to(device)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train, batch_size)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    for epoch in range(10):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        f1 = evaluate(model, val_loader, device)

        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return f1


# =========================
# MAIN
# =========================
def main():
    X_train, y_train, X_val, y_val, X_test, y_test, E = load_lstm_data()

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_tensor = torch.tensor(E, dtype=torch.float32)

    study = optuna.create_study(
        direction="maximize",
        study_name="lstm_bioembedding",
        storage="sqlite:///optimized_params/lstm.db",
        load_if_exists=True,
    )

    n_trials = int(os.getenv("N_TRIALS", 50))

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train_enc, X_val, y_val_enc, embedding_tensor, device
        ),
        n_trials=n_trials,
    )

    print("\nBest F1:", study.best_value)
    print("Best params:", study.best_params)

    os.makedirs("optimized_params", exist_ok=True)
    with open("optimized_params/lstm_best.json", "w") as f:
        json.dump(study.best_params, f, indent=4)


if __name__ == "__main__":
    main()
