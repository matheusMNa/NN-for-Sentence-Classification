import os
import json
import pickle
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 0
N_CLASSES = 5
SEED = int(os.getenv("SEED", 42))
# workers do DataLoader limitados pelas CPUs realmente visíveis ao processo (o processo
# principal também consome uma CPU); com 1 CPU visível isso resulta em 0 workers
NUM_WORKERS = int(os.getenv("NUM_WORKERS", max(0, min(4, len(os.sched_getaffinity(0)) - 1))))
# Conjuntos de larguras de filtro explorados pelo Optuna. Strings (e não tuplas)
# para que o valor seja serializável no storage SQLite e no JSON final.
KERNEL_SETS = {
    "3": (3,),
    "4": (4,),
    "5": (5,),
    "2-3": (2, 3),
    "2-3-4": (2, 3, 4),
    "3-4-5": (3, 4, 5),
    "2-3-4-5": (2, 3, 4, 5),
}

# Comprimento mínimo de qualquer sequência no batch = maior largura possível de filtro.
# Garante L >= k para todo k, isto é, L_out = L - k + 1 >= 1.
MIN_LEN = max(max(v) for v in KERNEL_SETS.values())


# =========================
# DATA LOADING
# (idêntico ao hpo_lstm.py, para que os dois modelos vejam exatamente os mesmos dados)
# =========================
def load_cnn_data(data_dir="data/processed"):
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

    # em validação e teste nenhuma sentença pode ser removida; se houvesse alguma de
    # comprimento zero, o conjunto de avaliação mudaria e a comparação com a LSTM seria inválida
    assert min(len(s) for s in X_val) > 0, "sentença de comprimento zero na validação"
    assert min(len(s) for s in X_test) > 0, "sentença de comprimento zero no teste"

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
    # padding à direita, dinâmico por batch: L_b = max(L_maior_no_batch, MIN_LEN)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    if padded.size(1) < MIN_LEN:
        padded = F.pad(padded, (0, MIN_LEN - padded.size(1)), value=PAD_IDX)
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
class TextCNN(nn.Module):
    """CNN 1D (Kim, 2014) com max-over-time pooling mascarado.

    Para uma sentença de comprimento real L e um filtro de largura k, só entram no
    max as posições t <= max(L - k, 0). Assim a saída não depende de quanto padding o
    batch adicionou. Para L < k a única janela válida é t = 0 (parcialmente preenchida).
    """

    def __init__(self, embedding_tensor, kernel_sizes, num_filters, dropout,
                 freeze_embeddings, n_classes=N_CLASSES, pad_idx=PAD_IDX):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor, freeze=freeze_embeddings, padding_idx=pad_idx
        )

        emb_dim = embedding_tensor.shape[1]
        self.kernel_sizes = list(kernel_sizes)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, kernel_size=k) for k in self.kernel_sizes]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), n_classes)

    def forward(self, x, lengths):
        # (B, L) -> (B, d, L), formato esperado por nn.Conv1d
        embedded = self.embedding(x).transpose(1, 2)
        lengths = lengths.to(embedded.device)

        pooled = []
        for k, conv in zip(self.kernel_sizes, self.convs):
            c = conv(embedded)  # (B, F, L - k + 1)
            positions = torch.arange(c.size(2), device=c.device).unsqueeze(0)  # (1, L-k+1)
            t_max = (lengths - k).clamp(min=0).unsqueeze(1)                     # (B, 1)
            invalid = positions > t_max                                          # (B, L-k+1)
            c = c.masked_fill(invalid.unsqueeze(1), float("-inf"))
            # relu(max(c)) == max(relu(c)), pois a ReLU é não decrescente
            pooled.append(torch.relu(c.max(dim=2).values))

        z = torch.cat(pooled, dim=1)  # (B, F * |K|)
        z = self.dropout(z)
        return self.fc(z)


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
def objective(trial, X_train, y_train, X_val, y_val, embedding_tensor, device, n_epochs):

    # -------- arquitetura --------
    kernel_set = trial.suggest_categorical("kernel_set", list(KERNEL_SETS.keys()))
    num_filters = trial.suggest_int("num_filters", 64, 256, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    freeze_embeddings = trial.suggest_categorical("freeze_embeddings", [True, False])

    # -------- otimização (mesmos espaços da LSTM) --------
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    torch.manual_seed(SEED + trial.number)

    # -------- modelo --------
    model = TextCNN(
        embedding_tensor=embedding_tensor,
        kernel_sizes=KERNEL_SETS[kernel_set],
        num_filters=num_filters,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    ).to(device)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    # sem pesos de classe, como no hpo_lstm.py
    loss_fn = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train, batch_size)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    for epoch in range(n_epochs):
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
    X_train, y_train, X_val, y_val, X_test, y_test, E = load_cnn_data()

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_tensor = torch.tensor(E, dtype=torch.float32)

    os.makedirs("optimized_params", exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="cnn_bioembedding",
        storage="sqlite:///optimized_params/cnn.db",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )

    n_trials = int(os.getenv("N_TRIALS", 50))
    n_epochs = int(os.getenv("N_EPOCHS", 10))

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train_enc, X_val, y_val_enc, embedding_tensor, device, n_epochs
        ),
        n_trials=n_trials,
    )

    print("\nBest F1:", study.best_value)
    print("Best params:", study.best_params)

    with open("optimized_params/cnn_best.json", "w") as f:
        json.dump(study.best_params, f, indent=4)


if __name__ == "__main__":
    main()
