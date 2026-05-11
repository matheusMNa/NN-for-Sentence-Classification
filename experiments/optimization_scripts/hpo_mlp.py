import os
import json
import numpy as np
import joblib
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader

from src.dataset import load_pubmed_rct

from torch.utils.data import Dataset

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

def make_loader(X, y, batch_size, shuffle=True):
    dataset = SparseDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

#=================
# MODEL
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 5))  # 5 classes

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# TRAIN ONE EPOCH
# =========================
def train_epoch(model, loader, optimizer, loss_fn, device):

    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(X)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# EVALUATION
# =========================
def evaluate(model, loader, device):

    model.eval()
    preds_total, y_total = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            preds_total.extend(preds)
            y_total.extend(y.numpy())

    return f1_score(y_total, preds_total, average="macro")


# =========================
# OPTUNA OBJECTIVE
# =========================
def objective(trial, X_train, y_train, X_val, y_val, device):

    # -------- architecture --------
    n_layers = trial.suggest_int("n_layers", 1, 3)

    hidden_dims = [
        trial.suggest_int(f"hidden_{i}", 64, 1024, log=True)
        for i in range(n_layers)
    ]

    dropout = trial.suggest_float("dropout", 0.1, 0.6)

    # -------- optimization --------
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    optimizer_name = trial.suggest_categorical(
        "optimizer", ["adam", "sgd", "rmsprop"]
    )

    # -------- model --------
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    # -------- optimizer --------
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9
        )

    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    loss_fn = nn.CrossEntropyLoss()

    # -------- loaders --------
    train_loader = make_loader(X_train, y_train, batch_size)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    # -------- training --------
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

    # -------- data --------
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_pubmed_rct()

    # -------- TF-IDF --------
    vectorizer = joblib.load("models/best_tfidf_vectorizer.joblib")

    X_train = vectorizer.transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(train_labels)
    y_val = label_encoder.transform(val_labels)
    y_test = label_encoder.transform(test_labels)

    # -------- device --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- study --------
    study = optuna.create_study(
        direction="maximize",
        study_name="mlp_tfidf",
        storage="sqlite:///optimized_params/mlp.db",
        load_if_exists=True
    )
    
    n_trials = int(os.getenv("N_TRIALS", 100))
    n_trials = int(os.getenv("N_TRIALS", 100))

    study.optimize(
        lambda trial: objective(
            trial,
            X_train, y_train,
            X_val, y_val,
            device
        ),
        n_trials=n_trials
    )

    # -------- results --------
    print("\nBest F1:", study.best_value)
    print("Best params:", study.best_params)

    os.makedirs("optimized_params", exist_ok=True)

    with open("optimized_params/mlp_best.json", "w") as f:
        json.dump(study.best_params, f, indent=4)


if __name__ == "__main__":
    main()