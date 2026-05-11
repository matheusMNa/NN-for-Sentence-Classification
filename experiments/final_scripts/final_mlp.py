import json
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import load_pubmed_rct


# =========================
# DATASET
# =========================
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


# =========================
# MODEL
# =========================
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


# =========================
# TRAIN ONE EPOCH
# =========================
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
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
            preds = torch.argmax(model(X), dim=1).cpu().numpy()
            preds_total.extend(preds)
            y_total.extend(y.numpy())

    return np.array(preds_total), np.array(y_total)


# =========================
# MAIN
# =========================
def main():

    print("Carregando dataset...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_pubmed_rct()

    print("\nCarregando o TF-IDF pré-treinado...")
    vectorizer = joblib.load("models/best_tfidf_vectorizer.joblib")

    X_train = vectorizer.transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    # -------- label encoding --------
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_val = label_encoder.transform(val_labels)
    y_test = label_encoder.transform(test_labels)

    n_classes = len(label_encoder.classes_)
    input_dim = X_train.shape[1]

     # -------- device --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsando device: {device}")

    # -------- class weights --------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # -------- hiperparâmetros --------
    with open("optimized_params/mlp_best.json", "r") as f:
        best_params = json.load(f)

    print("\nBest parameters loaded:")
    print(best_params)

    n_layers = best_params["n_layers"]
    hidden_dims = [best_params[f"hidden_{i}"] for i in range(n_layers)]

    # -------- model --------
    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=best_params["dropout"],
        n_classes=n_classes
    ).to(device)

    # -------- optimizer --------
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

    # -------- loaders --------
    batch_size = best_params["batch_size"]
    train_loader = make_loader(X_train, y_train, batch_size)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size, shuffle=False)

    # -------- treino com early stopping --------
    best_val_f1 = 0.0
    patience = 10
    epochs_without_improvement = 0
    n_epochs = 50

    print("\nIniciando treino...")

    for epoch in range(n_epochs):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_preds, val_true = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_true, val_preds, average="macro")

        print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Val F1 Macro: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "models/final_mlp_best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping na epoch {epoch+1}.")
                break

    # -------- avaliação final --------
    print("\nCarregando melhor modelo para avaliação no teste...")
    model.load_state_dict(torch.load("models/final_mlp_best.pt"))

    test_preds, test_true = evaluate(model, test_loader, device)
    test_f1 = f1_score(test_true, test_preds, average="macro")

    print("\nResultados - Teste Final:")
    print("\nF1 Macro:", round(test_f1, 4))
    print("\nReport de Classificação:")
    print(classification_report(test_true, test_preds, target_names=label_encoder.classes_))

    # -------- salvando resultados --------
    results = {
        "model": "mlp",
        "f1_macro": float(test_f1),
        "accuracy": float((test_preds == test_true).mean()),
        "classification_report": classification_report(test_true, test_preds, output_dict=True)
    }

    with open("results/mlp_test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
