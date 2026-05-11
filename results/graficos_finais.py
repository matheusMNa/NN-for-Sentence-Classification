import json
import matplotlib.pyplot as plt
import numpy as np

# =========================
# CARREGAR JSONs
# =========================

with open("results/logreg_test_metrics.json", "r") as f:
    logreg = json.load(f)

with open("results/linear_svc_test_metrics.json", "r") as f:
    svc = json.load(f)

with open("results/mlp_test_metrics.json", "r") as f:
    mlp = json.load(f)

print(list(logreg["classification_report"].keys()))
print(list(svc["classification_report"].keys()))
print(list(mlp["classification_report"].keys()))

models = {
    "Log. Reg.": logreg,
    "LinearSVC": svc,
    "MLP": mlp
}

colors = ["#534AB7", "#0F6E56", "#993C1D"]

class_names = ['background', 'conclusions', 'methods', 'objective', 'results']

def normalize_report(report):
    if '0' in report:
        return {class_names[i]: report[str(i)] for i in range(5)} | {
            k: report[k] for k in ("accuracy", "macro avg", "weighted avg")
        }
    return report

mlp["classification_report"] = normalize_report(mlp["classification_report"])

# =========================
# GRÁFICO 1 — F1 Macro e Acurácia
# =========================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Comparação de Modelos — PubMed RCT", fontsize=14, fontweight="bold")

names = list(models.keys())
f1_scores = [m["f1_macro"] for m in models.values()]
accuracies = [m["accuracy"] for m in models.values()]

x = np.arange(len(names))
width = 0.35

ax = axes[0]
bars1 = ax.bar(x - width/2, f1_scores, width, label="F1 Macro", color=colors, alpha=0.9)
bars2 = ax.bar(x + width/2, accuracies, width, label="Acurácia", color=colors, alpha=0.5)
ax.set_title("F1 Macro e Acurácia")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1)
ax.legend()
ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)

# =========================
# GRÁFICO 2 — F1 por classe
# =========================

classes = [k for k in logreg["classification_report"].keys() 
           if k not in ("accuracy", "macro avg", "weighted avg")]

x = np.arange(len(classes))
width = 0.25

ax = axes[1]
for i, (name, model) in enumerate(models.items()):
    f1_per_class = [model["classification_report"][c]["f1-score"] for c in classes]
    ax.bar(x + i * width, f1_per_class, width, label=name, color=colors[i], alpha=0.9)

ax.set_title("F1 por Classe")
ax.set_xticks(x + width)
ax.set_xticklabels([f"{c}" for c in classes])
ax.set_ylim(0, 1)
ax.legend()

# =========================
# GRÁFICO 3 — Precision e Recall (macro avg)
# =========================

precisions = [m["classification_report"]["macro avg"]["precision"] for m in models.values()]
recalls = [m["classification_report"]["macro avg"]["recall"] for m in models.values()]

x = np.arange(len(names))

ax = axes[2]
bars1 = ax.bar(x - width/2, precisions, width, label="Precision", color=colors, alpha=0.9)
bars2 = ax.bar(x + width/2, recalls, width, label="Recall", color=colors, alpha=0.5)
ax.set_title("Precision e Recall (macro avg)")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1)
ax.legend()
ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)

plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()