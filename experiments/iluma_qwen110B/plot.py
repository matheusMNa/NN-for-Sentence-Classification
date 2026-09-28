import json
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Modificado para versão de terminal
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from openai import OpenAI
from dataset import load_pubmed_rct

df_resultados = pd.read_csv('checkpoint_qwen_val.csv')

df_limpo = df_resultados[df_resultados['Classe_Prevista'] != 'ERRO'].copy()
y_true = df_limpo['Classe_Real']
y_pred = df_limpo['Classe_Prevista']

# ==========================================
# 4. AVALIAÇÃO E EXPORTAÇÃO DOS GRÁFICOS
# ==========================================
print("\nGerando métricas e gráficos...")
acc = accuracy_score(y_true, y_pred)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

report = classification_report(y_true, y_pred, output_dict=True)
classes = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'RESULTS']
f1_por_classe = [report.get(c, {}).get('f1-score', 0) for c in classes]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Desempenho do Modelo - Qwen 122B (4 Classes)", fontsize=16, fontweight='bold', y=1.05)

bar_width = 0.35
opacidade = 0.9
cor_1 = '#6a5acd' 
cor_2 = '#b0c4de' 

def rotular_barras(ax, barras):
    for barra in barras:
        altura = barra.get_height()
        ax.annotate(f'{altura:.3f}',
                    xy=(barra.get_x() + barra.get_width() / 2, altura),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

# Gráfico 1: F1 Macro e Acurácia
ax1 = axes[0]
x1 = np.arange(1)
bar1 = ax1.bar(x1 - bar_width/2, [f1_macro], bar_width, label='F1 Macro', color=cor_1, alpha=opacidade)
bar2 = ax1.bar(x1 + bar_width/2, [acc], bar_width, label='Acurácia', color=cor_2, alpha=opacidade)
ax1.set_title('F1 Macro e Acurácia')
ax1.set_xticks(x1)
ax1.set_xticklabels(['Qwen 122B'])
ax1.set_ylim(0, 1.1)
ax1.legend()
rotular_barras(ax1, bar1)
rotular_barras(ax1, bar2)

# Gráfico 2: F1 por Classe
ax2 = axes[1]
x2 = np.arange(len(classes))
bar_classes = ax2.bar(x2, f1_por_classe, bar_width*1.5, color=cor_1, alpha=opacidade)
ax2.set_title('F1 por Classe')
ax2.set_xticks(x2)
ax2.set_xticklabels([c.lower() for c in classes])
ax2.set_ylim(0, 1.1)
rotular_barras(ax2, bar_classes)

# Gráfico 3: Precision e Recall
ax3 = axes[2]
x3 = np.arange(1)
bar3 = ax3.bar(x3 - bar_width/2, [precision_macro], bar_width, label='Precision', color=cor_1, alpha=opacidade)
bar4 = ax3.bar(x3 + bar_width/2, [recall_macro], bar_width, label='Recall', color=cor_2, alpha=opacidade)
ax3.set_title('Precision e Recall (macro avg)')
ax3.set_xticks(x3)
ax3.set_xticklabels(['Qwen 122B'])
ax3.set_ylim(0, 1.1)
ax3.legend()
rotular_barras(ax3, bar3)
rotular_barras(ax3, bar4)

plt.tight_layout()

# Salva o gráfico em disco ao invés de tentar exibir na tela
plt.savefig('desempenho_qwen_122b.png', dpi=300, bbox_inches='tight')
print("Gráficos salvos em 'desempenho_qwen_122b.png'")

print("\n--- Relatório de Classificação Final ---")
print(classification_report(y_true, y_pred, target_names=classes))