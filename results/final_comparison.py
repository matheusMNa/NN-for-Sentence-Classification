import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Dados Globais dos Modelos
# Os dados foram retirados dos JSON da pasta results e das saídas do código
modelos = ['RegLog', 'SVC Linear', 'MLP', 'IlumA', 'Transformer', 'LSTM']
acuracia = [0.859, 0.856, 0.857, 0.834, 0.857, 0.886] 
f1_macro = [0.842, 0.839, 0.841, 0.808, 0.840, 0.873]

# 2. Dados por Classe
f1_background = [0.836, 0.839, 0.838, 0.873, 0.838, 0.878]
f1_methods = [0.900, 0.898, 0.900, 0.890, 0.898, 0.920]
f1_results = [0.878, 0.873, 0.875, 0.761, 0.878, 0.898]
f1_conclusions = [0.752, 0.745, 0.749, 0.646, 0.747, 0.796]

# 3. Configuração da Figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
sns.set_style("whitegrid")
bar_width = 0.35
x_indices = np.arange(len(modelos))

# Função de anotação
def autolabel(ax, rects, fontsize=10, rotation=0, formato='{:.3f}'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(formato.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', 
                    fontsize=fontsize, rotation=rotation, fontweight='bold')

# --- GRÁFICO 1: Acurácia e F1-Macro Global ---
bar1 = ax1.bar(x_indices - bar_width/2, f1_macro, bar_width, label='F1-Macro', color='#0072B2', alpha=0.9)
bar2 = ax1.bar(x_indices + bar_width/2, acuracia, bar_width, label='Acurácia', color='#E69F00', alpha=0.9)

ax1.set_title('(a)', loc='left', fontsize=26, fontweight='bold', pad=25)
ax1.set_title('Desempenho Global por Arquitetura', fontsize=24, fontweight='bold', pad=25)

ax1.set_xticks(x_indices)
ax1.set_xticklabels(modelos, fontsize=18)
ax1.tick_params(axis='y', labelsize=16) 

ax1.set_ylim(0, 1.15) 
ax1.legend(fontsize=15, loc='upper left')

autolabel(ax1, bar1, fontsize=14)
autolabel(ax1, bar2, fontsize=14)

# --- GRÁFICO 2: F1-Score das Classes Comuns ---
x_classes = np.arange(4) 
largura_agrupamento = 0.14  # Reduzido para caberem 6 barras
# 6ª cor adicionada da paleta Okabe-Ito (#CC79A7)
cores_okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7'] 

for i, modelo_nome in enumerate(modelos):
    valores_classe = [f1_background[i] ,f1_methods[i], f1_results[i], f1_conclusions[i]]
    # Cálculo atualizado de deslocamento para centralizar 6 barras (i - 2.5)
    posicoes = x_classes + (i - 2.5) * largura_agrupamento
    nome_legenda = modelo_nome.replace('\n', ' ') 
    barras_classe = ax2.bar(posicoes, valores_classe, largura_agrupamento, label=nome_legenda, color=cores_okabe_ito[i], alpha=0.9)
    # Fonte levemente reduzida para 13 para evitar sobreposição nos números rotacionados
    autolabel(ax2, barras_classe, fontsize=13, rotation=45, formato='{:.3f}')

ax2.set_title('(b)', loc='left', fontsize=26, fontweight='bold', pad=25)
ax2.set_title('Evolução do F1-Score nas Classes Comuns', fontsize=24, fontweight='bold', pad=25)

ax2.set_xticks(x_classes)
ax2.set_xticklabels(['BACKGROUND' ,'METHODS', 'RESULTS', 'CONCLUSIONS'], fontsize=18, fontweight='bold')
ax2.tick_params(axis='y', labelsize=16)

# Aumentado um pouco o limite Y para acomodar a legenda com 6 itens
ax2.set_ylim(0, 1.30) 
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), fontsize=15, ncol=6)

plt.tight_layout()
plt.savefig('comparacao.png', dpi=600, bbox_inches='tight')
plt.show()