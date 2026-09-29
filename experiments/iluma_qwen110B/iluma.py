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

# ==========================================
# 0. CONFIGURAÇÃO DA API 
# ==========================================
BASE_URL = "https://iluma.cnpem.br:4000/v1"
MODELO = "iluma"

# Resgata o token definido no job.sh (evita travar o HPC pedindo input)
token = os.environ.get("ILUMA_TOKEN")
if not token:
    raise ValueError("Token da IlumA não encontrado. Defina a variável ILUMA_TOKEN no job.sh.")

cliente = OpenAI(base_url=BASE_URL, api_key=token, timeout=600.0, max_retries=1)

# ==========================================
# 1. CARREGAMENTO E MAPEAMENTO DE DADOS
# ==========================================
print("Carregando dataset oficial...")
_, _, X_val, y_val, _, _ = load_pubmed_rct()

mapa_classes = {
    'background': 'BACKGROUND',
    'conclusions': 'CONCLUSIONS',
    'methods': 'METHODS',
    'objective': 'BACKGROUND', 
    'results': 'RESULTS'
}

df_val = pd.DataFrame({
    'Sentenca': X_val,
    'Classe_Real': [mapa_classes[label] for label in y_val]
})

# Seleciona aleatoriamente 3.000 frases com uma seed fixa para ser reprodutível
df_val = df_val.sample(n=3000, random_state=42).reset_index(drop=True)

print(f"Total de frases reduzido para validação: {len(df_val)}")

# ==========================================
# 2. CONFIGURAÇÃO DO MODELO QWEN
# ==========================================
SISTEMA_CLASSIFICACAO = """Você é um especialista em Processamento de Linguagem Natural e literatura médica. Sua tarefa é realizar a classificação sequencial de sentenças de resumos (abstracts) de ensaios clínicos aleatorizados (RCTs).

Para cada sentença fornecida, você deve classificá-la em exatamente UMA das quatro categorias abaixo. 

DEFINIÇÕES DAS CLASSES:
1. BACKGROUND: Estabelece o contexto, o conhecimento prévio, a lacuna científica que motivou o estudo e DEVE INCLUIR também o propósito, a meta ou a hipótese específica do estudo (ex: "To evaluate...", "We aimed to...").
2. METHODS: Descreve o desenho do ensaio clínico, características iniciais dos pacientes (baseline), critérios de inclusão, intervenções, dosagens e métodos estatísticos.
3. RESULTS: Apresenta as descobertas empíricas do estudo atual, dados quantitativos, valores de p, intervalos de confiança, eficácia e eventos adversos observados.
4. CONCLUSIONS: Oferece a interpretação dos resultados, implicações clínicas, recomendações, limitações ou a mensagem final dos autores.

REGRAS DE CLASSIFICAÇÃO E SAÍDA:
- Analise a semântica da sentença isoladamente, prestando atenção em falsos cognatos estruturais (ex: dados de baseline são METHODS, não RESULTS).
- Responda EXCLUSIVAMENTE com um JSON válido, sem formatação Markdown.
- O formato exato da resposta deve ser: {"classe": "NOME_DA_CLASSE"}
- NOME_DA_CLASSE deve ser estritamente uma destas opções: BACKGROUND, METHODS, RESULTS, CONCLUSIONS."""

EXEMPLOS_FEW_SHOT = [
    # ==========================================
    # CLASSE 1: BACKGROUND (Contexto + Objetivos)
    # ==========================================
    # 1. Contexto clássico com citação de resultados passados (ensina a não confundir com RESULTS)
    {"role": "user", "content": "Previous observational studies have suggested a 20% reduction in cardiovascular mortality associated with regular green tea consumption."},
    {"role": "assistant", "content": '{"classe": "BACKGROUND"}'},
    
    # 2. Objetivo puro (ensina a fusão da antiga classe OBJECTIVE no BACKGROUND)
    {"role": "user", "content": "To assess the long-term safety and efficacy of monoclonal antibody therapy in pediatric patients with refractory asthma."},
    {"role": "assistant", "content": '{"classe": "BACKGROUND"}'},
    
    # 3. Mistura de lacuna científica com hipótese
    {"role": "user", "content": "Despite the high prevalence of gestational diabetes, consensus on the optimal glycemic target remains controversial, prompting us to test a stricter monitoring regimen."},
    {"role": "assistant", "content": '{"classe": "BACKGROUND"}'},
    
    # 4. Desenho do estudo declarado como intenção
    {"role": "user", "content": "This randomized, double-blind trial was designed to determine whether prophylactic antibiotics reduce postoperative infection rates in elective orthopedic surgery."},
    {"role": "assistant", "content": '{"classe": "BACKGROUND"}'},
    
    # 5. Declaração direta de hipótese
    {"role": "user", "content": "We hypothesized that early administration of intravenous fluids would improve overall survival in septic shock patients."},
    {"role": "assistant", "content": '{"classe": "BACKGROUND"}'},

    # ==========================================
    # CLASSE 2: METHODS (Desenho, Intervenções e Baseline)
    # ==========================================
    # 1. Características da linha de base com números (ensina a não confundir com RESULTS)
    {"role": "user", "content": "At baseline, the mean age of the participants was 54.2 years, and 45% had a prior history of hypertension."},
    {"role": "assistant", "content": '{"classe": "METHODS"}'},
    
    # 2. Definição do desfecho primário (outcome definition)
    {"role": "user", "content": "The primary efficacy endpoint was the absolute change in the unified Parkinson's disease rating scale motor score from baseline to week 24."},
    {"role": "assistant", "content": '{"classe": "METHODS"}'},
    
    # 3. Metodologia estatística
    {"role": "user", "content": "Continuous variables were compared using the Student's t-test, while categorical data were analyzed using the Chi-square test."},
    {"role": "assistant", "content": '{"classe": "METHODS"}'},
    
    # 4. Desenho de randomização e recrutamento
    {"role": "user", "content": "Patients aged 18 to 65 years with a confirmed diagnosis of major depressive disorder were randomly assigned in a 2:1 ratio to receive either the novel antidepressant or a placebo."},
    {"role": "assistant", "content": '{"classe": "METHODS"}'},
    
    # 5. Detalhamento da intervenção/dosagem
    {"role": "user", "content": "Participants in the intervention arm attended twice-weekly cognitive behavioral therapy sessions lasting 45 minutes each for a duration of three months."},
    {"role": "assistant", "content": '{"classe": "METHODS"}'},

    # ==========================================
    # CLASSE 3: RESULTS (Achados, Estatísticas e Segurança)
    # ==========================================
    # 1. Resultado principal com blocos estatísticos pesados
    {"role": "user", "content": "At 12 months, the incidence of the primary composite endpoint was significantly lower in the treatment group than in the control group (12.4% vs. 18.9%; HR, 0.65; 95% CI, 0.52 to 0.81; P<0.001)."},
    {"role": "assistant", "content": '{"classe": "RESULTS"}'},
    
    # 2. Sentença que inicia com método estatístico mas entrega o resultado
    {"role": "user", "content": "Intention-to-treat analysis revealed that the mean reduction in fasting blood glucose was 18 mg/dL greater in the experimental arm."},
    {"role": "assistant", "content": '{"classe": "RESULTS"}'},
    
    # 3. Resultado nulo (diferença não significativa)
    {"role": "user", "content": "There was no statistically significant difference in the rate of hospital readmissions between the early discharge and standard care cohorts (p = 0.34)."},
    {"role": "assistant", "content": '{"classe": "RESULTS"}'},
    
    # 4. Dados de segurança e eventos adversos (ensina a separar de números de recrutamento de METHODS)
    {"role": "user", "content": "Severe adverse events, including gastrointestinal bleeding and acute renal failure, occurred in 5 patients (2%) receiving the active drug and 3 patients (1.2%) receiving placebo."},
    {"role": "assistant", "content": '{"classe": "RESULTS"}'},
    
    # 5. Resultados de subgrupos
    {"role": "user", "content": "In the predefined subgroup of patients over 65 years of age, the mortality benefit was even more pronounced, with a relative risk reduction of 30%."},
    {"role": "assistant", "content": '{"classe": "RESULTS"}'},

    # ==========================================
    # CLASSE 4: CONCLUSIONS (Interpretação e Implicações)
    # ==========================================
    # 1. Recapitulação numérica antes da interpretação (ensina a não classificar como RESULTS)
    {"role": "user", "content": "Although the 10% increase in progression-free survival was statistically significant, its clinical relevance remains uncertain given the high rate of toxicity."},
    {"role": "assistant", "content": '{"classe": "CONCLUSIONS"}'},
    
    # 2. Conclusão direta e implicação clínica
    {"role": "user", "content": "These findings demonstrate that implementing a structured discharge protocol significantly reduces 30-day readmission rates in heart failure patients."},
    {"role": "assistant", "content": '{"classe": "CONCLUSIONS"}'},
    
    # 3. Limitações do estudo
    {"role": "user", "content": "The short follow-up period and the single-center nature of this trial limit the generalizability of our findings."},
    {"role": "assistant", "content": '{"classe": "CONCLUSIONS"}'},
    
    # Veredicto clínico final de eficácia e segurança sem estatísticas em bruto
    {"role": "user", "content": "A single dose of oritavancin was noninferior to 10 days of vancomycin in treating ABSSSIs caused by gram-positive pathogens, and was well tolerated."},
    {"role": "assistant", "content": '{"classe": "CONCLUSIONS"}'},
    
    # Resumo de impacto usando verbo de ação ("produced") sem valores-p
    {"role": "user", "content": "Physician referral to an Internet-based behavioral weight-loss intervention produced clinically significant weight loss for over half of the patients studied."},
    {"role": "assistant", "content": '{"classe": "CONCLUSIONS"}'}
]

def classificar_pubmed(sentenca: str) -> str:
    mensagens = [{"role": "system", "content": SISTEMA_CLASSIFICACAO}]
    mensagens.extend(EXEMPLOS_FEW_SHOT)
    mensagens.append({"role": "user", "content": sentenca})
    
    try:
        resposta = cliente.chat.completions.create(
            model=MODELO,
            messages=mensagens,
            temperature=0.5,
            top_p=0.1
        )
        bruto = resposta.choices[0].message.content.strip()
        if bruto.startswith("```"):
            bruto = bruto.strip("`").replace("json", "", 1).strip()
        return json.loads(bruto).get("classe", "ERRO")
    except Exception as e:
        return "ERRO"

# ==========================================
# 3. PIPELINE DE INFERÊNCIA COM CHECKPOINT
# ==========================================
arquivo_checkpoint = "checkpoint_qwen_val_2.csv"

if os.path.exists(arquivo_checkpoint):
    df_resultados = pd.read_csv(arquivo_checkpoint)
    processados = len(df_resultados)
    print(f"Retomando do checkpoint: {processados} frases já processadas.")
else:
    df_resultados = pd.DataFrame(columns=['Sentenca', 'Classe_Real', 'Classe_Prevista'])
    processados = 0

for i in tqdm(range(processados, len(df_val)), desc="Inferindo no Qwen"):
    sentenca = df_val.iloc[i]['Sentenca']
    classe_real = df_val.iloc[i]['Classe_Real']
    
    classe_prev = classificar_pubmed(sentenca)
    
    novo_dado = pd.DataFrame([{
        'Sentenca': sentenca, 
        'Classe_Real': classe_real, 
        'Classe_Prevista': classe_prev
    }])
    df_resultados = pd.concat([df_resultados, novo_dado], ignore_index=True)
    
    # Salva checkpoint a cada 50 iterações
    if i % 50 == 0:
        df_resultados.to_csv(arquivo_checkpoint, index=False)

df_resultados.to_csv("resultados_finais_qwen.csv", index=False)

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