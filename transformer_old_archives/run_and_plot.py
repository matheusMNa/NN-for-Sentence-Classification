import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import optuna
import copy

# Importações dos módulos do seu projeto
from dataset import load_pubmed_rct, construir_vocabulario, pre_processar_textos, PubMedDataset
from model import TransformerClassifier

# 1. Configuração do dispositivo e carregamento dos dados
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rodando no dispositivo: {device}")

X_train, y_train, X_val, y_val, X_test, y_test = load_pubmed_rct()
vocabulario = construir_vocabulario(X_train)
tamanho_vocab = len(vocabulario)
PAD_IDX = vocabulario["<PAD>"]

# Geração dos tensores e DataLoaders para treino, validação e teste
print("\nProcessando tensores...")
X_train_tensor = pre_processar_textos(X_train, vocabulario, max_tamanho_seq=128)
X_val_tensor = pre_processar_textos(X_val, vocabulario, max_tamanho_seq=128)
X_test_tensor = pre_processar_textos(X_test, vocabulario, max_tamanho_seq=128)

train_dataset = PubMedDataset(X_train_tensor, y_train)
val_dataset = PubMedDataset(X_val_tensor, y_val)
test_dataset = PubMedDataset(X_test_tensor, y_test)

batch_size = 32 # Pode fixar um batch size seguro para o treino final
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Recuperar os melhores hiperparâmetros do Optuna do banco de dados
study_name = "transformer_pubmed_study"
storage_name = f"sqlite:///{study_name}.db"
study = optuna.load_study(study_name=study_name, storage=storage_name)

best_params = study.best_trial.params
print("\n--- Melhores Hiperparâmetros do Optuna Aplicados ---")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# 3. Instanciar o modelo definitivo com os parâmetros ótimos
modelo = TransformerClassifier(
    tamanho_vocab=tamanho_vocab,
    d_modelo=64,
    num_heads=4,
    d_ff=best_params["hidden_dims"],
    num_camadas=best_params["n_layers"],
    dropout=best_params["dropout"],
    num_classes=5,
    max_tamanho_seq=128
).to(device)

criterion = nn.CrossEntropyLoss()

# Configurar o otimizador com base no otimizador escolhido pelo Optuna
opt_name = best_params.get("optimizer", "Adam")
lr = best_params["lr"]
weight_decay = best_params["weight_decay"]

if opt_name == "Adam":
    optimizer = optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
elif opt_name == "SGD":
    optimizer = optim.SGD(modelo.parameters(), lr=lr, weight_decay=weight_decay)
else:
    optimizer = optim.RMSprop(modelo.parameters(), lr=lr, weight_decay=weight_decay)

# 4. Função de avaliação auxiliar
def avaliar(modelo, dataloader):
    modelo.eval()
    loss_acumulada = 0.0
    corretos = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = (inputs != PAD_IDX).unsqueeze(1).unsqueeze(2)
            outputs = modelo(inputs, mask)
            loss = criterion(outputs, targets)
            loss_acumulada += loss.item()
            _, previstos = outputs.max(dim=1)
            total += targets.size(0)
            corretos += previstos.eq(targets).sum().item()
    return loss_acumulada / len(dataloader), 100. * corretos / total

# 5. Loop de Treinamento Real do Modelo Final
epochs = 100  # Número de épocas para consolidar o aprendizado
melhor_loss_val = float('inf')
melhor_pesos = copy.deepcopy(modelo.state_dict())

print(f"\nIniciando o treinamento do modelo final por {epochs} épocas...")
for epoch in range(epochs):
    modelo.train()
    loss_treino_acumulada = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        mask = (inputs != PAD_IDX).unsqueeze(1).unsqueeze(2)
        
        optimizer.zero_grad()
        outputs = modelo(inputs, mask)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping para evitar instabilidade numérica em transformers
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        
        optimizer.step()
        loss_treino_acumulada += loss.item()
        
    loss_val, acc_val = avaliar(modelo, val_loader)
    print(f"Época {epoch+1}/{epochs} | Loss Treino: {loss_treino_acumulada/len(train_loader):.4f} | Loss Val: {loss_val:.4f} | Acc Val: {acc_val:.2f}%")
    
    if loss_val < melhor_loss_val:
        melhor_loss_val = loss_val
        melhor_pesos = copy.deepcopy(modelo.state_dict())

# Carregar os melhores pesos obtidos durante o treinamento
modelo.load_state_dict(melhor_pesos)

# 6. Avaliação final detalhada no conjunto de Teste e Geração de Gráficos
def coletar_previsoes_teste(modelo, dataloader):
    modelo.eval()
    todas_previsoes = []
    todos_alvos = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = (inputs != PAD_IDX).unsqueeze(1).unsqueeze(2)
            outputs = modelo(inputs, mask)
            _, previstos = outputs.max(dim=1)
            todas_previsoes.extend(previstos.cpu().numpy())
            todos_alvos.extend(targets.cpu().numpy())
    return todos_alvos, todas_previsoes

classes = ['background', 'conclusions', 'methods', 'objective', 'results']
print("\nAvaliando o modelo treinado no conjunto de teste...")
alvos, previsoes = coletar_previsoes_teste(modelo, test_loader)

# Plotar e salvar resultados
cm = confusion_matrix(alvos, previsoes)
acuracia_por_classe = cm.diagonal() / cm.sum(axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x=classes, y=acuracia_por_classe * 100, ax=ax1, palette="viridis", hue=classes, legend=False)
ax1.set_title('Acurácia por Classe (%) - Modelo Treinado', fontsize=14)
ax1.set_ylabel('Acurácia (%)')
ax1.set_ylim(0, 100)

for i, v in enumerate(acuracia_por_classe * 100):
    ax1.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, 
            xticklabels=classes, yticklabels=classes)
ax2.set_title('Matriz de Confusão (Teste)', fontsize=14)
ax2.set_xlabel('Previsão do Modelo')
ax2.set_ylabel('Classe Real')

plt.tight_layout()
plt.savefig("resultado_modelo_treinado.png", dpi=300)
plt.show()

print("\nRelatório de Classificação Detalhado:")
print(classification_report(alvos, previsoes, target_names=classes))