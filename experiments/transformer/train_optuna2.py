import argparse
import optuna
import optuna.exceptions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Importações dos módulos locais 
from dataset import load_pubmed_rct, construir_vocabulario, pre_processar_textos, PubMedDataset
from model import TransformerClassifier

def avaliar_com_f1_macro(modelo, dataloader, criterion, device, PAD_IDX):
    """
    Avalia o modelo calculando a Loss e o F1-Score (Macro) para lidar com classes desbalanceadas.
    """
    modelo.eval()
    loss_acumulada = 0.0
    
    todas_previsoes = []
    todos_alvos = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = (inputs != PAD_IDX).unsqueeze(1).unsqueeze(2)
            
            outputs = modelo(inputs, mask)
            loss = criterion(outputs, targets)
            loss_acumulada += loss.item()
            
            _, previstos = outputs.max(dim=1)
            
            # Guardamos os arrays na CPU para o scikit-learn
            todas_previsoes.extend(previstos.cpu().numpy())
            todos_alvos.extend(targets.cpu().numpy())
            
    loss_media = loss_acumulada / len(dataloader)
    
    # Cálculo do F1-Macro
    f1_macro = f1_score(todos_alvos, todas_previsoes, average='macro')
    
    return loss_media, f1_macro

def main():
    # 1. Configurar argparse para receber parâmetros no ambiente HPC
    parser = argparse.ArgumentParser(description="Otimização de Hiperparâmetros - Transformer PubMed (F1-Macro)")
    parser.add_argument("--trials", type=int, default=100, help="Número de trials do Optuna")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas por trial")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A executar no dispositivo: {device}")

    # 2. Carregamento e Pré-processamento dos Dados
    X_train, y_train, X_val, y_val, X_test, y_test = load_pubmed_rct()

    vocabulario = construir_vocabulario(X_train)
    tamanho_vocab = len(vocabulario)
    PAD_IDX = vocabulario["<PAD>"]

    print("\nA processar conjunto de Treino...")
    X_train_tensor = pre_processar_textos(X_train, vocabulario, max_tamanho_seq=128)

    print("\nA processar conjunto de Validação...")
    X_val_tensor = pre_processar_textos(X_val, vocabulario, max_tamanho_seq=128)

    train_dataset = PubMedDataset(X_train_tensor, y_train)
    val_dataset = PubMedDataset(X_val_tensor, y_val)

    # 3. Definição da Função Objetivo para o Optuna
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        num_camadas = trial.suggest_int("n_layers", 1, 3)
        d_ff = trial.suggest_int("hidden_dims", 64, 1024)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        d_modelo = 64
        num_heads = 4
        num_classes = 4
        max_tamanho_seq = 128

        modelo_opt = TransformerClassifier(
            tamanho_vocab=tamanho_vocab,
            d_modelo=d_modelo,
            num_heads=num_heads,
            d_ff=d_ff,
            num_camadas=num_camadas,
            dropout=dropout,
            num_classes=num_classes,
            max_tamanho_seq=max_tamanho_seq
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        if optimizer_name == "Adam":
            optimizer = optim.Adam(modelo_opt.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(modelo_opt.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.RMSprop(modelo_opt.parameters(), lr=lr, weight_decay=weight_decay)

        # Loop de Treino e Validação
        for epoch in range(args.epochs):
            modelo_opt.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                mask = (inputs != PAD_IDX).unsqueeze(1).unsqueeze(2)
                
                optimizer.zero_grad()
                outputs = modelo_opt(inputs, mask)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping para estabilidade do Transformer
                torch.nn.utils.clip_grad_norm_(modelo_opt.parameters(), max_norm=1.0)
                
                optimizer.step()
                
            # Validação utilizando a métrica F1-Macro
            loss_val, f1_val = avaliar_com_f1_macro(
                modelo_opt, val_loader, criterion, device, PAD_IDX
            )
            
            # Informa o F1-Macro para o Pruner do Optuna
            trial.report(f1_val, epoch)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        # O Optuna avaliará este trial com base no F1-Macro final
        return f1_val

    # 4. Configuração e Execução do Estudo 
    study_name = "transformer_pubmed_f1_macro"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize", # F1-Macro deve ser maximizado
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True 
    )
    
    print(f"\nA iniciar otimização com {args.trials} trials e {args.epochs} épocas por trial...")
    study.optimize(objective, n_trials=args.trials)

    # 5. Guardar resultados
    print("\n--- Resultados da Otimização ---")
    print(f"Melhor trial (F1-Macro de Validação): {study.best_trial.value:.4f}")
    
    with open("melhores_parametros_f1.txt", "w") as f:
        f.write(f"Melhor F1-Macro de Validação: {study.best_trial.value:.4f}\n")
        f.write("Melhores Hiperparâmetros:\n")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
            f.write(f"  {key}: {value}\n")

if __name__ == "__main__":
    main()