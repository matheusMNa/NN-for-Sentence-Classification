import optuna
import numpy as np
import time
import os
import joblib

from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.dataset import load_pubmed_rct

n_trials = int(os.getenv("N_TRIALS", 100))

train_texts, train_labels, val_texts, val_labels,_,_ = load_pubmed_rct()

def objective(trial):

    start_time = time.time()
    print(f'\n Começado o trial {trial.number}')

    max_features = trial.suggest_int(
    'max_features',
    5000,
    30000
    )

    ngram_upper = trial.suggest_int(
    'ngram_upper',
    1,
    2
    )

    min_df = trial.suggest_int(
    'min_df',
    2,
    10
    )
       
    C = trial.suggest_float(
        'C',
        1e-3,
        1e2,
        log=True
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1,ngram_upper),
        min_df=min_df,
        dtype=np.float32
    )
    
    # print("\n Fitando TF-IDF... ")

    X_train_tfidf = vectorizer.fit_transform(train_texts)

    # print("\n Transformando dados de validação")

    X_val_tfidf = vectorizer.transform(val_texts)

    # Construindo regressor

    model = LogisticRegression(
        C=C,
        solver='saga',
        max_iter=1000,
        random_state=404,
        )

    # print('\n Treinando modelo de Regressão Logística')

    model.fit(X_train_tfidf, train_labels)

    #validação

    # print(" \n Previsões:")

    val_predictions = model.predict(X_val_tfidf)

    #computando métrica (f1 Macro)

    macro_f1 = f1_score(
        val_labels,
        val_predictions,
        average='macro'
    )

    elapsed = time.time() - start_time

    print(
    f"\nTrial {trial.number} completo\n"
    f"F1 Macro: {macro_f1:.4f}\n"
    f"Tempo = {elapsed:.2f} segundos\n"
)

    return macro_f1

#Criando estudos

study = optuna.create_study(
    direction="maximize",
    study_name="tfidf_logreg",
    storage="sqlite:///tfidf_logreg.db",
    load_if_exists=True
)

#rodando otimização

study.optimize(
    objective,
    n_trials=n_trials,
    timeout=3600,
    catch=(Exception,)
)

#Melhores resultados

best_params = study.best_params

best_vectorizer = TfidfVectorizer(
    max_features=best_params["max_features"],
    ngram_range=(1, best_params["ngram_upper"]),
    min_df=best_params["min_df"]
 )

best_vectorizer.fit(train_texts)

joblib.dump(
       value= best_vectorizer,
       filename= "models/best_tfidf_vectorizer.joblib"
)

print("\n Vectorizer salvo com sucesso")

print("\n Otimização completa.")
print(f"Melhor F1 Macro: {study.best_value:.4f}")

print("\n Melhores hiperparâmetros:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

import json
with open("best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
