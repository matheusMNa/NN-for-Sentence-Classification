import optuna
import joblib
import os
import json

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

from src.dataset import load_pubmed_rct

def main():
    print("Carregando os dados de teste... ")

    train_texts, train_labels, val_texts, val_labels, _,_ = load_pubmed_rct()

    print("\n Carregando o TF-IDF. ")

    vectorizer = joblib.load("models/best_tfidf_vectorizer.joblib")

    X_train = vectorizer.transform(train_texts)
    X_val = vectorizer.transform(train_labels)

    def objective(trial):

        C = trial.suggest_float("C", 1e-3, 1e2, log=True)

        class_weight = trial.suggest_categorical(
            "class_weight",
            [None, "balanced"]
        )

        model = LinearSVC(
            C=C,
            class_weight=class_weight
        )

        model.fit(X_train, train_labels)

        preds = model.predict(X_val)

        f1 = f1_score(val_labels, preds, average="macro")

        print(f'Trial {trial.number} -> F1: {f1:.4f}')

        return f1
    
    study = optuna.create_study(
        direction="maximize",
        study_name="linear_svc_tfidf",
        storage="sqlite:///optimized_params/db_files/linear_svc.db",
        load_if_exists=True
    )

    n_trials = int(os.getenv("N_TRIALS", 50))

    study.optimize(objective, n_trials=n_trials)

    print("\n Otimização completa.")
    print("Melhor F1:", study.best_value)
    print("Melhores params:", study.best_params)

    #salvando os melhores parâmetros
    os.makedirs("optimized_params", exist_ok=True)
    with open("optimized_params/linear_svc.json","w") as f:
        json.dump(study.best_params, f, indent=4)

if __name__ == "__main__":
    main()