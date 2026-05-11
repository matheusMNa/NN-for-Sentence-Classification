import json
import joblib

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

from src.dataset import load_pubmed_rct

def main():

    print("Carregando dataset...")

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_pubmed_rct()

    # carregando TF-IDF congelado a ser reutilizado por todos os modelos

    print("\nCarregando o TF-IDF pré-treinado: ")

    vectorizer = joblib.load("models/best_tfidf_vectorizer.joblib")

    # transformar os dados usando o mesmo espaço de features

    X_train = vectorizer.transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    with open("optimized_params/linear_svc.json", "r") as f:
        best_params = json.load(f)

    print("\nBest parameters loaded: ")
    print(best_params)

    # treinando a SVC final

    model = LinearSVC(
        C=best_params["C"],
        class_weight=best_params["class_weight"],
        max_iter=2000,
    )

    model.fit(X_train, train_labels)

    # check de sanidade

    val_preds = model.predict(X_val)
    val_f1 = f1_score(val_labels, val_preds, average="macro")

    print("\nValidation F1 Macro:", round(val_f1, 4))

    # avaliação final

    print("\nRealizando a avaliação no conjunto de teste: ")

    test_preds = model.predict(X_test)

    test_f1 = f1_score(test_labels, test_preds, average="macro")

    print("\nResultados - Teste Final: ")

    print("\nF1 Macro:", round(test_f1, 4))

    print("\nReport de Classificação: ")

    print(classification_report(test_labels, test_preds))

    # salvando o modelo treinado
    joblib.dump(model, "models/final_linear_svc_model.joblib")

    results = {
        "model": "linear_svc",
        "f1_macro": float(test_f1),
        "accuracy": float((test_preds == test_labels).mean()),
        "classification_report": classification_report(test_labels, test_preds, output_dict=True)
    }

    with open("results/linear_svc_test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()