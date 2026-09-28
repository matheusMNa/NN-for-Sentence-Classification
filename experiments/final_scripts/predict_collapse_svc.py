import json
import joblib
import numpy as np

from sklearn.metrics import classification_report, f1_score

from src.dataset import load_pubmed_rct

CLASS_MERGE = {"objective": "background"}


def main():
    print("Carregando dataset, vetorizador e modelo SVC...")
    _, _, _, _, test_texts, test_labels = load_pubmed_rct()
    vectorizer = joblib.load("models/best_tfidf_vectorizer.joblib")
    model = joblib.load("models/final_linear_svc_model.joblib")

    X_test = vectorizer.transform(test_texts)
    preds = model.predict(X_test)

    true_labels = np.array(test_labels)
    preds_collapsed = np.array([CLASS_MERGE.get(l, l) for l in preds])
    true_collapsed = np.array([CLASS_MERGE.get(l, l) for l in true_labels])
    collapsed_classes = sorted(set(true_collapsed))

    f1 = f1_score(true_collapsed, preds_collapsed, average="macro")
    print(f"\nF1 Macro (colapsado): {f1:.4f}")
    print(classification_report(true_collapsed, preds_collapsed, labels=collapsed_classes))

    results = {
        "model": "linear_svc",
        "classes_avaliacao": collapsed_classes,
        "class_merge": CLASS_MERGE,
        "f1_macro": float(f1),
        "accuracy": float((preds_collapsed == true_collapsed).mean()),
        "classification_report": classification_report(
            true_collapsed, preds_collapsed, labels=collapsed_classes, output_dict=True
        ),
    }

    with open("results/linear_svc_test_metrics_collapsed.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
