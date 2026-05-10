from src.dataset import (
    load_pubmed_rct,
    build_tfidf_vectorizer,
    vectorize_data
)

def main():

    X_train,y_train,X_val,y_val,X_test,y_test= load_pubmed_rct()

    print("Dataset carregado corretamente")

    vectorizer = build_tfidf_vectorizer()

    (
        X_train_tfidf,
        X_val_tfidf,
        X_test_tfidf
    ) = vectorize_data(
        vectorizer,
        X_train,
        X_val,
        X_test
    )

    print('Vetorização TF-IDF completa. ')

    print(f"Train shape: {X_train_tfidf.shape}")
    print(f"Validation shape: {X_val_tfidf.shape}")
    print(f"Test shape: {X_test_tfidf.shape}")
