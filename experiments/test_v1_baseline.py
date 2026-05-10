from src.dataset import (
    load_pubmed_rct,
    build_tfidf_vectorizer,
    vectorize_data
)

from experiments.baselines.log_reg import (
    construir_reg_logistica
)

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

def main():

    print("Carregando dataset...")

    X_train,y_train,X_val,y_val,X_test,y_test= load_pubmed_rct()

    print("Dataset carregado corretamente")

    vectorizer = build_tfidf_vectorizer()

    (   X_train_tfidf,
        X_val_tfidf,
        X_test_tfidf ) = vectorize_data(
        
        vectorizer,
        X_train,
        X_val,
        X_test
    )

    print('Vetorização TF-IDF completa. ')

    # print(f"Train shape: {X_train_tfidf.shape}")
    # print(f"Validation shape: {X_val_tfidf.shape}")
    # print(f"Test shape: {X_test_tfidf.shape}")

    model = construir_reg_logistica()

    print('Treinando a regressão logística')

    model.fit(X_train_tfidf, y_train)

    print('Treino completo.')

    print('\n Passo de avaliação: ')

    y_pred = model.predict(X_test_tfidf)

    acuracia = accuracy_score(y_test, y_pred)

    print(f'Acurácia: {acuracia:.4f}')

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":

    print('Script funcionando.')
    main()
