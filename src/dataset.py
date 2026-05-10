from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

#carregando o dataset e fazendo data split de acordo com os splits oficiais do dataset
def load_pubmed_rct():
    dataset = load_dataset("armanc/pubmed-rct20k")

    print("Dentro da função load_pubmed_rct")

    # Inside load_pubmed_rct in src/dataset.py
    #print(f"Dataset columns: {dataset['train'].column_names}")

    #corrigindo as labels
    X_train = dataset['train']['text']
    y_train = dataset['train']['label']

    X_val = dataset['validation']['text']
    y_val = dataset['validation']['label']

    X_test = dataset['test']['text']
    y_test = dataset['test']['label']

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )

#cosntruindo vetorizador
def build_tfidf_vectorizer():
    vectorizer = TfidfVectorizer(
        lowercase=True, 
        stop_words=None,

        #forçamos o modelo a reter termos mais relevantes
        #diminui dimensionalidade
        max_features=20000,

        #incluo bigramas -> número de features maior
        ngram_range=(1, 2),

        #remove palavras que aparecerem só uma vez no corpus de treino
        min_df=2
    )

    return vectorizer

#vetorização

def vectorize_data(
        vectorizer,
        X_train,
        X_val,
        X_test
):
    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_val_tfidf = vectorizer.transform(X_val)

    X_test_tfidf = vectorizer.transform(X_test)

    return (
        X_train_tfidf,
        X_val_tfidf,
        X_test_tfidf
    )
