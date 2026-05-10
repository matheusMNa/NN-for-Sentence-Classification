import optuna

from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.dataset import load_pubmed_rct

train_texts, train_labels, val_texts, val_labels,_,_ = load_pubmed_rct()

def objective(trial):


    print(f'\n Começado um trial {trial.number}')

    max_features = trial.suggest_init(
    'max_features',
    5000,
    50000
    )

    n_gram_upper = trial.suggest_init(
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
        ngram_range(1,ngram_upper)

    )
  






    return macro_f1