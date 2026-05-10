from sklearn.linear_model import LogisticRegression

def construir_reg_logistica():

    model = LogisticRegression(

        max_iter=1000,
        random_state=404,
        n_jobs=-1
    )

    return model 