import torch
from torch.utils.data import Dataset
from collections import Counter
import spacy
from datasets import load_dataset

# carregando o tokenizador em inglês
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])

def load_pubmed_rct():
    """
    Carrega o dataset e faz o data split de acordo com os splits oficiais.
    """
    dataset = load_dataset("armanc/pubmed-rct20k")

    print("Dentro da função load_pubmed_rct")

    # corrigindo as labels
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

def construir_vocabulario(textos, frequencia_minima=2):
    print("Iniciando a contagem e construção do vocabulário...")
    contador = Counter()
    
    for doc in nlp.pipe(textos, batch_size=1000):
        tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        contador.update(tokens)
    
    # adição do token <CLS>
    vocabulario = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
    idx = 3
    
    for palavra, freq in contador.items():
        if freq >= frequencia_minima:
            vocabulario[palavra] = idx
            idx += 1
    return vocabulario

def pre_processar_textos(textos, vocabulario, max_tamanho_seq=128):
    resultados = []
    print("Pré-processando textos...")
    for doc in nlp.pipe(textos, batch_size=1000):
        tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        
        indices = [vocabulario["<CLS>"]] + [vocabulario.get(t, vocabulario["<UNK>"]) for t in tokens]
        
        if len(indices) < max_tamanho_seq:
            indices = indices + [vocabulario["<PAD>"]] * (max_tamanho_seq - len(indices))
        else:
            indices = indices[:max_tamanho_seq]
        resultados.append(indices)
        
    return torch.tensor(resultados, dtype=torch.long)

class PubMedDataset(Dataset):
    def __init__(self, tensores_texto, labels):
        self.textos = tensores_texto
        self.label_map = {'background': 0, 'conclusions': 1, 'methods': 2, 'objective': 3, 'results': 4}
        
        # Converte labels para números
        labels_num = [self.label_map[l] if isinstance(l, str) else l for l in labels]
        self.labels = torch.tensor(labels_num, dtype=torch.long)

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        return self.textos[idx], self.labels[idx]