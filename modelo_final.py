# Pasos

import pandas as pd
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import textstat
import language_tool_python
from nltk import word_tokenize
from collections import Counter
import joblib  
from sklearn.metrics import classification_report
import joblib  
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib 
import sys
import os

nltk.download('stopwords')
nltk.download('punkt_tab')
# Cargar modelos
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageToolPublicAPI('en-US')


def json_to_dataframe(json_file):
    """
    Convierte un archivo JSON en un DataFrame de pandas.
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    
    df = pd.DataFrame(data)
    return df


def calculate_lexical_diversity(df):
    """
    Agrega una nueva columna con la diversidad léxica de cada texto usando el índice TTR.
    """
    stop_words = set(stopwords.words('english'))  # Stopwords en inglés
    df['lexical_diversity'] = df['text'].apply(lambda text: calculate_ttr(text, stop_words))
    return df

def calculate_ttr(text, stop_words):
    """
    Calcula el índice de riqueza léxica (TTR) de un texto dado.
    Un valor más alto indicaría una mayor diversidad léxica (es decir, el texto utiliza un vocabulario más variado), 
    mientras que un valor más bajo indicaría una menor diversidad.
    No toma en cuenta la importancia de las palabras. Si un texto usa 100 palabras, pero muchas son palabras comunes, 
    eso no aumenta la diversidad léxica.
    """
    words = word_tokenize(text.lower())  # Tokenizamos y convertimos a minúsculas
    words = [word for word in words if word.isalpha()]  # Eliminar números y puntuaciones
    words = [word for word in words if word not in stop_words]  # Eliminar stopwords
    
    # Contamos el número total de palabras en el texto.
    total_words = len(words)
    unique_words = len(set(words))  # Número de palabras únicas
    
    # Para evitar división por cero, si el texto no tiene palabras, asignamos TTR como 0
    return unique_words / total_words if total_words > 0 else 0


def calculate_lexical_frequency(df):
    """
    Agrega una nueva columna con la frecuencia léxica media de cada texto usando TF-IDF.
    """
    # Instanciamos el vectorizador TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')  # Excluye palabras comunes (stop words)
    
    # Calculamos la matriz TF-IDF
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    
    # Calculamos la frecuencia léxica media para cada texto (media de los valores TF-IDF)
    # TF-IDF mide cuán relevante es una palabra dentro de un documento con respecto al resto del corpus.
    # Palabras muy comunes en el corpus (como "el", "y", "es", etc.) tienen un IDF bajo, y por lo tanto
    #  su valor TF-IDF será bajo.
    lexical_frequencies = tfidf_matrix.mean(axis=1).A1  # A1 convierte la matriz en un array 1D
    
    # Agregamos la nueva columna al DataFrame
    df['lexical_frequency'] = lexical_frequencies
    
    return df

import time


def extract_lingfeat(text):
    doc = nlp(text)  
    
    # Features básicas
    num_tokens = len(doc)
    num_sentences = len(list(doc.sents))
    avg_sentence_length = num_tokens / num_sentences if num_sentences else 0
    conteos = {
        "num_det": sum(1 for token in doc if token.pos_ == "DET"),
        "num_adj": sum(1 for token in doc if token.pos_ == "ADJ"),
        "num_nouns": sum(1 for token in doc if token.pos_ == "NOUN"),
        "num_adp": sum(1 for token in doc if token.pos_ == "ADP"),
        "num_cconj": sum(1 for token in doc if token.pos_ == "CCONJ"),
        "num_pron": sum(1 for token in doc if token.pos_ == "PRON"),
        "num_x": sum(1 for token in doc if token.pos_ == "X"),
        "num_aux": sum(1 for token in doc if token.pos_ == "AUX"),
        "num_verbs": sum(1 for token in doc if token.pos_ == "VERB"),
        "num_punct": sum(1 for token in doc if token.pos_ == "PUNCT"),
        "num_adv": sum(1 for token in doc if token.pos_ == "ADV"),
        "num_sconj": sum(1 for token in doc if token.pos_ == "SCONJ"),
        "num_part": sum(1 for token in doc if token.pos_ == "PART"),
        "num_propn": sum(1 for token in doc if token.pos_ == "PROPN"),
        "num_space": sum(1 for token in doc if token.pos_ == "SPACE"),
    }
    
    # Lexical diversity
    words = [token.text.lower() for token in doc if token.is_alpha]
    lexical_diversity = len(set(words)) / len(words) if words else 0
    flesch = textstat.flesch_reading_ease(text)
    total_words = 0
    total_length = 0
    total_words += len(words)
    total_length += sum(len(word) for word in words)

    if total_words > 0:
        avg_word_length = total_length / total_words
    
    return {
        "num_tokens": num_tokens,
        "num_sentences": num_sentences,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "conteos": conteos,
        "lexical_diversity": lexical_diversity,
        "flesch_reading_ease": flesch,
        # "grammar_errors": num_errors
    }

def f_05u(y_true, y_score):
    mod_pred = []
    for p in y_score:
        if p == 0.5:
            mod_pred.append(0)  # abstención tratada como falso negativo
        else:
            mod_pred.append(int(p > 0.5))
    prec = precision_score(y_true, mod_pred)
    rec = recall_score(y_true, mod_pred)
    beta = 0.5
    if prec + rec == 0:
        return 0
    return (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)


def c_at_1(y_true, y_score, threshold=0.5):
    import numpy as np
    y_pred = (y_score > threshold).astype(int)
    unanswered = (y_score == 0.5).astype(int)
    n = len(y_true)
    n_u = unanswered.sum()
    
    answered_mask = y_score != 0.5
    n_c = ((y_pred == y_true) & answered_mask).sum()

    acc = n_c / (n - n_u) if (n - n_u) > 0 else 0.0

    return (1.0 / n) * (n_c + n_u * acc)


def preparar_dataframe( json_file ):
    # 1. Cargamos los datos 

    df = json_to_dataframe(json_file)
    features_df = df['text'].apply(extract_lingfeat).apply(pd.Series)

    # Expandir la columna 'conteos' en columnas separadas
    pos_df = features_df['conteos'].apply(pd.Series)

    # Concatenar todo junto al DataFrame original
    df = pd.concat([df, features_df.drop(columns=['conteos']), pos_df], axis=1)

    # Normalizamos estos datos 
    columnas = [
        'num_det', 'num_adj', 'num_nouns', 'num_adp', 'num_cconj', 'num_pron',
        'num_x', 'num_aux', 'num_verbs', 'num_punct', 'num_adv', 'num_sconj',
        'num_part', 'num_propn', 'num_space']

    # 1. Contar número de palabras en la columna 'text'
    df['num_palabras'] = df['text'].apply(lambda x: len(str(x).split()))

    # 2. Crear nuevas columnas de proporción
    for col in columnas:
        nueva_columna = f'proporcion_{col}'
        df[nueva_columna] = df[col] / df['num_palabras']

    # Calculamos la diversidad y frecuencia lexica
    df = calculate_lexical_diversity(df)
    df = calculate_lexical_frequency(df)

    val_df = df.drop(['model', 'num_det', 'num_adj', 'num_nouns', 'num_adp',
       'num_cconj', 'num_pron', 'num_x', 'num_aux', 'num_verbs', 'num_punct',
       'num_adv', 'num_sconj', 'num_part', 'num_propn', 'num_space','proporcion_num_space'], axis=1)
    
    return val_df

def evaluate_model_predictions(model, path_val, output_dir):
    """
    Evaluates a model pipeline by predicting on X_val and comparing to y_val.
    
    Args:
        final_pipeline: Trained pipeline with `predict` and `predict_proba` methods.
        X_val: Validation features.
        y_val: True labels (ground truth).
        index_to_inspect: Specific index to show class probability for (default: 10).

    Returns:
        y_true, y_pred, y_prob
    """
    df_val = preparar_dataframe(path_val)
    X_val = df_val.drop(["label", "id"],  axis=1)
    y_val = df_val["label"]
    y_true = y_val
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]  # Probabilities for class 1
    
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = output_dir + "/predicciones.jsonl"

    with open(output_path, "w") as f:
        for idx, prob in zip(df_val["id"], y_prob):
            record = {
                "id": idx,
                "label": round(float(prob), 4)  # redondeo opcional
            }
            f.write(json.dumps(record) + "\n")


    f05u = f_05u(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    f1 = f1_score(y_true, y_pred)
    c1_score = c_at_1(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    brier_complement = 1 - brier
    roc_auc = roc_auc_score(y_true, y_prob)
    mean_score = np.mean([roc_auc, brier_complement, c1_score, f1, f05u])


    dic_evaluacion = {
        "roc-auc":float(roc_auc),
        "brier":float(brier_complement),
        "c@1": float(c1_score),
        "f1": float(f1),
        "f05u": float(f05u),
        "mean": float(mean_score),
        "confusion": [
            [
                float(tn),
            float(fp)
            ],
            [
            float(fn),
                float(tp)
            ]
        ]
    }
    return dic_evaluacion

def calculate_lexical_frequency(df):
    """
    Agrega una nueva columna con la frecuencia léxica media de cada texto usando TF-IDF.
    """
    # Instanciamos el vectorizador TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')  # Excluye palabras comunes (stop words)
    
    # Calculamos la matriz TF-IDF
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    
    # Calculamos la frecuencia léxica media para cada texto (media de los valores TF-IDF)
    # TF-IDF mide cuán relevante es una palabra dentro de un documento con respecto al resto del corpus.
    # Palabras muy comunes en el corpus (como "el", "y", "es", etc.) tienen un IDF bajo, y por lo tanto
    #  su valor TF-IDF será bajo.
    lexical_frequencies = tfidf_matrix.mean(axis=1).A1  # A1 convierte la matriz en un array 1D
    
    # Agregamos la nueva columna al DataFrame
    df['lexical_frequency'] = lexical_frequencies
    
    return df


if __name__ == "__main__":

    inputDataset= sys.argv[1]
    output_dir = sys.argv[2]
    modelo = joblib.load('modelo_tfid_todo.pkl') # Carga del modelo.
    dic = evaluate_model_predictions(modelo, inputDataset, output_dir )
    print(dic)
