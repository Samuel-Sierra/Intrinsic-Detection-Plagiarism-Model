#intrinsec detection plagiarism model
import os
import spacy
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

nlp = spacy.load("en_core_web_sm")

"""
Hyperparameters
"""
window_size = 200
overlap = 100

"""
Módulo de preprocesamiento y segmentación
"""
def get_sliding_windows(text, window_size, overlap):
    """Divide el texto en ventanas solapadas basadas en tokens."""
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_space]
    
    windows = []
    for i in range(0, len(tokens) - window_size + 1, overlap):
        windows.append(tokens[i : i + window_size])
    return windows

#----!!Falta preprocesamiento!!----

"""
Módulo de extracción de características
"""
def extract_features(window_tokens):
    text_segment = " ".join([t.text for t in window_tokens])
    words_only = [t.text.lower() for t in window_tokens if t.is_alpha]
    
    # --- 1. Palabras funcionales (Stopwords) ---
    stop_count = sum(1 for t in window_tokens if t.is_stop)
    
    # --- 2. N-gramas de caracteres ---
    char_vec = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    try:
        char_matrix = char_vec.fit_transform([text_segment])
        char_ngram_count = char_matrix.sum()
    except: 
        char_ngram_count = 0

    # --- 3. N-gramas de categorías gramaticales (POS) ---
    pos_tags = [t.pos_ for t in window_tokens]
    pos_bigrams = ["_".join(pair) for pair in zip(pos_tags, pos_tags[1:])]
    
    # --- 4. Frecuencia de Signos de Puntuación ---
    punct_count = sum(1 for t in window_tokens if t.is_punct)
    
    # --- 5. Frecuencia de Caracteres Especiales ---
    special_chars = "@#$%^&*+|<>=_~"
    special_count = sum(text_segment.count(c) for c in special_chars)
    
    # --- 6. Longitud de oraciones (en esta ventana) ---
    temp_doc = nlp(text_segment)
    sentences = list(temp_doc.sents)
    avg_sent_len = len(window_tokens) / len(sentences) if sentences else 0
    
    # --- 7. Medida de riqueza léxica (TTR) ---
    ttr = len(set(words_only)) / len(words_only) if words_only else 0

    return {
        "stop_freq": stop_count / len(window_tokens),
        "char_ngrams": char_ngram_count,
        "pos_bigrams_unique": len(set(pos_bigrams)),
        "punct_freq": punct_count / len(window_tokens),
        "special_freq": special_count / len(text_segment) if text_segment else 0,
        "avg_sent_len": avg_sent_len,
        "lexical_richness_ttr": ttr
    }

def transformation_features(df):
    print(df)
    x = df[features_cols]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    svd = TruncatedSVD(n_components=3, random_state=42)
    x_reduced = svd.fit_transform(x_scaled)

    df_reduced = pd.DataFrame(
        x_reduced, 
        columns=[f'Component_{i+1}' for i in range(x_reduced.shape[1])]
    )

    df_final = pd.concat([df[['file', 'window_id']], df_reduced], axis=1)
    return df_reduced, df_final

def clasiffier_temp(df_reduced, df_final):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df_final['cluster'] = kmeans.fit_predict(df_reduced)

    counts = df_final['cluster'].value_counts()
    clase_predominante = counts.idxmax()

    df_final['style_class'] = df_final['cluster'].apply(
        lambda x: 'Clase 1 (Original)' if x == clase_predominante else 'Clase 2 (Sospechoso)'
    )

    print("\nConteo de ventanas por clase:")
    print(df_final['style_class'].value_counts())

"""
Main loop
"""
folder_path = 'part1/' #cambiar
all_data = []
features_cols = ["stop_freq", "char_ngrams", "pos_bigrams_unique", 
                 "punct_freq", "special_freq", "avg_sent_len", "lexical_richness_ttr"]

for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            content = f.read()
            windows = get_sliding_windows(content, window_size, overlap)
            
            print("break1")
            for idx, win in enumerate(windows):
                feat = extract_features(win)
                feat['file'] = file
                feat['window_id'] = idx
                all_data.append(feat)

            
            df = pd.DataFrame(all_data)
            df_reduced, df_final = transformation_features(df)
            clasiffier_temp(df_reduced, df_final)
            break