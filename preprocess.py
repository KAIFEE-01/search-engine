"""
preprocess.py
Processes Speakers.csv, cleans text, computes CountVectorizer and Cosine Similarity,
and dumps data.pkl and similarity.pkl.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def process_data(csv_path="Speakers.csv", data_out="data.pkl", similarity_out="similarity.pkl"):
    start_time = time.time()
    print(f"Loading {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source dataset '{csv_path}' not found.")

    speakers = pd.read_csv(csv_path)
    print(f"Loaded {len(speakers)} rows.")

    # Drop unnecessary columns if present
    drop_cols = [c for c in ['main_category', 'sub_category'] if c in speakers.columns]
    if drop_cols:
        speakers.drop(columns=drop_cols, inplace=True)

    # Clean missing values in critical fields
    speakers['name'] = speakers['name'].fillna('').astype(str)
    speakers.reset_index(drop=True, inplace=True)

    # Keywords preparation
    print("Preparing keywords and stemming...")
    stemmer = PorterStemmer()

    # Lowercase & strip punctuation
    keywords_series = speakers['name'].str.lower().str.replace(r'[^\w\s]', ' ', regex=True)

    def stem_text(text):
        return ' '.join(stemmer.stem(w) for w in text.split() if w)

    speakers['#keywords'] = keywords_series.apply(stem_text)

    # Vectorization
    print("Vectorizing keywords using CountVectorizer (max_features=5000)...")
    cv = CountVectorizer(max_features=5000, stop_words='english', dtype=np.uint8)
    vector = cv.fit_transform(speakers['#keywords'])

    # Similarity Matrix (float32 reduces memory by 50% with zero loss in ranking precision)
    print("Computing cosine similarity matrix...")
    similarity = cosine_similarity(vector, dense_output=True).astype(np.float32)

    # Save to disk
    print(f"Saving DataFrame to {data_out}...")
    with open(data_out, 'wb') as f:
        pickle.dump(speakers, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saving Similarity Matrix ({similarity.nbytes / (1024 * 1024):.1f} MB) to {similarity_out}...")
    with open(similarity_out, 'wb') as f:
        pickle.dump(similarity, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - start_time
    print(f"Data preprocessing completed in {elapsed:.2f} seconds.")
    return speakers, similarity


if __name__ == "__main__":
    process_data()

