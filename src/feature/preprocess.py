from typing import List, Tuple
import re
from collections import Counter
import math

import numpy as np


def extract_ngrams(
    x_raw: str,
    ngram_range: tuple = (1, 3),
    token_pattern: str = r"\b[A-Za-z][A-Za-z]+\b",
    stop_words: list = [],
    vocab: set = set(),
    char_ngrams: bool = False,
) -> List[str]:
    x = []
    x_raw = re.sub("\s+", " ", x_raw).lower()
    if char_ngrams:
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(x_raw) - n + 1):
                x.append(x_raw[i : i + n])
    else:
        x_raw = re.findall(token_pattern, x_raw)
        words = [word for word in x_raw if word not in stop_words]
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(words) - n + 1):
                x.append(" ".join(words[i : i + n]))

    if len(vocab) > 0:
        x = [word for word in x if word in vocab]
    return x


def get_vocab(
    X_raw,
    ngram_range: tuple=(1, 3),
    token_pattern=r"\b[A-Za-z][A-Za-z]+\b",
    min_df=1,
    keep_topN=7,
    stop_words: list=[],
    char_ngrams=False,
) -> Tuple[set, Counter, Counter]:
    df = Counter()
    ngram_counts = Counter()

    for x in X_raw:
        x_ngram = extract_ngrams(
            x, ngram_range, token_pattern, stop_words, char_ngrams=char_ngrams
        )

        df.update(set(x_ngram))
        ngram_counts.update(x_ngram)

    df = {key: value for key, value in df.items() if value > min_df}
    topn = sorted(ngram_counts.items(), key=lambda x:x[1], reverse=True)[:keep_topN]
    ngram_counts = dict(topn)
    vocab = set([word for word in df.keys() if word in ngram_counts])
    return vocab, df, ngram_counts


def filter_content(
    X_raw: List[str],
    ngram_range: tuple = (1, 3),
    token_pattern: str = r"\b[A-Za-z][A-Za-z]+\b",
    stop_words: list = [],
    vocab: set = set(),
    char_ngrams: bool = False,
):
    for i in range(len(X_raw)):
        X_raw[i] = extract_ngrams(X_raw[i], ngram_range, token_pattern, stop_words, vocab, char_ngrams)
    return X_raw


def vectorise(X_ngrams: List[str], vocab: set) -> np.ndarray:
    X_vec = np.empty((len(X_ngrams), len(vocab)))
    vocab_lst = sorted(list(vocab))
    for row in range(X_vec.shape[0]):
        for column in range(X_vec.shape[1]):
            X_vec[row][column] = X_ngrams[row].count(vocab_lst[column])
    return X_vec


def compute_tfidf(vec: np.ndarray, df: dict, vocab: dict) -> dict:
    document_counts = len(vec)
    idf = {}
    for word in vocab:
        idf[word] = math.log10(document_counts/df[word])
    vocab_lst = sorted(list(vocab))

    tf = np.empty(vec.shape)
    for row in range(tf.shape[0]):
        for column in range(tf.shape[1]):
            tf[row][column] = vec[row][column] / vec[row].sum()
    
    tfidf = np.empty(vec.shape)
    for row in range(tfidf.shape[0]):
        for column in range(tfidf.shape[1]):
            tfidf[row][column] = tf[row][column] * idf[vocab_lst[column]]
        mean = np.mean(tfidf[row])
        std = np.std(tfidf[row])
        tfidf[row] = (tfidf[row] - mean) / std
    return tfidf


if __name__ == "__main__":
    text1 = "it's many good days and I have"
    text2 = "it's have some good ideas"
    texts = [text1, text2]
    vocab, df, nc = get_vocab(X_raw=texts, min_df=0)
    train = filter_content(texts, vocab=vocab)
    train_vec = vectorise(train, vocab=vocab)
    
    tfidf = compute_tfidf(train_vec, df, vocab)
    # print(train_vec)
