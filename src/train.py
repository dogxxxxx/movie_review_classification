import pandas as pd
import numpy as np

from feature import get_vocab, filter_content, vectorise, compute_tfidf
from utils import confusion_matrix
from model import LogisticRegression


STOP_WORDS = ['a', 'in', 'on', 'at', 'and', 'or', 'to', 'the', 
              'of', 'an', 'by', 'as', 'is', 'was', 'were', 'been', 'be', 
              'are', 'for', 'this', 'that', 'these', 'those', 'you', 'i', 
              'it', 'he', 'she', 'we', 'they', 'will', 'have', 'has', 
              'do', 'did', 'can', 'could', 'who', 'which', 'what', 
              'his', 'her', 'they', 'them', 'from', 'with', 'its']


def train():
    train_data = pd.read_csv("data/train.csv")
    dev_data = pd.read_csv("data/dev.csv")
    test_data = pd.read_csv("data/test.csv")

    train_content = train_data.loc[:, "content"].tolist()
    dev_content = dev_data.loc[:, "content"].tolist()
    test_content = test_data.loc[:, 'content'].tolist()
    train_label = train_data.loc[:, "label"].to_numpy().reshape(-1, 1)
    dev_label = dev_data.loc[:, "label"].to_numpy().reshape(-1, 1)
    test_label = test_data.loc[:, 'label'].to_numpy().reshape(-1, 1)

    vocab, df, ngrams = get_vocab(
        train_content, min_df=30, keep_topN=100, stop_words=STOP_WORDS, char_ngrams=True
    )
    index = 0
    word2id = dict()
    id2word = dict()
    for word in vocab:
        word2id[word] = index
        id2word[index] = word
        index += 1
    train_content = filter_content(train_content, vocab=vocab)
    dev_content = filter_content(dev_content, vocab=vocab)
    test_content = filter_content(test_content, vocab=vocab)

    train_vec = vectorise(train_content, vocab)
    dev_vec = vectorise(dev_content, vocab)
    test_vec = vectorise(test_content, vocab)

    train_tfidf = compute_tfidf(train_vec, df, vocab)
    print(train_tfidf.shape)
    dev_tfidf = compute_tfidf(dev_vec, df, vocab)
    test_tfidf = compute_tfidf(test_vec, df, vocab)
    model = LogisticRegression()
    model.fit(train_tfidf, train_label, dev_tfidf, dev_label, lr=0.001, epochs=100, tolerance=0.00001, alpha=0.001)
    test_predict = model.predict(test_tfidf)

    tp, tn, fp, fn = confusion_matrix(test_label, test_predict)
    print(tp, tn, fp, fn)


if __name__ == "__main__":
    train()
