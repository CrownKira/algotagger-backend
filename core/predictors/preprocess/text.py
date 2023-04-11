# Latent dirichlet allocation using gensim
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api

CLEAN_QUESTIONS_PATH = "core/predictors/preprocess/cleaned_questions.csv"
TRAIN_PATH = "core/predictors/preprocess/train.csv"
GLOVE_MODEL_PATH = "core/predictors/glove/glove_model.bin"

# remove punctuation, stopwords, word length > 2 and lemmatize
stop = set(stopwords.words("english"))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    """
    Function to clean the text
    Input: doc - string
    Output: normalized - string
    """
    doc = "".join([i for i in doc if not i.isdigit()])
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    word_length = " ".join(word for word in punc_free.split() if len(word) > 2)
    normalized = " ".join(
        lemma.lemmatize(word) for word in word_length.split()
    )
    return normalized


def clean_questions(qns):
    """
    Function to clean the questions
    Input: qns: list of strings
    Output: doc_clean - list of strings
    """
    return [clean(doc) for doc in qns]


def number_count(doc):
    """
    Function to count the number of numbers in the question
    Input: doc - string
    Output: count - int
    """
    count = 0
    for i in doc:
        if i.isdigit():
            count += 1
    return count


def number_count_questions(qns):
    """
    Function to count the number of numbers in the questions
    Input: qns: list of strings
    Output: doc_clean - list of ints
    """
    return [number_count(doc) for doc in qns]


def embed(test_documents, embed_type="tfidf"):
    """
    Function to embed the text
    Input: test_documents - string
    Output: embedding
    """

    df = pd.read_csv(CLEAN_QUESTIONS_PATH)
    # Preprocess your documents
    documents = df["cleaned_questions"].values
    tokenized_docs = [doc.lower().split() for doc in documents]
    test_tokenized_docs = [doc.lower().split() for doc in test_documents]
    test_doc_embeddings = []
    if embed_type == "w2v":
        # Create a Word2Vec model
        word2vec_model = Word2Vec(
            tokenized_docs, vector_size=50, window=5, min_count=1, workers=4
        )
        # Generate word2vec embeddings for the documents
        for doc in test_tokenized_docs:
            embedding = []
            for word in doc:
                if word in word2vec_model.wv:
                    embedding.append(word2vec_model.wv[word])
            test_doc_embeddings.append(sum(embedding))
    if embed_type == "glove":
        # Load the pre-trained GloVe embeddings
        glove_model = api.load("glove-wiki-gigaword-200")
        # model_path = GLOVE_MODEL_PATH
        # glove_model = KeyedVectors.load_word2vec_format(
        #     model_path, binary=True
        # )
        # Create a dictionary and tf-idf model
        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        tfidf = TfidfModel(corpus)
        for doc in test_tokenized_docs:
            embedding = []
            for word in doc:
                if word in glove_model and word in dictionary.token2id:
                    tfidf_weight = tfidf[dictionary.doc2bow([word])][0][1]
                    embedding.append(glove_model[word] * tfidf_weight)
            test_doc_embeddings.append(sum(embedding))
    if embed_type == "tfidf":
        df_train = pd.read_csv(TRAIN_PATH)
        trainX_raw = df_train[["cleaned_questions", "Number Count"]]
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_doc_embeddings = tfidf_vectorizer.fit_transform(
            trainX_raw["cleaned_questions"].values
        ).toarray()
        test_doc_embeddings = tfidf_vectorizer.transform(
            test_documents
        ).toarray()
    return test_doc_embeddings


def combine_number_count(embed_questions, number_count):
    return np.concatenate(
        (np.array(number_count).reshape(-1, 1), np.array(embed_questions)),
        axis=1,
    )


def dirty_to_clean(questions):
    """
    Function to clean the questions
    Input: qns: list of strings
    Output: doc_clean - list of strings
    """
    cleaned_questions = clean_questions(questions)
    number_count = number_count_questions(questions)
    embed_questions = embed(cleaned_questions)
    tmp = np.array(embed_questions)
    return combine_number_count(embed_questions, number_count)
