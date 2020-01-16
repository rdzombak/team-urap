import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import GoalCongruence as gc

wv_model = KeyedVectors.load_word2vec_format(r'~/Documents/GitHub/team-urap/test_data/GoogleNews-vectors-negative300.bin.gz', binary=True)

def cosine_sim_word2vec(dataframe):
    """Takes csv_file and finds cosine similarity, returning values in a Series"""
    documents = dataframe
    labels = []
    for label in documents.index:
        labels.append(label)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
