import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import GoalCongruence as gc

wv_model = KeyedVectors.load_word2vec_format(r'~/Documents/GitHub/team-urap/test_data/GoogleNews-vectors-negative300.bin.gz', binary=True)

def word2vec_embed(document):
    """Takes in document and converts statements into sentence vectors"""

    #First, clean data of all stop words
    sentence_vectors = []
    for member in document:
        statement = gc.tokenize(member, ",.?!;:/\|[]{}()<>& _")
        sentence_vector = [0 for i in np.arange(300)]
        for word in statement:
            if word not in stopwords.words('english') and word in wv_model.vocab:
                sentence_vector = [np.sum(i) for i in zip(wv_model[word], sentence_vector)]
                sentence_vector /= np.linalg.norm(sentence_vector)
        sentence_vectors.append(sentence_vector)
    return sentence_vectors

