import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('all')

###############
# Table Clean #
###############

def goals_pruner(table):
    """Removes NaN values, fixes labels and leaves just teamname and shared goal"""

    table['Teamname'], table['Shared Goal'] = table.iloc[:, 0], table.iloc[:, 1]
    return table[['Teamname', 'Shared Goal']].dropna(subset=['Shared Goal'])

def table_to_bundles(table):
    """Takes table and returns one with shared goals as text bundles."""

    table['Shared Goal'] = [TextBundle(responses).lemmatized for responses in table['Shared Goal']]
    return table

def table_by_group(table):
    """Takes table and returns one with entries grouped by team"""

    summed_table = table.groupby('Teamname')
    return summed_table['Shared Goal'].agg(list)

##############################
# Goal Relatedness Functions #
##############################

def cosine_sim(csv_file):
    documents = table_by_group(goals_pruner(pd.read_csv(csv_file)))
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    cosine_quants: []
    for document in documents:
        sparse_matrix = tfidf_vectorizer.fit_transform(document)
        print(cosine_similarity(sparse_matrix, sparse_matrix))
