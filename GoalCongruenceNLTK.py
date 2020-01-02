import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('all')

###############
# Table Clean #
###############

def goals_pruner(dataframe):
    """Removes NaN values, fixes labels and leaves just teamname and shared goal"""

    dataframe['Teamname'], dataframe['Shared Goal'] = dataframe.iloc[:, 0], dataframe.iloc[:, 1]
    return dataframe[['Teamname', 'Shared Goal']].dropna(subset=['Shared Goal'])

def table_to_bundles(table):
    """Takes table and returns one with shared goals as text bundles."""

    table['Shared Goal'] = [TextBundle(responses).lemmatized for responses in table['Shared Goal']]
    return table

def table_by_group(dataframe):
    """Takes table and returns one with entries grouped by team"""

    summed_table = dataframe.groupby('Teamname')
    return summed_table['Shared Goal'].agg(list)

def check_accuracy(dataframe, sim_values):
    """Takes table and returns groupby agg'd by sum. Then joins simulated values to the sim values based on index."""
    labeled_data = dataframe.iloc[:, :5].groupby('Teamname').agg(sum)
    labeled_data['Calculated Values'] = sim_values
    return labeled_data

def team_renamer(dataframe, column_index=0):
    """Relabels teamname so that they are all unique. Does so without changing team composition. Also verifies column_index
       is called called 'Teamname'"""

    new_teams, new_name, old_name, dataframe = [], -1, 0, verify_identity(dataframe, column_index)
    for team_name in dataframe['Teamname']:
        if team_name == old_name:
            new_teams.append(new_name)
        else:
            new_name += 1
            new_teams.append(new_name)
        old_name = team_name
    dataframe['Teamname'] = new_teams
    return dataframe

def binner(dataframe, column, div_labels=np.arange(1,6), lower=0, upper=1):
    """Creates bins to convert quantitative similarity values into qualitative measurements. Column identifier
       can be int or string. div labels MUST be array or list"""

    bin_size = (upper - lower) / len(div_labels)
    bins = np.arange(lower, upper, bin_size)
    if type(column) != str:
        column = dataframe.columns[column]

    new_values = []
    for value in dataframe[column]:
        n = 0
        while n < len(bins) and bins[n] < value:
            n += 1
        new_values.append(div_labels[n-1])
    dataframe['Calculated Labels'] = new_values
    return dataframe

def verify_identity(dataframe, column_index=0):
    if dataframe.columns[column_index] != 'Teamname':
        dataframe.columns =  list(dataframe.columns[0:column_index]) + ['Teamname'] + list(dataframe.columns[(column_index+1):])
    return dataframe

##############################
# Goal Relatedness Functions #
##############################

def cosine_sim(dataframe):
    """Takes csv_file and finds cosine similarity, returning values in a Series"""

    documents = dataframe
    labels = []
    for label in documents.index:
        labels.append(label)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    #cosine_quants gives us matrices comparing all the things!
    cosine_matrices = []
    for document in documents:
        sparse_matrix = tfidf_vectorizer.fit_transform(document)
        cosine_matrices.append(cosine_similarity(sparse_matrix, sparse_matrix))

    cosine_quants = []
    for matrix in cosine_matrices:
        row_sums = []
        for row in matrix:
            row_sum = np.sum(row)
            if row_sum < 1:
                row_sums.append(row_sum)
            else:
                row_sums.append(row_sum - 1)
        cosine_quants.append(np.sum(row_sums) / len(row_sums))
    cosine_quants = pd.Series(cosine_quants, index=labels)
    return cosine_quants
