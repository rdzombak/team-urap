import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

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

def series_builder(dataframe, array_or_list):
    """Takes teamname labels from a dataframe and makes it the index of array to make a Series"""

    documents = dataframe
    labels = []
    for label in documents.index:
        labels.append(label)
    new_series = pd.Series(array_or_list, index=labels)
    return new_series

def tokenize(string, category_string=' '):
    """Takes in string and splits text up when encountering text in the specified category. Returns an array"""

    tokens = np.array([])
    curr_token = ""
    for char in string:
        if char not in category_string:
            curr_token += char
        else:
            if curr_token:
                tokens = np.append(tokens, curr_token)
                curr_token = ""
    return tokens


##############################
#      Word Embeddings       #
##############################


def tfidf_embed(document):
    """Generates tfidf matrices that can then be used to for comparison and sorts out stop words"""

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    return tfidf_vectorizer.fit_transform(document)

##############################
#   Similarity Calculators   #
##############################

class Cosine_sim:
    def __init__(self, dataframe, embedding):
        self.individ = np.array([]) #REQUIRED INFO
        self.team = np.array([]) #REQUIRED INFO
        for team in Cosine_sim.sim_matrix(dataframe, embedding):
            team_member = Cosine_sim.personal_sim(team)
            self.individ = np.append(self.individ, team_member)
            self.team = np.append(self.team, np.mean(team_member))

    def sim_matrix(dataframe, embedding):
        """Creates series with index of dataframe with cosine similarity calculations for each team"""

        cosine_matrices = []
        for document in dataframe:
            wv_matrix = embedding(document)
            cosine_matrices.append(cosine_similarity(wv_matrix, wv_matrix))
        return cosine_matrices

    def personal_sim(matrices):
        """Returns an ARRAY of similarity values that correspond to each person"""

        personal_sim_values = np.array([])
        for row in matrices:
            if np.sum(row) and len(row) - 1:
                personal_sim_values = np.append(personal_sim_values, (np.sum(row) - 1) / (len(row) - 1))
            else:
                personal_sim_values = np.append(personal_sim_values, 0)
        return personal_sim_values

