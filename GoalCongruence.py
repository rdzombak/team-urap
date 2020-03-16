import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

###############
# Table Clean #
###############

def goals_pruner(dataframe):
    """Returns a new dataframe generated by removing NaN
       values, reassigning proper 'teamname' and 'shared
       goal' column titles, and dropping all other columns
       from input DATAFRAME."""

    dataframe['Teamname'], dataframe['Shared Goal'] = dataframe.iloc[:, 0], dataframe.iloc[:, 1]
    return dataframe[['Teamname', 'Shared Goal']].dropna(subset=['Shared Goal'])

def table_to_bundles(table):
    """Takes table and returns one with shared goals as text bundles."""

    table['Shared Goal'] = [TextBundle(responses).lemmatized for responses in table['Shared Goal']]
    return table

def series_by_team(dataframe):
    """Returns series indexed by teamname with entries
       containing lists of team responses generated from
       aggregating DATAFRAME by teamname."""

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

class Diagnostics:
    """Contains all tools necessary to determine accuracy of results"""

    def __init__(self, sim_object):
        """Creates all dataframes required for quantifying result accuracy

           SIM OBJECT: Simulated values from another object passed in
           INDIVID QUANT: Dataframe with COLUMNS teamname, AGGREGATED responses in lists, and individual scores
           TEAM QUANT: Dataframe with COLUMNS teamname, calculated score, calculated label, actual rating"""

        self.sim_object = sim_object
        self.individ_quant = self.comparison_builder(False) #Some dataframe containing individs
        self.team_quant = self.comparison_builder() #some dataframe containing teams
        self.team_ord = self.quant_to_ord()

    def quant_to_ord(self, column='Calculated Values', num_labels=5, lower=0, upper=1, team_calc=True):
        """Returns binned similarity measurements in a new column called
           'Calculated Labels'. Created by converting quantitative similarity
           values in COLUMN to qualitative measurements by placing each
           value in NUM_LABELS bins ranging from LOWER to UPPER. Values
           organized by team or individual depending on value of TEAM_CALC."""

        if team_calc:
            dataframe = self.team_quant
        else:
            dataframe = self.individ_quant

        bins = np.linspace(lower, upper, num_labels + 1)
        if type(column) != str:
            column = dataframe.columns[column]
        ord_values = []
        for value in dataframe[column]:
            lower_bound_index = 0
            while lower_bound_index < len(bins) - 1 \
                    and bins[lower_bound_index] < value:
                lower_bound_index += 1
            ord_values.append(lower_bound_index)
        dataframe['Calculated Labels'] = ord_values
        return dataframe

    def comparison_builder(self, team_calc=True):
        """RETURNS table GROUPED BY teamname and AGGREGATED by sum,
           and JOINS simulated values to the labels based on index"""

        if team_calc:
            alignment = self.sim_object.dataframe.iloc[:, :5].groupby(
                'Teamname').agg(sum)[['Degree of goal alignment  (1 = lo, 5 = hi)']]
            alignment['Calculated Values'] = self.sim_object.team
            return alignment
        else:
            individ_labels = goals_pruner(self.sim_object.dataframe)
            individ_labels['Calculated Values'] = self.sim_object.individ
            return individ_labels


class Similarity_Calcs:
    """Object's attributes represent data structures generated from applying
       similarity calculations across team survey."""

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.series = series_by_team(goals_pruner(dataframe))

        self.individ = np.array([])
        self.team = np.array([])

class Wmd_Sim(Similarity_Calcs):
    """Object's attributes represent data structures generated from applying
       cosine similarity across team survey."""

    def __init__(self, dataframe, embedding=tfidf_embed):
        """Creation of cosine_sim object involves creation of three object
                   instances:
                   DATAFRAME: dataframe that was passed in. RAW DATA!
                   SERIES: series made after data cleaning.

                   INDIVID: array containing all avg. similarity values per team member
                   in order of entry
                   TEAM: array containing all avg. similarity values for teams in order
                   of entry"""

        super().__init__(dataframe)

    def sim_matrix(series, embedding):
        """Creates series with index of series with cosine similarity
           calculations for each team"""



class Cosine_Sim(Similarity_Calcs):
    """Object's attributes represent data structures generated from applying
       cosine similarity across team survey."""

    def __init__(self, dataframe, embedding=tfidf_embed):
        """Creation of cosine_sim object involves creation of three object
           instances:
           DATAFRAME: dataframe that was passed in. RAW DATA!
           SERIES: series made after data cleaning.

           INDIVID: array containing all avg. similarity values per team member
           in order of entry
           TEAM: array containing all avg. similarity values for teams in order
           of entry"""

        super().__init__(dataframe)

        for team in Cosine_Sim.sim_matrix(self.series, embedding):
            team_member = Cosine_Sim.personal_sim(team)
            self.individ = np.append(self.individ, team_member)
            self.team = np.append(self.team, np.mean(team_member))

    def sim_matrix(series, embedding):
        """Creates series with index of series with cosine similarity
           calculations for each team"""

        """Returns list of matrices created by """

        cosine_matrices = []
        for document in series:
            wv_matrix = embedding(document)
            cosine_matrices.append(cosine_similarity(wv_matrix, wv_matrix))
        return cosine_matrices

    def personal_sim(matrices):
        """Returns an array of similarity values that correspond to
           each person by taking the average of each row in MATRICES."""

        personal_sim_values = np.array([])
        for row in matrices:
            if np.sum(row) and len(row) - 1:
                personal_sim_values = np.append(personal_sim_values, (np.sum(row) - 1) / (len(row) - 1))
            else:
                personal_sim_values = np.append(personal_sim_values, 0)
        return personal_sim_values

