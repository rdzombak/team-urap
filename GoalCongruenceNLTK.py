import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('all')

class TextBundle:
    """Pass in a string to make a bundle of text types"""

    def __init__(self, string):
        """Each TextBundle has the following instance attributes:

        --main attributes--
        self.string = the unchanged string
        self.words = a list of lowercased words without punctuation
        self.no_stop = a list of lowercased wrods without stop words

        --data cleaning intermediates--
        self.no_yn = self.no_stop with no initial yes's or no's
        self.pos_tagged = the list self.no_stop with pos_tags"""

        self.string = string
        self.words = TextBundle.into_words(string)
        self.no_stop = TextBundle.into_no_stop(string)

        self.no_yn = TextBundle.into_no_yn(self.no_stop)
        self.pos_tags = [(word, TextBundle.get_wordnet_pos(word)) for word in self.no_yn]
        self.lemmatized = TextBundle.into_lemmatized(self.no_yn)

    def __str__(self):
        return self.string

    def into_string(attrib):
        """Converts nonstring attributes into string objects."""

        assert type(attrib) != str, "It's already a string!"
        text = ''
        if type(attrib) == list:
            for word in attrib:
                text += word + ' '
            return text

    def into_words(attrib):
        """Converts TextBundle into list of words"""

        assert type(attrib) == str, "This requires a string!"
        return [word for word in nltk.word_tokenize(attrib.lower()) if word.isalpha()]

    def into_no_stop(attrib):
        """Removes stopwords from a list of words"""

        assert type(attrib) == str, "This requires a string!"
        return [word for word in TextBundle.into_words(attrib) if word not in set(nltk.corpus.stopwords.words('english'))]

    def into_no_yn(attrib):
        """Removes yes's and no's from beginning of response"""

        assert type(attrib) == list, "This requires a list!"
        for word in attrib:
            if word in ['yes', 'no']:
                del word
        return attrib

    def into_lemmatized(attrib):
        "Lemmatizes and returns list of words!"

        assert type(attrib) == list, "This requires a list!"
        return [WordNetLemmatizer().lemmatize(word, TextBundle.get_wordnet_pos(word)) for word in attrib]

    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

###############
# Table Clean #
###############

def goals_pruner(table):
    """Removes NaN values, fixes labels and leaves just teamname and shared goal"""

    table['Teamname'], table['Shared Goal'] = table.iloc[:, 0], table.iloc[:, 1]
    return table[['Teamname', 'Shared Goal']].dropna(subset=['Shared Goal'])

######################
# Document Functions #
######################

def tf(word, blob):
    '''creates vectors based on word frequency'''

    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    '''Tells us how many blobs in the bloblist contain the word'''

    return sum(1 for blob in bloblist if word in blob.words)


def idf(word, bloblist):
    '''Inverse Document Frequency'''

    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def tfidf(word, blob, bloblist):
    '''creates TFIDF word vectors'''

    return tf(word, blob) * idf(word, bloblist)


def glossary_maker(column_or_series):
    '''Creates a wordlist of all words used in a column, series, or wordlist'''

    for response in column_or_series:
        unsorted_master_list += response

##############################
# Goal Relatedness Functions #
##############################

def jaccard_calc(csv_file):
    """Takes in a table and returns another with a column of jaccard_values"""

    table = goals_pruner(pd.read_csv(csv_file))
    table['Shared Goal'] = [TextBundle(responses).lemmatized for responses in table['Shared Goal']]
    summed_table = table.groupby('Teamname')[['Shared Goal']].sum()
    summed_table['Shared Goal'] = table.groupby('Teamname')['Shared Goal'].apply(list)
    table_with_values = summed_table

    return table
