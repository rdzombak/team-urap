import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from textblob import blob
import nltk
import math

def lowercase(column):
    '''Converts a dataframe column into strings and lowercases everything'''
    return column.str.lower()

def lowercase_textblobs(column):
    '''Converts a dataframe column into strings, lowercases everything, and makes them textblobs'''
    return lowercase(column).apply(TextBlob)

# Hack to avoid strange errors with textblob's lemmatizer.
def _penn_to_wordnet(tag):
    _wordnet = blob._wordnet
    """Converts a Penn corpus tag into a Wordnet tag."""
    if tag in ("NN", "NNS", "NNP", "NNPS"):
        return _wordnet.NOUN
    if tag in ("JJ", "JJR", "JJS"):
        return _wordnet.ADJ
    if tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
        return _wordnet.VERB
    if tag in ("RB", "RBR", "RBS"):
        return _wordnet.ADV
    return _wordnet.NOUN
blob._penn_to_wordnet = _penn_to_wordnet

def column_lemmatize(column):
    '''Lemmatizes down an entire column'''
    lemmatized_goals = []
    for response in column.apply(lambda goal: goal.tags):
        lemmatized_words = TextBlob('').words
        for word_and_tag in response:
            corrected_tag = _penn_to_wordnet(word_and_tag[1])
            lemmatized_words.append(word_and_tag[0].lemmatize(corrected_tag))
        lemmatized_goals.append(lemmatized_words)
    return lemmatized_goals

def remove_y_and_n(column):
    '''removes yes and no responses from responses in a column'''
    for response in column:
        if response[0] == 'yes' or response[0] == 'no':
            del response[0]
    return column

def array_to_blob(array):
    '''Convert an array of strings to a blob'''
    converted_blob = ''
    for word in array:
        converted_blob += word + ' '
    return TextBlob(converted_blob)

def remove_stop_words(column):
    '''gets rid of any words that are on the stop word list'''
    cleaned_responses = []
    for response in column:
        cleaned_response = TextBlob('').words
        for word in response:
            if word not in stop_words:
                cleaned_response.append(word)
        cleaned_responses.append(cleaned_response)
    return cleaned_responses

def cleaning_and_lemmatizing(table):
    '''lemmatizes, removes yes and no responses, and stop words in one fell swoop. Returns WORDLISTS'''
    table['Teamname'] = table.iloc[:, 0]
    table = table[['Teamname', 'Shared Goal']].dropna(subset=['Shared Goal'])
    if type(table.loc[0, 'Teamname']) == str:
        table['Teamname'] = lowercase(table['Teamname'])
    else:
        string_labels = []
        for i in table['Teamname']:
            string_labels.append(str(i))
    table['Shared Goal'] = remove_stop_words(remove_y_and_n(column_lemmatize(lowercase_textblobs(table['Shared Goal']))))
    return table

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
    unsorted_master_list = TextBlob('').words
    for response in column_or_series:
        unsorted_master_list += response

    #creating string and blob of all words used in the survey
    unsorted_master_str = ''
    for word in unsorted_master_list:
        unsorted_master_str += word.__str__() + ' '
    unsorted_master_blob = TextBlob(unsorted_master_str)

    #get rid of duplicates in string list
    master_array = np.array([])
    master_freq = unsorted_master_blob.word_counts
    for entry in list(master_freq):
        master_array = np.append(master_array, entry)
    return master_array

def remove_dupes(wordlist):
    '''Takes a wordlist and removes all duplicate words used within it, returning a wordlist'''
    dupe_dict = TextBlob(wordlist.__str__()).word_counts
    return list(dupe_dict)

######

def jaccard_calc(csv_file):
    '''Takes in a a table and Returns another with a column of jaccard_values for each team'''
    table = pd.read_csv(csv_file)

    #Cleaning, lemmatizing, then grouping responses by team into lists of wordlists.
    cleaned_table = cleaning_and_lemmatizing(table)
    summed_table = cleaned_table.groupby('Teamname')[['Shared Goal']].sum()
    summed_table['Shared Goal'] = cleaned_table.groupby('Teamname')['Shared Goal'].apply(list)
    table_with_values = summed_table

    #Jaccard Calculation
    jaccard_value = np.array([])
    for responses in table_with_values['Shared Goal']:
        response_length = 0
        numerator = 0
        past_responses = TextBlob('').words
        for response in responses:
            past_responses += remove_dupes(response)
            response_length += len(response)
        past_word_freq = TextBlob(past_responses.__str__()).word_counts
        for word in list(past_word_freq):
            if past_word_freq[word] > 1:
                numerator += past_word_freq[word] - 1
        jaccard_value = np.append(jaccard_value, numerator / response_length)
    table_with_values['jaccard value'] = jaccard_value
    return table_with_values

def cosine_sim(csv_file):
    '''Takes in a table and returns a table with a column of dot products for each team and generates a table with word vector dimensions'''
    table = pd.read_csv(csv_file)
    table_with_values = cleaning_and_lemmatizing(table)
    full_wordlist = glossary_maker(table_with_values['Shared Goal'])

    shared_blobs = []
    for response in table['Shared Goal']:
        intermed_str = ''
        for word in response:
            intermed_str += word.__str__() + ' '
        shared_blobs.append(TextBlob(intermed_str))
    table_with_values['Shared Blob'] = shared_blobs

    for word in full_wordlist:
        word_vector = []
        for blob in table_with_values['Shared Blob']:
            word_vector.append(tfidf(word, blob, shared_blobs))
        table_with_values[word.__str__()] = word_vector

    multiplied_cosine_sim = table_with_values.groupby('Teamname').agg(np.prod)
    return multiplied_cosine_sim.sum(axis=1)

stop_words = array_to_blob(np.array(['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']))
