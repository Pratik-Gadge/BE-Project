import csv
import numpy as np
import os
import pandas as pd
import re
import scipy.sparse as sp

from styleclass.settings import REGEX_REMOVE, MIN_REF_LEN, REGEX_MONTH_REMOVE, \
    REGEX_RANDOM_REMOVE, REGEX_WORD_TO_TOKEN, NGRAM_RANGE

from random import random, randint, seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2

###############################################################################
# Dataset processing                                                          #
###############################################################################


def read_dataset(path):
    '''Read dataset from a file'''

    dataset = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        dataset = [row for row in reader]
    return pd.DataFrame(dataset, columns=['doi', 'style', 'string'])


def remove_technical_parts(s):
    '''Remove Crossref-related fragments from the reference strings'''

    for r in REGEX_REMOVE:
        s = re.sub(r, '', s.strip())
    s = re.sub('\n', ' ', s)
    s = re.sub(' +', ' ', s)
    return s.strip()


def clean_dataset(dataset):
    '''Clean dataset. This removes empty and short reference strings and
    removes Crossref-related fragments from the strings'''

    dataset = dataset[pd.notnull(dataset['string'])].copy()
    dataset['string'] = dataset['string'].apply(remove_technical_parts)
    dataset = dataset[dataset.apply(lambda x: len(x['string']) >= MIN_REF_LEN,
                                    axis=1)]
    return dataset


def add_noise_string(s):
    '''Randomly remove from the reference string: the name of the month,
    enumeration at the beginning and dot at the end'''

    for r, e in REGEX_MONTH_REMOVE.items():
        if re.search(r, s) and random() < 0.75:
            s = re.sub(r, e, s).strip()
    for r in REGEX_RANDOM_REMOVE:
        if random() < 0.5:
            s = re.sub(r, '', s).strip()
    return s


def add_noise(dataset, random_state=0):
    '''Add noise to the reference strings in the dataset'''

    seed(random_state)
    dataset['string'] = dataset['string'].apply(add_noise_string)
    return dataset


def rearrange_tokens(s):
    '''Randomly rearrange tokens in the reference string'''

    parts = s.split(' ')
    for _ in range(randint(10, 20)):
        i1, i2 = randint(0, len(parts) - 1), randint(0, len(parts) - 1)
        parts[i1], parts[i2] = parts[i2], parts[i1]
    return ' '.join(parts)


def generate_unknown(dataset, n, random_state=0):
    '''Generate a dataset of "unknown" reference strings'''

    dataset_unknown = dataset.sample(n, random_state=random_state)
    dataset_unknown['string'] = \
        dataset_unknown['string'].apply(rearrange_tokens)
    dataset_unknown['style'] = 'unknown'
    return dataset_unknown


def get_default_training_dataset():
    return read_dataset(
        os.path.join(os.path.dirname(__file__), 'datasets/training.csv'))


def get_default_test_dataset():
    return read_dataset(
        os.path.join(os.path.dirname(__file__), 'datasets/test.csv'))


###############################################################################
# Features                                                                    #
###############################################################################


def tokens_to_classes(s):
    '''Map tokens to token classes'''

    for r, t in REGEX_WORD_TO_TOKEN.items():
        s = re.sub(r, ' ' + t + ' ', s)
    s = 'start ' + s + 'end'
    s = re.sub(' +', ' ', s).strip()
    return s


def select_features_rf(tfidf, response, feature_names, nfeatures):
    '''Select features using feature importance from Random Forest'''

    if nfeatures >= len(feature_names):
        return feature_names
    rf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=5)
    rf_model = rf.fit(tfidf, response)
    feature_importances = np.argsort(rf_model.feature_importances_)
    feature_names = np.array(feature_names)
    feature_names = feature_names[feature_importances]
    return feature_names[-nfeatures:]


def select_features_chi2(tfidf, response, feature_names, nfeatures):
    '''Select features using Chi-squared correlations'''

    if nfeatures >= len(feature_names):
        return feature_names
    feature_names_sorted = []
    for label in list(set(response)):
        features_chi2 = chi2(tfidf, response == label)[0]
        indices = np.argsort(features_chi2)
        fns = np.array(feature_names)
        fns = fns[indices][::-1]
        feature_names_sorted.append(fns)
    feature_names = set()
    for i in range(nfeatures):
        if len(feature_names) == nfeatures:
            break
        nf = [x[i] for x in feature_names_sorted]
        for n in nf:
            if len(feature_names) == nfeatures:
                break
            feature_names.add(n)
    return feature_names


def get_tfidf_features(strings,
                       response=None,
                       count_vectorizer=None,
                       tfidf_transformer=None,
                       nfeatures=None,
                       ngrams=NGRAM_RANGE,
                       feature_selector=None):
    '''Extract TF-IDF from reference strings'''

    if count_vectorizer is None:
        # fit and calculate features (train set mode)
        freq_nfeatures = None
        if feature_selector is None:
            freq_nfeatures = nfeatures
        count_vectorizer = CountVectorizer(preprocessor=tokens_to_classes,
                                           max_features=freq_nfeatures,
                                           ngram_range=ngrams)
        counts = count_vectorizer.fit_transform(strings)
        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(counts)
        if feature_selector is not None and nfeatures is not None \
                and response is not None:
            # feature selection
            feature_names = count_vectorizer.get_feature_names()
            if nfeatures < len(feature_names):
                feature_names = feature_selector(tfidf, response,
                                                 feature_names, nfeatures)
            count_vectorizer = CountVectorizer(preprocessor=tokens_to_classes,
                                               ngram_range=ngrams,
                                               vocabulary=feature_names)
            counts = count_vectorizer.fit_transform(strings)
            tfidf_transformer = TfidfTransformer()
            tfidf = tfidf_transformer.fit_transform(counts)
    else:
        # calculate features (test set mode)
        counts = count_vectorizer.transform(strings)
        tfidf = tfidf_transformer.transform(counts)
    return count_vectorizer, tfidf_transformer, tfidf


def get_features(strings,
                 response=None,
                 count_vectorizer=None,
                 tfidf_transformer=None,
                 nfeatures=None,
                 ngrams=NGRAM_RANGE,
                 feature_selector=None):
    '''Extract full feature vector from reference strings'''

    count_vectorizer, tfidf_transformer, features = \
        get_tfidf_features(strings, response=response, nfeatures=nfeatures,
                           count_vectorizer=count_vectorizer,
                           tfidf_transformer=tfidf_transformer,
                           ngrams=ngrams, feature_selector=feature_selector)
    lengths = [[len(s)] for s in strings]
    features = sp.hstack((features, sp.csr_matrix(lengths)))

    return count_vectorizer, tfidf_transformer, features
