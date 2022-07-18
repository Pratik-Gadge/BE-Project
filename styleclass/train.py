import os
import pickle

from styleclass.settings import N_FEATURES, NGRAM_RANGE
from styleclass.features import get_features, select_features_chi2

from sklearn.linear_model import LogisticRegression

###############################################################################
# Training                                                                    #
###############################################################################


def get_default_algorithm(random_state):
    return LogisticRegression(random_state=random_state,
                              solver='liblinear',
                              multi_class='ovr')


def train(dataset, random_state=0):
    '''Train the model'''

    count_vectorizer, tfidf_transformer, features = \
        get_features(dataset['string'], response=dataset['style'],
                     nfeatures=N_FEATURES,
                     feature_selector=select_features_chi2, ngrams=NGRAM_RANGE)
    model = get_default_algorithm(random_state).fit(features, dataset['style'])
    return count_vectorizer, tfidf_transformer, model


def store_model(fn, count_vectorizer, tfidf_transformer, model):
    '''Store the trained model in a file'''

    with open(fn, 'wb') as file:
        pickle.dump((count_vectorizer, tfidf_transformer, model), file)


def read_model(fn):
    '''Read the model from a file'''

    with open(fn, 'rb') as file:
        model = pickle.load(file)
    return tuple(model)


def get_default_model():
    return read_model(
        os.path.join(os.path.dirname(__file__), 'models/default.model'))
