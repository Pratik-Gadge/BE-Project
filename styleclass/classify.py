import logging
import pandas as pd

from styleclass.settings import N_FEATURES, NGRAM_RANGE
from styleclass.features import get_features, select_features_chi2

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

###############################################################################
# Prediction                                                                  #
###############################################################################


def classify(ref_strings, count_vectorizer, tfidf_transformer, model):
    '''Classify reference strings into citation styles'''

    if isinstance(ref_strings, str):
        ref_strings = [ref_strings]
    _, _, features = get_features(ref_strings,
                                  count_vectorizer=count_vectorizer,
                                  tfidf_transformer=tfidf_transformer)
    return model.predict(features)


###############################################################################
# Evaluation                                                                  #
###############################################################################

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def evaluate_cv(data,
                algorithm,
                folds=5,
                feature_fun=get_features,
                nfeatures=N_FEATURES,
                ngrams=NGRAM_RANGE,
                feature_selector=select_features_chi2,
                random_state=0):
    '''Perform cross-validation on a dataset'''

    fold_data = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    accuracies = []
    cv_data = []
    dois = data['doi'].drop_duplicates().reset_index(drop=True)
    i = 0
    for train_index, test_index in fold_data.split(dois):
        logging.info('Fold ' + str(i))
        i += 1
        # calculate train and test for the current fold
        train_data = data.loc[data['doi'].isin(dois[train_index])]
        test_data = data.loc[data['doi'].isin(dois[test_index])]
        # learn features and IDFs from the fold train and calculate train
        # feature matrix
        count_vectorizer, tfidf_transformer, train_features = \
            feature_fun(train_data['string'], response=train_data['style'],
                        nfeatures=nfeatures, ngrams=ngrams,
                        feature_selector=feature_selector)
        # calculate test feature matrix
        _, _, test_features = feature_fun(test_data['string'],
                                          count_vectorizer=count_vectorizer,
                                          tfidf_transformer=tfidf_transformer)
        # train the main prediction algorithm
        model = algorithm.fit(train_features, train_data['style'])
        # predict styles of the test set
        prediction = model.predict(test_features)
        # update the return data
        accuracies.append(accuracy_score(test_data['style'], prediction))
        cv_data.extend(
            list(
                zip(test_data['doi'], test_data['string'], test_data['style'],
                    prediction)))
    return accuracies, \
        pd.DataFrame(cv_data,
                     columns=['doi', 'string', 'style_true', 'style_pred'])


def evaluate_model(test_data, count_vectorizer, tfidf_transformer, model):
    '''Evaluate a model on tests data'''

    styles = classify(test_data['string'], count_vectorizer, tfidf_transformer,
                      model)
    accuracy = accuracy_score(test_data['style'], styles)
    prediction = test_data.copy()
    prediction['style_pred'] = styles
    prediction.columns = ['doi', 'string', 'style_true', 'style_pred']
    return accuracy, prediction
