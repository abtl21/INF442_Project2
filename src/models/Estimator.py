from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, check_cv
from warnings import warn
import abc
import numpy as np

from src.utils import *


def metric_callable(metric):
    if metric == 'accuracy':
        return accuracy_score
    elif metric == 'balanced_accuracy':
        return balanced_accuracy_score
    elif metric == 'roc_auc':
        return roc_auc_score
    elif metric == 'precision':
        return precision_score
    elif metric == 'recall':
        return recall_score
    else:
        if not metric == 'f1':
            warn("Metric {} not available. Switched to default metric = f1_score.".format(metric))

        return f1_score


class Estimator:
    """Super-class for custom estimators."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, p, q, metric='balanced_accuracy', write_log=True):
        self.p = p
        self.q = q
        self.score_metric = metric
        self.write_log = write_log
        self.fitted = False

    @abc.abstractmethod
    def fit(self, X_train, Y_train):
        self.fitted = True

    @abc.abstractmethod
    def predict(self, X_test):
        if not self.fitted:
            raise Exception("Model not yet fitted.")

    @abc.abstractmethod
    def get_test_labels(self, X, Y_raw):
        pass

    def score(self, X_train, Y_raw, cv=5, scoring=METRIC_LIST):
        metric_list = []

        if isinstance(scoring, list):
            metric_list = scoring
        elif isinstance(scoring, str):
            metric_list = [scoring]

        acc = dict().fromkeys(metric_list)
        for key in acc.keys():
            acc[key] = []

        cv_object = check_cv(cv)
        for train_index, test_index in cv_object.split(X_train):
            X_train_batch, X_test_batch = X_train[train_index], X_train[test_index]
            Y_raw_train, Y_raw_test = Y_raw[train_index], Y_raw[test_index]
            self.fit(X_train_batch, Y_raw_train)
            y_pred = self.predict(X_test_batch)
            y_true = self.get_test_labels(X_test_batch, Y_raw_test)
            for metric in acc.keys():
                metric_call = metric_callable(metric)
                acc.get(metric).append(metric_call(y_true, y_pred))

        mean_acc = dict()
        for metric in metric_list:
            mean_acc[metric] = np.mean(acc[metric])

        return mean_acc
