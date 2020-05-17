import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from src.decorators import report_eval
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, \
    f1_score
from warnings import warn
from itertools import product

from src.utils import *


def get_name(estimator):
    custom = True
    name = estimator.__class__.__name__
    if name == 'SVC':
        name += '_{}'.format(estimator.kernel)
        custom = False
    return name, custom


def get_param_dict(name, hyperparams_):
    param_names = []
    if name == 'PosScoringMatrix' or name == 'SVC_linear':
        param_names = ['p', 'q', 'C']
    elif name == 'SVC_rbf' or name == 'SVC_poly' or name == 'SVC_sigmoid':
        param_names = ['p', 'q', 'C', 'gamma']
    else:
        raise ValueError("Invalid class name.")

    while len(hyperparams_) < len(param_names):
        hyperparams_.append("Default")

    return dict(zip(param_names, hyperparams_))


class Model:

    def __init__(self, estimator, hyperparams, cv=5, metric=METRIC_LIST, write_log=True):
        self.estimator = estimator
        self.cv = cv
        self.score_metric = metric
        self.estimator_name, self.custom = get_name(estimator)
        self.param_dict = get_param_dict(self.estimator_name, hyperparams)
        self.write_log = write_log

    # Decorator for writing evaluation results in logs
    @report_eval(dataset="SIG_13")
    def evaluate(self, X, y, scoring=None):
        """
        Return the score of the estimator applied on the feature data X and target labels y.

        Parameters
        ----------
        X : array-like, size (n_samples, n_features)
            Feature training data.
        y : array-like, size (n_samples)
            Target labels for the training data.
        scoring : {'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1'}, list or None, optional
            Metric (or metrics) that will be used for computing the score. Multiple metrics must be passed as a list of
            strings. Default is None. In this case, the class score_metric attribute is used.

        Returns
        -------
        score_dict : dict
            A dictionary with the chosen scoring metrics as strings and their corresponding scores as values.
        """

        score_dict = dict()

        if scoring is None:
            eval_metric = self.score_metric
        else:
            eval_metric = scoring
        if self.custom:
            score_dict = self.estimator.score(X, y, cv=self.cv, scoring=eval_metric)
        else:
            if isinstance(eval_metric, list):
                for metric in eval_metric:
                    score_dict[metric] = cross_val_score(self.estimator, X, y, cv=self.cv, scoring=metric).mean()
            else:
                score_dict[eval_metric] = cross_val_score(self.estimator, X, y, cv=self.cv, scoring=eval_metric).mean()

        return score_dict

    def search(self, X, y, param_grid, scoring=None, best_score="f1", *args, **kwargs):
        param_list = list(param_grid.keys())

        # Defining scoring
        if scoring is None:
            eval_metric = self.score_metric
        else:
            eval_metric = scoring

        # Different methods for custom or built-in estimators
        if not self.custom:
            # Calling GridSearchCV on the estimator, parameters
            gridresults = GridSearchCV(self.estimator, param_grid, scoring=eval_metric, cv=self.cv,
                                       refit=best_score, *args, **kwargs)
            gridresults.fit(X, y)
            return gridresults

        if self.custom:
            # Creating search dictionary that will store the best results for each metric
            best_search_dict = dict()

            # Grid search
            for grid_pair in product(*list(param_grid.values())):
                # Update estimator's parameteres for each iteration
                search_args = dict(zip(list(param_grid.keys()), grid_pair))
                self.estimator.set_params(**search_args)

                # Getting the score for the iteration's parameters
                search_dict = self.estimator.score(X, y, cv=self.cv, scoring=eval_metric)
                if isinstance(eval_metric, list):
                    for metric in eval_metric:
                        if metric in best_search_dict:
                            if search_dict[metric] > best_search_dict[metric]:
                                best_search_dict[metric] = [search_dict[metric], dict(zip(param_list, grid_pair))]
                        else:
                            best_search_dict[metric] = [search_dict[metric], dict(zip(param_list, grid_pair))]
                else:
                    metric = eval_metric
                    if metric in best_search_dict:
                        if search_dict[metric] > best_search_dict[metric]:
                            best_search_dict[metric] = [search_dict[metric], dict(zip(param_list, grid_pair))]
                    else:
                        best_search_dict[metric] = [search_dict[metric], dict(zip(param_list, grid_pair))]

            return best_search_dict
