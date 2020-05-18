import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from src.decorators import report_eval
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
        """
        Perform a search over a parameter grid and return statistics on the best estimator's score.

        If the model's custom attribute is set to False, the search is preformed by sklearn's GridSearchCV. Specific
        arguments to this function can be passed via *args and **kwargs.

        If the custom attribute is set to True, the function performs exhaustive search over the parameter grid using the
        estimator's score method at each iteration.

        Parameters
        ----------
        X: list or array-like, size (n_samples, n_features). Format depends on the class's estimator.
            Training data.
        Y: list or array-like, size (n_samples, n_features). Format depends on the class's estimator.
            Target labels for training data.
        param_grid: dictionary {str : list}
            Dictionary of parameters to be considered on search and with a list of their respective values. Note:
            the parameter key must be exactly the parameter's name to be called on estimator.
        scoring: list or str
            Defines one (or more) scoring metrics to be used on the search.
        best_score: str
            The score with which the best estimator will be chosen. Used only if the model's custom attribute is set to
        False.

        Returns
        -------
        A summary of the search results. If the model's custom attribute is set to False, this summary is actually an
        instance of the fitted GridSearchCV class, whose results can be accesed via the cv_results_ attribute. If custom
        is True, a dictionary with the best estimator and their score for each metric in the scoring argument.
        """
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
                # Update estimator's parameters for each iteration
                search_args = dict(zip(list(param_grid.keys()), grid_pair))
                self.estimator.set_params(**search_args)

                # Getting the score for the iteration's parameters
                search_dict = self.estimator.score(X, y, cv=self.cv, scoring=eval_metric)
                if isinstance(eval_metric, list):
                    for metric in eval_metric:
                        # Updating the best parameter if necessary
                        if metric in best_search_dict:
                            if search_dict[metric] > best_search_dict[metric]:
                                best_search_dict[metric] = [search_dict[metric], dict(zip(param_list, grid_pair))]
                        else:
                            best_search_dict[metric] = [search_dict[metric], dict(zip(param_list, grid_pair))]
                else:
                    metric = eval_metric
                    if metric in best_search_dict:
                        # Updating the best parameter if necessary
                        if search_dict[metric] > best_search_dict[metric]:
                            best_search_dict[metric] = [search_dict[metric], dict(zip(param_list, grid_pair))]
                    else:
                        best_search_dict[metric] = [search_dict[metric], dict(zip(param_list, grid_pair))]

            return best_search_dict
