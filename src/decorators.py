import functools
from src.utils import *

# Parameters
file_path = LOG_PATH + '/PredictionLogs.txt'


def report(method, dataset, hparams):
    def decorator_report(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            hparams_dict = dict()
            for param in hparams:
                hparams_dict[param] = getattr(self, param)
            f = open(file_path, 'a')
            f.write('----------------------------------------------------------------------\n')
            acc = func(self, *args, **kwargs)
            f.write('Dataset: ' + dataset + '\n')
            f.write('Estimator: ' + method + '\n')
            f.write('Method: {}-fold cross-validation \n'.format(getattr(self, 'cv_k')))
            f.write('Hyperparameters: ' + str(hparams_dict) + '\n')
            f.write('Accuracy: {:.3f} % \n'.format(acc))
            f.close()
            return acc

        return wrapper

    return decorator_report
