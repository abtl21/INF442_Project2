import functools
from src.utils import *

# Parameters
file_path = LOG_PATH + '/PredictionLogs.txt'


def report_eval(dataset):
    def decorator_report(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            acc = func(self, *args, **kwargs)
            if getattr(self, 'write_log'):
                hparam_dict = getattr(self, 'param_dict')
                f = open(file_path, 'a')
                f.write('----------------------------------------------------------------------\n')
                f.write('Dataset: ' + dataset + '\n')
                f.write('Estimator: ' + getattr(self, 'estimator_name') + '\n')
                f.write('Model assessment method: {}-fold cross-validation \n'.format(getattr(self, 'cv')))
                f.write('Hyperparameters: \n')
                for key in hparam_dict.keys():
                    f.write('\t {} = {}\n'.format(key, hparam_dict[key]))
                f.write('Score summary: \n')
                for key in acc.keys():
                    f.write('\t {} : {:.3f} %\n'.format(key, 100*acc[key]))
                f.close()
            return acc

        return wrapper

    return decorator_report
