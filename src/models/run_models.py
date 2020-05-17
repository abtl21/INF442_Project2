from src.models.Model import Model
from sklearn.svm import SVC
from src.models.PositionScoringMatrix import PosScoringMatrix
from src.models.SVM_standard_kernels import get_kernel_choice
from src.utils import *
from src.features.build_features import *

# Datafile
data_file = "EUKSIG_13.red.txt"

# Hyperparamaters
p = 11
q = 4
C = 1


def choose_estimator(kernel=None, *args, **kwargs):
    est_label = []
    kernel_ = []
    estimator = None
    est_choice = 'Available estimators : \n' \
                 '\t - Position Scoring Matrix (to choose input: PSM) \n' \
                 '\t - Substitution Matrix (to choose input: SM) \n' \
                 '\t - sklearn-SVM (with multiple kernels) (to choose input: SVM) \n' \
                 'Enter estimator of choice: '
    while True:
        try:
            est_label = input(est_choice)
        except ValueError:
            print('Please enter a valid choice of estimator.')
        if est_label == 'PSM':
            estimator = PosScoringMatrix(*args, **kwargs)
            break
        elif est_label == 'SM':
            print('Substitution Matrix method not yet implemented.')
            continue
        elif est_label == 'SVM':
            if kernel is None:
                kernel_ = get_kernel_choice()
            else:
                kernel_ = kernel
            estimator = SVC(kernel=kernel_, *args, **kwargs)
            break
        elif est_label == 'quit':
            return None
        else:
            print('Please enter a valid choice of estimator.')
            continue

    return estimator


if __name__ == '__main__':
    # Hyperparameters
    params = [C]
    paramgrid = {'C': np.logspace(-2, 2, num=5, base=2)}
    paramgrid_poly = {'C': np.logspace(-2, 2, num=5, base=2),
                      'degree': [1, 2, 3, 4, 5]
                      }
    # Optional keyword arguments
    class_weight = {'class_weight': 'balanced'}
    kernel_list = ['linear', 'sigmoid']

    # Defining estimator and model
    for kernel in kernel_list:
        try:
            estimator = SVC(kernel=kernel, class_weight='balanced')
            model = Model(estimator, params)

            # Getting features
            X, Y = get_encoded_features(DATA_PATH + data_file, p, q)

            # Evaluating model
            score = model.evaluate(X, Y)
            if kernel == 'poly':
                search_grid = model.search(X, Y, paramgrid_poly)
            else:
                search_grid = model.search(X, Y, paramgrid)
            print(search_grid.cv_results_)
        except:
            print("Error was raised.")
