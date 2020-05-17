import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC

from src.models.Estimator import Estimator
from src.models.Model import Model
from src.features.build_features import *
from src.decorators import report_eval
from src.utils import *
from src.visualization.heatmap import *

# Datafile
data_file = "EUKSIG_13.red.txt"

# Hyperparameters
p = 11
q = 4
C = 1
gamma = 10
cv_k = 5


def get_kernel_choice():
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'quit']
    msg_string = "Available kernels: linear, poly, rbf, sigmoid. \n Enter choice of kernel: "
    kernel_ = []
    while True:
        try:
            kernel_ = input(msg_string)
        except ValueError:
            print("Please enter a valid kernel option.")
            continue
        if kernel_ not in kernels:
            print("Please enter a valid kernel option.")
            continue
        else:
            break
    return kernel_


if __name__ == "__main__":
    params = [p, q]
    while True:
        kernel = get_kernel_choice()
        if kernel == 'quit':
            break
        else:
            print("Instantiating model and getting features...")
            estimator = SVC(C, kernel, class_weight='balanced')
            model = Model(estimator, params, cv_k)
            X, Y = get_encoded_features(DATA_PATH + data_file, p, q)
            print("Evaluating model...")
            score = model.evaluate(X, Y)
            print("Finished model evaliation.")
