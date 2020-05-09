import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from src.models.Estimator import Estimator
from src.features.build_features import *
from src.models.Model import Model
from src.utils import *
from src.visualization.heatmap import *

# Hyperparameters
p_max = 13
q_max = 13
p = 11
q = 4
alpha = 0.5
C = -58
cv = 5


def add_smoothing(num, denom, alpha=1.):
    return (num + alpha) / (denom + alpha)


def psm_accuracy(cleav_test_batch, cleav_predicted_batch):
    accuracy = 0
    if len(cleav_predicted_batch) > 0:
        for i in range(len(cleav_test_batch)):
            if i in cleav_predicted_batch:
                for prev in cleav_predicted_batch[i]:
                    if prev == cleav_test_batch[i]:
                        accuracy += 1
        accuracy /= len(cleav_predicted_batch)
    return 100 * accuracy


class PosScoringMatrix(Estimator):

    def __init__(self, p, q, C, alpha=0.5, ignore_first=True, metric='balanced_accuracy', write_log=True):

        super().__init__(p, q, metric, write_log)
        self.C = C
        self.alpha = alpha
        self.ignore_first = ignore_first
        self.alphabet = None
        self.d = None
        self.score_matrix = None

    def word_score(self, sequence, lbound, ubound):

        score_sum = 0
        cont = 0
        for i in range(lbound, ubound):
            score_sum += self.score_matrix[self.d[sequence[i]], cont]
            cont += 1

        return score_sum

    def get_test_labels(self, seq_test_batch, cleav_test_batch, crit='exact'):

        exact_comp = lambda target, start: (start + self.p) == target
        neighbourhood_comp = lambda target, start: start <= target <= (start + self.p + self.q)

        if crit == 'neighbour':
            comp = neighbourhood_comp
        else:
            comp = exact_comp

        test_labels = []

        seq_cont = 0
        for test_seq in seq_test_batch:
            if self.ignore_first:
                test_seq = test_seq[1:]

            n = len(test_seq)
            i = 0
            while i < n - self.p - self.q:
                if comp(cleav_test_batch[seq_cont], i):
                    test_labels.append(1)
                else:
                    test_labels.append(-1)
                i += 1
            seq_cont += 1

        return test_labels

    def fit(self, seq_train_batch, cleav_train_batch):
        super().fit(seq_train_batch, cleav_train_batch)

        if self.alphabet is None:
            self.alphabet = return_alphabet(seq_train_batch)

        if self.d is None:
            self.d = dict_from_alphabet(self.alphabet)

        dim = len(self.d)
        N = len(seq_train_batch)

        # Filling up matrix with observed frequencies
        score = np.zeros((dim, self.p + self.q))
        for letter in self.alphabet:
            seq_cont = 0
            for seq in seq_train_batch:
                if self.ignore_first:
                    seq = seq[1:]

                seq_len = len(seq)
                cp = cleav_train_batch[seq_cont]
                max_iter = min(self.p + self.q, seq_len + self.p - cp)
                for i in range(max_iter):
                    if seq[cp - self.p + i] == letter:
                        score[self.d[letter], i] += 1
                seq_cont += 1

        # Additive smoothing
        for a in range(dim):
            for i in range(self.p + self.q):
                score[a, i] = add_smoothing(score[a, i], N, self.alpha)

        # Computing total frequency of each letter
        total_freq = np.zeros(dim)
        for letter in self.alphabet:
            for seq in seq_train_batch:
                for seq_letter in seq:
                    if seq_letter == letter:
                        total_freq[self.d[letter]] += 1

        total_freq = np.log(total_freq / N)

        # Taking the logarithm
        for a in range(dim):
            for i in range(self.p + self.q):
                score[a, i] = np.log(score[a, i]) - total_freq[a]

        self.score_matrix = score

    def predict(self, seq_test_batch):

        super().predict(seq_test_batch)

        cleav_predictor = []

        seq_cont = 0
        for test_seq in seq_test_batch:
            if self.ignore_first:
                test_seq = test_seq[1:]

            n = len(test_seq)
            i = 0
            while i < n - self.p - self.q:
                ws = self.word_score(test_seq, i, i + self.p + self.q)
                if ws > self.C:
                    cleav_predictor.append(1)
                else:
                    cleav_predictor.append(-1)
                i += 1
            seq_cont += 1

        return cleav_predictor

    """def scorer(self, seq_list_, cleavpos_, scoring='f1'):
        acc = []

        # Cross-validation loop
        for train_index, test_index in self.cv.split(seq_list_):
            train_batch_, test_batch_ = seq_list_[train_index], seq_list_[test_index]
            cleavpos_train_, cleavpos_test_ = cleavpos_[train_index], cleavpos_[test_index]
            self.fit(train_batch_, cleavpos_train_)
            y_pred = self.predict(test_batch_)
            y_true = self.get_test_labels(test_batch_, cleavpos_test_)
            acc.append(self.score_metrics(y_true, y_pred))

        return np.mean(acc)"""

    def score(self, seq_list, cleavpos, cv=5, scoring=METRIC_LIST):
        return super().score(seq_list, cleavpos, cv=cv, scoring=scoring)


if __name__ == "__main__":
    # Data processing
    data_file = "EUKSIG_13.red.txt"
    seq_list, cleavpos = get_features(DATA_PATH + data_file)

    p_list = np.arange(p_max - 1)
    q_list = np.arange(q_max - 1)
    acc_matrix = np.zeros((p_max - 1, q_max - 1))

    # Model training and assessment
    params = [p, q, C]
    estimator = PosScoringMatrix(p, q, C, alpha)
    model = Model(estimator, params, cv)
    X, Y = get_features(DATA_PATH + data_file)
    score = model.evaluate(X, Y)

    # Plotting
    """fig, ax = plt.subplots()

    im, cbar = heatmap(acc_matrix, p_list, q_list, ax=ax,
                       cmap="YlGn", cbarlabel="Accuracy heatmap (%)")
    texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    fig.tight_layout()
    plt.show()"""
