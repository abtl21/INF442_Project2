import numpy as np
import matplotlib.pyplot as plt
from src.features.build_features import *
from sklearn.model_selection import train_test_split, KFold
from src.decorators import report
from src.utils import *

METHOD = 'Position Scoring Matrix'
PARAM_LIST = ('p', 'q', 'C')


# Hyperparameters
p_max = 5
q_max = 5
alpha = 0.5
C = -100
cv_k = 5


def add_smoothing(num, denom, alpha=1):
    return (num + alpha) / (denom + alpha)


def psm_accuracy(cleav_test_batch, cleav_predicted_batch):
    accuracy = 0
    for i in range(len(cleav_test_batch)):
        if i in cleav_predicted_batch:
            if cleav_predicted_batch[i] == cleav_test_batch[i]:
                accuracy += 1
    accuracy /= len(cleav_predicted_batch)
    return 100 * accuracy


class PosScoringMatrix:

    def __init__(self, p_, q_, C_, cv_k_=5, alpha_=0.5, ignore_first_=True):

        self.p = p_
        self.q = q_
        self.C = C_
        self.cv_k = cv_k_
        self.alpha = alpha_
        self.ignore_first = ignore_first_
        self.alphabet = None
        self.d = None
        self.score_matrix = None

    def get_attributes(self):
        return vars(self)

    def word_score(self, sequence, lbound, ubound):

        score_sum = 0
        cont = 0
        for i in range(lbound, ubound):
            score_sum += self.score_matrix[self.d[sequence[i]], cont]
            cont += 1

        return score_sum

    def fit(self, seq_train_batch, cleav_train_batch):

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

        if self.score is None:
            raise ValueError("Must execute fit method first.")

        cleav_dict = dict()

        seq_cont = 0
        for test_seq in seq_test_batch:
            if self.ignore_first:
                test_seq = test_seq[1:]

            n = len(test_seq)
            i = 0
            while i < n - self.p - self.q:
                ws = self.word_score(test_seq, i, i + self.p + self.q)
                if ws > self.C:
                    if seq_cont not in cleav_dict:
                        cleav_dict[seq_cont] = i + self.p
                        ws_max = ws
                    elif ws > ws_max:
                        ws_max = ws
                        cleav_dict[seq_cont] = i + self.p
                i += 1
            seq_cont += 1

        return cleav_dict

    @report(method=METHOD, dataset="EUKSIG_13", hparams=PARAM_LIST)
    def score(self, seq_list_, cleavpos_):
        acc = []
        kf = KFold(n_splits=self.cv_k)
        for train_index, test_index in kf.split(seq_list_):
            train_batch_, test_batch_ = seq_list_[train_index], seq_list_[test_index]
            cleavpos_train_, cleavpos_test_ = cleavpos_[train_index], cleavpos_[test_index]
            self.fit(train_batch_, cleavpos_train_)
            pred_cleav_ = self.predict(test_batch_)
            acc.append(psm_accuracy(cleavpos_test_, pred_cleav_))

        return np.mean(acc)


if __name__ == "__main__":

    # Data processing
    data_file = "EUKSIG_13.red.txt"
    seq_list, cleav = read_sequence(DATA_PATH + data_file)
    seq_list = np.array(seq_list)
    cleavpos = return_cleavpos(cleav)

    p_list = np.arange(p_max - 1)
    q_list = np.arange(q_max - 1)
    acc_matrix = np.zeros((p_max - 1, q_max - 1))

    # K-fold cross validation
    for pp in range(p_max - 1):
        for qq in range(q_max - 1):
            psm = PosScoringMatrix(pp + 1, qq + 1, C, cv_k)
            acc_matrix[pp][qq] = psm.score(seq_list, cleavpos)

    # Plotting
    fig, ax = plt.subplots()
    im = ax.imshow(acc_matrix)
    ax.set_xticks(p_list)
    ax.set_yticks(q_list)
    ax.set_xticklabels(p_list + 1)
    ax.set_yticklabels(q_list + 1)
    ax.set_ylim(len(q_list) - 0.5, -0.5)
    for i in range(q_max - 1):
        for j in range(p_max - 1):
            text = ax.text(j, i, "{:.2f}".format(acc_matrix[i, j]),
                           ha="center", va="center", color="w")
    ax.set_title("Accuracy heatmap")
    fig.tight_layout()
    plt.show()

