import numpy as np
import matplotlib.pyplot as plt
from Models.DataProcessing import read_sequence, return_alphabet, return_cleavpos, dict_from_alphabet
from sklearn.model_selection import train_test_split, KFold


def add_smoothing(num, denom, alpha=1):
    return (num + alpha) / (denom + alpha)


class PosScoringMatrix:

    def __init__(self, p, q, seq_list, cleav_pos):

        self.p = p
        self.q = q
        self.seq_list = seq_list
        self.alphabet = return_alphabet(seq_list)
        self.cleav_pos = cleav_pos
        self.score = None

    def fit(self, alpha=1, ignore_first=True):

        d = dict_from_alphabet(self.alphabet)
        dim = len(d)
        N = len(self.seq_list)

        # Filling up matrix with observed frequencies
        score = np.zeros((dim, self.p + self.q))
        for letter in self.alphabet:
            seq_cont = 0
            for seq in self.seq_list:
                if ignore_first:
                    seq = seq[1:]

                seq_len = len(seq)
                cp = self.cleav_pos[seq_cont]
                max_iter = min(self.p + self.q, seq_len + self.p - cp)
                for i in range(max_iter):
                    if seq[cp - self.p + i] == letter:
                        score[d[letter], i] += 1
                seq_cont += 1

        # Additive smoothing
        for a in range(dim):
            for i in range(self.p + self.q):
                score[a, i] = add_smoothing(score[a, i], N, alpha)

        # Computing total frequency of each letter
        total_freq = np.zeros(dim)
        for letter in self.alphabet:
            for seq in self.seq_list:
                for seq_letter in seq:
                    if seq_letter == letter:
                        total_freq[d[letter]] += 1

        total_freq = np.log(total_freq / N)

        # Taking the logarithm
        for a in range(dim):
            for i in range(max_iter):
                score[a, i] = np.log(score[a, i]) - total_freq[a]

        self.score = score

    def word_score(self, sequence, lbound, ubound):

        d = dict_from_alphabet(self.alphabet)
        sum = 0
        cont = 0
        for i in range(lbound, ubound):
            sum += self.score[d[sequence[i]], cont]
            cont += 1

        return sum

    def predict(self, test_batch, treshold, ignore_first=True):

        if self.score is None:
            self.fit(ignore_first)

        cleav_dict = dict()

        seq_cont = 0
        for test_seq in test_batch:
            if ignore_first:
                test_seq = test_seq[1:]

            n = len(test_seq)
            i = 0
            while i < n - self.p - self.q:
                ws = self.word_score(test_seq, i, i + self.p + self.q)
                if ws > treshold:
                    if seq_cont not in cleav_dict:
                        cleav_dict[seq_cont] = i + self.p
                        ws_max = ws
                    elif ws > ws_max:
                        ws_max = ws
                        cleav_dict[seq_cont] = i + self.p
                i += 1
            seq_cont += 1

        return cleav_dict

    def accuracy(self, cleav_test_batch, cleav_predicted_batch):

        accuracy = 0
        for i in range(len(cleav_test_batch)):
            if i in cleav_predicted_batch:
                if cleav_predicted_batch[i] == cleav_test_batch[i]:
                    accuracy += 1
        accuracy /= len(cleav_predicted_batch)
        return 100 * accuracy


if __name__ == "__main__":

    # Hyperparameters
    p = 13
    q = 2
    p_max = 15
    q_max = 15
    alpha = 0.5
    test_size = 0.2
    treshold = -100
    CV_k = 5

    # Data processing
    data_path = "/Users/bernardoveronese/Documents/INF442/INF442_Project2/Datasets/"
    data_file = "EUKSIG_13.red.txt"
    seq, cleav = read_sequence(data_path + data_file)
    seq = np.array(seq)
    cleavpos = return_cleavpos(cleav)

    p_list = np.arange(p_max - 1)
    q_list = np.arange(q_max - 1)
    acc_matrix = np.zeros((p_max - 1, q_max - 1))

    # K-fold cross validation
    for pp in range(p_max - 1):
        for qq in range(q_max - 1):
            acc = []
            kf = KFold(n_splits=CV_k)
            for train_index, test_index in kf.split(seq):
                train_batch, test_batch = seq[train_index], seq[test_index]
                cleavpos_train, cleavpos_test = cleavpos[train_index], cleavpos[test_index]
                psm = PosScoringMatrix(pp + 1, qq + 1, train_batch, cleavpos_train)
                psm.fit(alpha, ignore_first=True)
                pred_cleav = psm.predict(test_batch, treshold, ignore_first=True)
                acc.append(psm.accuracy(cleavpos_test, pred_cleav))
            acc_matrix[pp][qq] = np.mean(acc)

    # Plotting
    print(acc_matrix)
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
    # fig.tight_layout()
    plt.show()

    # train_batch, test_batch, cleav_pos_train, cleav_pos_test = train_test_split(seq, cleavpos, test_size=test_size, random_state=42)

    # Creating model, fitting and predicting
    """psm = PosScoringMatrix(p, q, train_batch, cleav_pos_train)
    psm.fit(alpha=alpha, ignore_first=True)
    predicted_cleav = psm.predict(test_batch, treshold, ignore_first=True)

    # Computing accuracy
    accuracy = psm.accuracy(cleav_pos_test, predicted_cleav)
    print("Accuracy: "+str(accuracy)+"%")"""
    """for i in range(len(cleav_pos_test)):
        if i in predicted_cleav:
            print("("+str(predicted_cleav[i])+","+str(cleav_pos_test[i])+") Result: "+
                  str(predicted_cleav[i]==cleav_pos_test[i]))"""
