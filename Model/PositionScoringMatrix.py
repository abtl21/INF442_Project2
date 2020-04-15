import numpy as np
from Model.DataProcessing import *


def add_smoothing(num, denom, alpha=1):
    return (num + alpha) / (denom + alpha)


def dict_from_alphabet(alphabet):
    dict_alphabet = dict()
    cont = 0
    for letter in alphabet:
        dict_alphabet[letter] = cont
        cont += 1
    return dict_alphabet


class PosScoringMatrix:

    def __init__(self, p, q, seq_list, cleav_pos, alphabet=None):

        self.p = p
        self.q = q
        self.seq_list = seq_list
        self.alphabet = alphabet
        self.cleav_pos = cleav_pos
        self.score = None

    def score_matrix(self, alpha=1):
        if self.alphabet is None:
            self.alphabet = return_alphabet(self.seq_list)
        dim = len(self.alphabet)
        d = dict_from_alphabet(self.alphabet)

        N = len(self.seq_list)

        # Filling up matrix with observed frequencies
        score = np.zeros((dim, self.p + self.q))
        for letter in self.alphabet:
            seq_cont = 0
            for seq in self.seq_list:
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

        # Taking the logarithm
        for a in range(dim):
            log_total_freq_i = np.log(np.sum(score[a, :]))
            for i in range(max_iter):
                score[a, i] = np.log(score[a, i]) - log_total_freq_i

        self.score = score

    def word_score(self, word):
        if self.alphabet is None:
            self.alphabet = return_alphabet(self.seq_list)
        d = dict_from_alphabet(self.alphabet)

        sum = 0
        cont = 0
        for letter in word:
            sum += self.score[d[letter], cont]
            cont += 1

        return sum

    def fit(self, test_batch, treshold):

        if self.score is None:
            self.score_matrix()

        cleav_dict = dict()

        seq_cont = 0
        for test_seq in test_batch:
            n = len(test_seq)
            i = 0
            while i < n - p - q:
                ws = self.word_score(test_seq[i:i + self.p + self.q])
                if ws > treshold:
                    if seq_cont not in cleav_dict:
                        cleav_dict[seq_cont] = i + p
                        ws_max = ws
                    elif ws > ws_max:
                        ws_max = ws
                        cleav_dict[seq_cont] = i + p
                i += 1
            seq_cont += 1

        return cleav_dict


if __name__ == "__main__":
    #Hyperparameters
    p = 11
    q = 5
    alpha = 0.5
    train_perct = 0.8
    treshold = -40
    #Data processing
    data_path = "/Users/bernardoveronese/Documents/INF442/INF442_Project2/Datasets/"
    data_file = "EUKSIG_13.red.txt"
    seq, cleav = read_sequence(data_path + data_file)
    cleavpos = return_cleavpos(cleav)
    alphabet = return_alphabet(seq)


    train_len = int(0.8*len(seq))

    #Defining sequence batches
    train_batch = seq[:train_len]
    test_batch = seq[train_len:]

    #Defining cleavage indices batches
    cleav_pos_train = cleavpos[:train_len]
    cleav_pos_test = cleavpos[train_len:]

    #Creating model and fitting
    psm = PosScoringMatrix(p, q, train_batch, cleav_pos_train)
    predicted_cleav = psm.fit(test_batch, treshold)

    #Computing accuracy
    accuracy = 0
    for i in range(len(cleav_pos_test)):
        if i in predicted_cleav:
                if predicted_cleav[i] == cleav_pos_test[i]:
                    accuracy += 1

    accuracy /= len(predicted_cleav)
    print(str(100*accuracy)+" %")
    """for i in range(len(cleav_pos_test)):
        if i in predicted_cleav:
            print("("+str(predicted_cleav[i])+","+str(cleav_pos_test[i])+")")"""



