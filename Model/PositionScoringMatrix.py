from DataProcessing import *
import numpy as np

def add_smoothing(num, denom, alpha=1):
    return (num + alpha)/(denom + alpha)

def dict_from_alphabet(alphabet):
    dict_alphabet = dict()
    cont = 0
    for letter in alphabet:
        dict_alphabet[letter] = cont
        cont += 1
    return dict_alphabet

class PosScoringMatrix():

    def __init__(self, p, q, seq_list, alphabet=None, cleav_pos=None):
        self.p = p
        self.q = q
        self.seq_list = seq_list
        self.alphabet = alphabet
        self.cleav_pos = cleav_pos

    def score_matrix(self, alpha=1):
        if self.alphabet is None:
            self.alphabet, _ = return_alphabet(self.seq_list)
        dim = len(self.alphabet)
        d = dict_from_alphabet(self.alphabet)
        if self.cleav_pos is None:
            #TODO
            pass

        N = len(self.seq_list)

        #Filling up matrix with observed frequencies
        score = np.zeros((dim, self.p + self.q))
        for letter in self.alphabet:
            seq_cont = 0
            for seq in self.seq_list:
                seq_len = len(seq)
                cp = self.cleav_pos[seq_cont]
                max_iter = max(self.p + self.q, seq_len + self.p - cp)
                for i in range(max_iter):
                    if seq[cp - self.p + i] == letter:
                        score[d[letter], i] += 1
                seq_cont += 1

        #Additive smoothing
        for a in range(dim):
            for i in range(self.p + self.q):
                score[a, i] = add_smoothing(score[a, i], N, alpha)

        #Taking the logarithm
        for a in range(dim):
            log_total_freq_i = np.log(np.sum(score[a, :]))
            for i in range(self.p + self.q):
                score[a, i] = np.log(score[i, a]) - log_total_freq_i

        return score








