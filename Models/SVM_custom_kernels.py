import numpy as np
import matplotlib.pyplot as plt
from Models.DataProcessing import read_sequence, return_alphabet, return_cleavpos, dict_from_alphabet
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC, LinearSVC

# Hyperparameters
p = 13
q = 2
CV_k = 5

# Data processing
data_path = "/Users/bernardoveronese/Documents/INF442/INF442_Project2/Datasets/"
data_file = "EUKSIG_13.red.txt"
seq, cleav = read_sequence(data_path + data_file)
cleavpos = return_cleavpos(cleav)
alphabet = return_alphabet(seq)


# solution presented in 3.2 second paragraph
def phi1(word):
    # getting all necessary dimensions info
    n = p + q
    if self.alphabet is None:
        self.alphabet = return_alphabet(self.seq_list)
    dim = len(self.alphabet)
    d = dict_from_alphabet(self.alphabet)

    # encoding the word
    encoding = [0] * (dim * n)
    i = 0
    for letter in word:
        encoding[dim * i + d[letter]] = 1
    return (encoding)


def seq_list_encoding(sequence_list, word_length, cleav_pos, alphabet):
    encoding_list = []
    cls_cleav_pos = []
    dim = len(alphabet)
    d = dict_from_alphabet(alphabet)

    for cp_iter in range(len(sequence_list)):
        for seq_iter in range(len(sequence_list[cp_iter]) - word_length):
            wordarray = np.zeros(word_length * dim)
            for word_iter in range(word_length):
                index = dim * word_iter + d[sequence_list[cp_iter][seq_iter + word_iter]]
                wordarray[index] = 1
            encoding_list.append(wordarray)
            if seq_iter <= cleav_pos[cp_iter] <= seq_iter + word_length:
                cls_cleav_pos.append(1)
            else:
                cls_cleav_pos.append(-1)

    return np.array(encoding_list), np.array(cls_cleav_pos)

if __name__ == "__main__":
    X, Y = seq_list_encoding(seq, p + q, cleavpos, alphabet)
    lsvm = LinearSVC(class_weight='balanced')
    csv_score = cross_val_score(lsvm, X, Y, cv=CV_k)
    print(csv_score)



