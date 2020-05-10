from collections import Counter
import numpy as np


def read_sequence(filepath):
    """
    Return the list of protein sequences and cleavage sites from datapath.

    The data file must strictly follow the following pattern:
    - first line is a description of the sequence
    - second line is the sequence itself
    - third line is a list of chars S, C, M

    Example:
    52 AGP_ECOLI      22 GLUCOSE-1-PHOSPHATASE PRECURSOR (EC 3.1.3.10) (G1PASE).
    MNKTLIAAAVAGIVLLASNAQAQTVPEGYQLQQVLMMSRHNLRAPLANNGSV
    SSSSSSSSSSSSSSSSSSSSSSCMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    """
    protein_sequence = []
    cleavage_site = []

    # Loop condition conveniently discards the description lines
    with open(filepath, 'r') as f:
        while f.readline() is not '':
            # Slicing with :-1 to discard "\n" character
            protein_sequence.append(f.readline()[:-1])
            cleavage_site.append(f.readline()[:-1])

    return protein_sequence, cleavage_site

def get_similarity_matrix(filepath,d) :
    """
    Retruns the similarity matrix as an array
    
    The data file must follow the pattern : 
    -6 lines of comments
    -7th line is a blank space followed by all column entries
    -following lines follow the pattern : the line entry followed by the numbers
    
    Example :
    
    #  Matrix made by matblas from blosum40.iij
    #  * column uses minimum score
    #  BLOSUM Clustered Scoring Matrix in 1/4 Bit Units
    #  Blocks Database = /data/blocks_5.0/blocks.dat
    #  Cluster Percentage: >= 40
    #  Entropy =   0.2851, Expected =  -0.2090
       A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
    A  5 -2 -1 -1 -2  0 -1  1 -2 -1 -2 -1 -1 -3 -2  1  0 -3 -2  0 -1 -1  0 -6 
    R -2  9  0 -1 -3  2 -1 -3  0 -3 -2  3 -1 -2 -3 -1 -2 -2 -1 -2 -1  0 -1 -6 
    
    d is the dictionnary for the equivalence letters<->numbers, which is not guaranteed to be the 'natural one'
    """
    
    M=np.eyes(23,23)
    with open(filepath, 'r') as f:
        #discards description lines
        for i in range (7) :
            f.readline()
        #we must remember the 7th line
        column_entries=f.readline()
        #then fill the matrix
        while f.readline() is not '':
            line=f.readline()
            letter1=line[0]
            for i in range(1,len(line)) :
                M[d[letter1]][d[i]]=line[i]
     return(M)

def return_alphabet(sequence_list):
    # Return the alphabet present in sequence_list. Useful for minimising the dimension for the SVM classifier.
    alphabet = Counter()
    for seq in sequence_list:
        for letter in seq:
            alphabet[letter] += 1

    alphabet = sorted(list(alphabet))
    return alphabet


def return_cleavpos(cleavage_list):
    # Return a list with the position of the cleavage point for each sequence in cleavage_list.
    position_list = [0] * len(cleavage_list)
    cont = 0
    for seq in cleavage_list:

        # Index is found using binary search.
        start = 0
        end = len(seq)
        index = int((end + start) / 2)

        while seq[index] is not 'C':
            if seq[index] == 'S':
                start = index
            else:
                end = index
            index = int((end + start) / 2)

        position_list[cont] = index
        cont += 1

    return np.array(position_list)


def get_features(filepath):
    sequence_list, cleavage_site = read_sequence(filepath)
    cleavage_pos = return_cleavpos(cleavage_site)
    sequence_list = np.array(sequence_list)
    return sequence_list, cleavage_pos


def dict_from_alphabet(alphabet):
    # Return a dictionary encoding a number to each letter in the list alphabet. Alphabet must be a list of char.
    if alphabet is not None:
        dict_alphabet = dict()
        cont = 0
        for letter in alphabet:
            dict_alphabet[letter] = cont
            cont += 1
        return dict_alphabet
    else:
        return None


def seq_list_encoding(sequence_list, cleav_pos, p, q, alphabet):
    """
    Return the list of all possible subsequence of aminoacids of fixed length word_length in a given list of protein
    sequences.

    Return a corresponding list of 1s and -1s depending on whether the cleavage site in included in a
    given subsequence or not.
    """
    encoding_list = []
    cls_cleav_pos = []
    dim = len(alphabet)
    d = dict_from_alphabet(alphabet)
    word_length = p + q

    for cp_iter in range(len(sequence_list)):
        for seq_iter in range(len(sequence_list[cp_iter]) - word_length):
            wordarray = np.zeros(word_length * dim)
            for word_iter in range(word_length):
                index = dim * word_iter + d[sequence_list[cp_iter][seq_iter + word_iter]]
                wordarray[index] = 1
            encoding_list.append(wordarray)
            if cleav_pos[cp_iter] == seq_iter + p:
                cls_cleav_pos.append(1)
            else:
                cls_cleav_pos.append(-1)

    return np.array(encoding_list), np.array(cls_cleav_pos)

def get_encoded_features(filepath, p, q):
    sequence_list, cleav_pos = get_features(filepath)
    alphabet = return_alphabet(sequence_list)
    return seq_list_encoding(sequence_list, cleav_pos, p, q, alphabet)



"""if __name__ == "__main__":
    # Functionality testing
    data_path = "/Users/bernardoveronese/Documents/INF442/INF442_Project2/Datasets/"
    data_file = "EUKSIG_13.red.txt"
    seq, cleav = read_sequence(data_path + data_file)
    arr = return_cleavpos(cleav)
    print(arr)
    alphabet = return_alphabet(seq)
    print(alphabet)
    print(dim)"""
