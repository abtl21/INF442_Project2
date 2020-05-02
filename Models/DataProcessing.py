from collections import Counter


def read_sequence(datapath):
    protein_sequence = []
    cleavage_site = []

    # Loop condition conveniently discards the description lines
    with open(datapath, 'r') as f:
        while f.readline() is not '':
            # Slicing with :-1 to discard "\n" character
            protein_sequence.append(f.readline()[:-1])
            cleavage_site.append(f.readline()[:-1])

    return protein_sequence, cleavage_site


def return_alphabet(sequence_list):
    # Returns the alphabet present in sequence_list. Useful for dimension minimality.
    alphabet = Counter()
    for seq in sequence_list:
        for letter in seq:
            alphabet[letter] += 1

    alphabet = sorted(list(alphabet))
    return alphabet


def return_cleavpos(cleavage_list):
    # Returns a list with the position of the cleavage point for each sequence in cleavage_list.
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

    return position_list


def all_subsequences(sequence, p, q):
    n = len(sequence)
    subseq_list = []
    i = 0
    while i < n - p - q:
        subseq_list.append(sequence[i:i + p + q])
        i += 1
    return subseq_list


if __name__ == "__main__":
    # Functionality testing
    data_path = "/Users/bernardoveronese/Documents/INF442/INF442_Project2/Datasets/"
    data_file = "EUKSIG_13.red.txt"
    seq, cleav = read_sequence(data_path + data_file)
    arr = return_cleavpos(cleav)
    print(arr)
    alphabet = return_alphabet(seq)
    print(alphabet)
    print(dim)
