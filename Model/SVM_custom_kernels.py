#Hyperparameters
p=13
q=2

#Data processing
data_path = "/Users/antoi/X2A/P3/INF442/Projet2/Datasets/"
data_file = "EUKSIG_13.red.txt"
seq, cleav = read_sequence(data_path + data_file)
cleavpos = return_cleavpos(cleav)
alphabet = return_alphabet(seq)


#solution presented in 3.2 second paragraph
def phi1(word) :
  #getting all necessary dimensions info
  n=p+q
  if self.alphabet is None:
    self.alphabet = return_alphabet(self.seq_list)
  dim = len(self.alphabet)
  d = dict_from_alphabet(self.alphabet)
  
  #encoding the word
  encoding=[0]*(dim*n)
  i=0
  for letter in word :
    encoding[dim*i+d[letter]]=1
  return(encoding)
