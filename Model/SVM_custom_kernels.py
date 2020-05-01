from Model.DataProcessing import *
import numpy as np
from sklearn import svm


#Hyperparameters
p=13
q=2
n=p+q
gamma=1 #bandwidth for rbf
train_perct=0.8
def resumeparametres():
  return("p :"+p+", q :"+q+", gamma :"+gamma)

#Data processing
data_path = "/Users/antoi/X2A/P3/INF442/Projet2/Datasets/"
data_file = "EUKSIG_13.red.txt"
seq, cleav = read_sequence(data_path + data_file)
alphabet = return_alphabet(seq)
print("data processed")

#Creating all n-sized subsequences with labels, by 'sliding the window' to have a classification problem
d = dict_from_alphabet(alphabet)

alllongseq=[]
for sequence in seq :
  sequence_encoded=[]
  for letter in sequence :
    sequence_encoded.append(d[letter])
  allseq.append(sequence_encoded)
  
x=[]
y=[]
cont=0
for sequence in alllongseq :
  for i in range(len(sequence)-n) :
    X.append(sequence[i:i+n])
    if (cleav_pos[cont]==i+p) :
      Y.append(1)
    else :
      Y.append(0)
  cont +=1

X=np.array(x)
Y=np.array(y)

print("data augmented (created all n-sized subsequences with labels, by sliding the window)")
                                
#Defining batches
train_len = int(train_perct*len(Y))

train_batch = X[:train_len]
test_batch = X[train_len:]
cleav_pos_train = Y[:train_len]
cleav_pos_test = Y[train_len:]


########################Similarity kernel

def K1(u,v):
  count = 0
  for i in range(n) :
    if (u[i]==v[i]) :
      count+=1

#fitting 
clf=svm.SVC(kernel=K1)
clf.fit(train_batch,cleav_pos_train)
print("fitting done...")

#computing accuracy
print("predicting...")
accuracy=0
for sequence in test_batch :
  predict=clf.predict(sequence)[0][0]
  if (predict==cleav_pos_test) :
    accuracy+=1
accuracy /= len(predicted_cleav)
print("Accuracy computed with "+resumeparametres()+" with similarity kernel")
print(str(100*accuracy)+" %")

########################Substitution matrix

#Importer M ?????

#Score
def s(a,b) :
  n=p+q
  sum=0
  for i in range(n) :
    sum+=M(a[i],b[i])
  return(sum)

def K2(a,b,gamma) :
  return (exp(-gamma*s(a,b)))

#RBF Kernel

rbf=svm.SVC(kernel=K2)
rbf.fit(train_batch,cleav_pos_train)
print("fitting done...")
  
print("predicting...")
accuracy=0
for sequence in test_batch :
  predict=clf.predict(sequence)[0][0]
  if (predict==cleav_pos_test) :
    accuracy+=1
accuracy /= len(predicted_cleav)
print("Accuracy computed with "+resumeparametres()+" with substitution matrix")
print(str(100*accuracy)+" %")   
