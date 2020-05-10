import numpy as np
import matplotlib.pyplot as plt
from src.features.build_features import *
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC, LinearSVC

# Hyperparameters
CV_k = 5
max_iter = 5

# Data processing
data_path = "/Users/bernardoveronese/Documents/INF442/INF442_Project2/Datasets/"
data_file = "EUKSIG_13.red.txt"
seq, cleav = read_sequence(data_path + data_file)
cleavpos = return_cleavpos(cleav)
alphabet = return_alphabet(seq)
d = dict_from_alphabet(alphabet)


################################################Similarity kernel


#solution presented in 3.2 second paragraph
def K1(u,v):
def phi1(word) :
  count = 0
  #getting all necessary dimensions info
  for i in range(n) :
    if (u[i]==v[i]) :
      count+=1

def PredictionSimilarityKernel(train_data,train_labels,test_data,test_labels):
    #trains a SVM with train_data labelled with train_labels, tests on test_data and computes accuracy
    #fitting
    print("fitting...")
    clf=svm.SVC(kernel=K1)
    clf.fit(train_data,train_labels)
    print("OK")

    #computing accuracy
    print("predicting...")
    accuracy=0
    cont=0
    for sequence in test_data :
      predict=clf.predict(sequence)[0][0]
      if (predict==test_labels[cont]) :
        accuracy+=1
      cont++
    accuracy /= len(test_labels)
    print("Accuracy computed with similarity kernel")
    print(str(100*accuracy)+" %")

########################Substitution matrix

#hyperparameters : substitution matrix and bandwidth

path="" #change it to your convenience to choose one of the matrices
M=get_similarity_matrix(path,d)

gamma=1

#Score
def s(a,b) :
  n=p+q
  for i in range(n) :
    sum+=M[a[i]][b[i]]
  return(sum)

def K2(a,b) :
  return (exp(-gamma*s(a,b)))

def PredictionSimiliarityKernel(train_data,train_labels,test_data,test_labels):
    #trains a SVM with train_data labelled with train_labels, tests on test_data and computes accuracy
    print("fitting ...")
    rbf=svm.SVC(kernel=K2)
    rbf.fit(train_data,train_labels)
    print("OK")

    print("predicting...")
    accuracy=0
    cont=0
    for sequence in test_data :
      predict=clf.predict(sequence)[0][0]
      if (predict==test_labels[cont]) :
        accuracy+=1
      cont+=1
    accuracy /= len(predicted_cleav)
    print("Accuracy computed with substitution matrix")
    print(str(100*accuracy)+" %")

if __name__ == "__main__":


    """Csearch = np.logspace(-3, -1, num=5, base=10.0)
    print(Csearch)
    Csearch_dict = {'C': np.logspace(-1, 1, base=10)}"""
    help(seq_list_encoding)
    #clf = GridSearchCV(lsvc, param_grid=Csearch, refit=True)
    #clf.fit(X, Y)
    #print(clf.cv_results_)
    for iter in range(max_iter):
        p = np.random.randint(1, 15)
        q = np.random.randint(1, 15)
        X, Y = seq_list_encoding(seq, p + q, cleavpos, alphabet)
        lsvc = LinearSVC(C=0.01, class_weight='balanced', max_iter=5000)
        csv_score = cross_val_score(lsvc, X, Y, cv=CV_k, scoring='accuracy')
        csv_score_balanced = cross_val_score(lsvc, X, Y, cv=CV_k, scoring='balanced_accuracy')
        csv_score_roc = cross_val_score(lsvc, X, Y, cv=CV_k, scoring='roc_auc')
        #print("(p, q) = (", p, q, ") Score (method = accuracy) : {:.4f}".format(np.mean(csv_score)))
        print("(p, q) = (", p, q, ") Score (method = balanced accuracy) : {:.4f}".format(np.mean(csv_score_balanced)))
        #print("(p, q) = (", p, q, ") Score (method = roc_auc) : {:.4f}".format(np.mean(csv_score_roc)))



