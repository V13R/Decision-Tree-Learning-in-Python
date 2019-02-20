import os
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

out = open("Output.txt","w")
#Functia pentru importarea datelor direct de pe internet
def importdata():
    data = pd.read_csv(
#    'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                               sep= ',', 
                               header= None)
    
    print ('Lungime set de date: ', 
           len(data),file=out)
    
    print ('Dimensiuni set de date: ', 
           data.shape,file=out)
    
    print ('Set de date: \n',
           data.head(),file=out)      #Observam proprietati ale datelor de antrenare
    
    return data


#Functia pentru impartirea datelor in setul de antrenare, respectiv setul de test
def split(data):
    X = data.values[:,1:len(data)] #toate liniile, col. 1-final pentru intrari
    Y = data.values[:,0]    #toate liniile, col. 0 pentru clasificare
#Am ales coloanele astfel deoarece baza de date are pe prima pozitie rezultatul dorit.
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size = 0.3,
                                                        random_state = 42) 
                            #random_state este seed-ul ales (acelasi in toate cazurile pt a compara direct)
                            #30% sunt date de test
    
    return X, Y, X_train, X_test, Y_train, Y_test


#Functia pentru antrenare cu giniIndex
def gini_train(X_train,Y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini",
                                      random_state = 42,
                                      max_depth = 3,        #adancimea max. a arborelui
                                      min_samples_leaf = 5) #nr. minim de frunze
    #Antrenam modelul folosind masura de impuritate Gini
    clf_gini.fit(X_train,Y_train)
    
    return clf_gini


def entropy_train(X_train,Y_train):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy",
                                         random_state = 42,
                                         max_depth = 3, 
                                         min_samples_leaf = 5) 
    clf_entropy.fit(X_train,Y_train)
    
    return clf_entropy

def prediction(X_test,model):
    Y_pred = model.predict(X_test)
    print('\nValori prezise: ',
          Y_pred,file=out)
    
    return Y_pred

#Functia care calculeaza precizia estimarii:
def accuracy(Y_test,Y_pred):    
    print('\nMatricea de confuzie: \n',
          confusion_matrix(Y_test,Y_pred),file=out)
    
    print('\nPrecizie: ',
          accuracy_score(Y_test,Y_pred)*100,file=out)
    
    print('\nRaport: \n',
          classification_report(Y_test,Y_pred),file=out) 
    #classification_report ne va afisa informatii detaliate despre precizie - scor F1 etc.
    

def main():
    #Initializarea variabilelor si antrenarea modelelor
    data = importdata()
    X, Y, X_train, X_test, Y_train, Y_test = split(data)
    clf_gini = gini_train(X_train,Y_train)
    clf_entropy = entropy_train(X_train,Y_train)

    
    #Testarea datelor
    print('\n\nRezultate cu Gini Index: ',file=out)
    Y_pred_gini = prediction(X_test,clf_gini)   #predictie pe baza modelului Gini
    accuracy(Y_test,Y_pred_gini)
    
    print('\nRezultate pe baza entropiei: ',file=out)
    Y_pred_entropy = prediction(X_test,clf_entropy)
    accuracy(Y_test,Y_pred_entropy)
    out.close()
    os.startfile("Output.txt")
    
    
if __name__=="__main__": 
    main()     