
'''

Some partially defined functions for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls different functions to perform the required tasks.

'''

import csv
import numpy as np

# Import utilities
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from scipy.stats import uniform as sp_rand

import matplotlib.pyplot as plt
# Import hyperparameters
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV

# Import Classifiers 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9901990, 'Jia Sheng', 'Chong'), (9532897, 'TeeKen', 'Lau'), (9552286, 'Yew Lren', 'Chong') ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    # Step to read the dataset text file
    # Try and catch to capture the runtime and data reading issues
    try:
        with open (dataset_path, 'r') as csvfile:
            lines = csv.reader(csvfile)
            dataset_x = []
            dataset_y = []
            for row in lines:
                dataset_x.append(row[2:])
                dataset_y.append(1 if (row[1] is 'M') else 0)

        return np.asarray(dataset_x).astype(np.float),np.asarray(dataset_y).astype(np.float)
    except RuntimeError as e:
        print(e)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.
    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''    
    model = GaussianNB()
    #model = MultinomialNB()
    #model = BernoulliNB()
    model.fit(X_training, y_training)
    
    clf = model
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    
    # Set the parameters
    parameters = {
            'criterion':['gini', 'entropy'],
            'max_depth': [5, 6, 7, 8, 9, 10],
            'min_samples_leaf': [1, 2, 3],
            'min_samples_split': [4, 5, 6, 7, 8]
            }
    
    # Initialize classifier and fit the training data
    DT = DecisionTreeClassifier()
    clf = GridSearchCV(DT, parameters, cv=6)
    clf.fit(X_training, y_training)


    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''  
    
    # Set parameters
    parameters = {
            "n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9],
            "metric": ["euclidean", "cityblock"],
            "algorithm": ["brute", "ball_tree", "kd_tree"],
            "weights": ["uniform", "distance"]
            }
    
    # Initialize classifier and fit the training data
    KNN = KNeighborsClassifier()
    clf = GridSearchCV(KNN, parameters, cv=3)
    clf.fit(X_training, y_training)
    
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''   
    
    # Set the parameters
    parameters = {
            "gamma": [0.001, 0.0001],
            "C": [100, 1000]
            }

    # Because we know that linear is the better kernel
    # We do not include it within the parameters or
    # It would take a very long time to train   
    
    # Initialize the classifier and fit the training data
    SVM = svm.SVC(
            decision_function_shape='ovo',
            kernel='linear'
            )
    clf = GridSearchCV(SVM, parameters)
    clf.fit(X_training, y_training)

    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=128)
    
    return X_train, X_test, y_train, y_test

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def find_best_svm(X_raw, y_raw):
    gamma_range = range (0, 15)
    svm_scores = []
    best_svm_range = 0
    best_svm_score = 0
    for ga in gamma_range:
        g = pow(10, -ga)
        SVM = svm.SVC(gamma=g)
        score = cross_val_score(SVM,X_raw,y_raw,scoring="accuracy",cv=10).mean()
        if score >= max(svm_scores+[0]):
            best_svm_range = ga
            best_svm_score = score
        svm_scores.append(score)   

    return best_svm_score,best_svm_range,svm_scores,gamma_range

def find_best_decisionTree(X_raw, y_raw):
    decisonTree_range = range (1, 100)
    decisionTree_scores = []
    best_dT_range = 0
    best_dT_score = 0
    for d in decisonTree_range:
        decision_Tree = DecisionTreeClassifier(max_depth=d)
        score = cross_val_score(decision_Tree,X_raw,y_raw,scoring="accuracy",cv=10).mean()
        if score >= max(decisionTree_scores+[0]):
            best_dT_range = d
            best_dT_score = score
        decisionTree_scores.append(score)   

    return best_dT_score,best_dT_range,decisionTree_scores,decisonTree_range

def find_best_nearest_neighbours(X_raw, y_raw):
    knn_range = range(1,100)
    score_list = []
    best_krange = 0
    best_score = 0
    for k in knn_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn,X_raw,y_raw,scoring="accuracy",cv=10).mean()
        if score >= max(score_list+[0]):
            best_krange = k
            best_score = score
        score_list.append(score)    
    return best_score,best_krange,score_list,knn_range
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == "__main__":
    pass
    # call your functions here
    X, y = prepare_dataset('medical_records.data')
    X_training, X_testing, y_training, y_testing = split_dataset(X, y)
    
    # Create classifiers
    NB  = build_NB_classifier(X_training, y_training)
    DT  = build_DT_classifier(X_training, y_training)
    NN  = build_NN_classifier(X_training, y_training)
    SVM = build_SVM_classifier(X_training, y_training)
    
    
    # FInd the best k range for each classifier
    # Nearest Neighbours Classifier
    best_knn_score,best_krange,knn_scores,knn_range = find_best_nearest_neighbours(X,y)
    
    # Decision Tree Classifier
    best_dT_score,best_dT_range,decisionTree_score,decision_range = find_best_decisionTree(X,y)
    
    # SVM Classifier
    best_svm_score,best_svm_range,svm_scores,gamma_range = find_best_svm(X,y)
    
    # Plot the classifier scores to one of their parameter's value
    print()
    print("Decision Tree")
    plt.plot(decision_range,decisionTree_score)
    plt.xlabel("Value d for DT")
    plt.ylabel("Score for d ")
    plt.show()
    print('Best Score {}'.format(best_dT_score))
    print('Best Max Depth Range {}'.format(best_dT_range))
    
    print()
    print("SVM")
    plt.plot(gamma_range,svm_scores)
    plt.xlabel("g, where -g is the power of 10")
    plt.ylabel("Score for g")
    plt.show()
    print('Best Score {}'.format(best_svm_score))
    print('Best Gamma Range {}'.format(pow(10,-best_svm_range)))
    
    print()
    print("KNN")
    plt.plot(knn_range,knn_scores)
    plt.xlabel("Value k for kNN")
    plt.ylabel("Score for k")
    plt.show()
    print('Best Score {}'.format(best_knn_score))
    print('Best knn range {}'.format(best_krange))
    
    # Test predictions
    print('Test accuracy for NB->', NB.score(X_testing, y_testing))
    print('Test accuracy for DT->', DT.score(X_testing, y_testing))
    print('Test accuracy for NN->', NN.score(X_testing, y_testing))
    print('Test accuracy for SVM->', SVM.score(X_testing, y_testing))
