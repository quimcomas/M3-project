import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pickle as cPickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import itertools
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import confusion_matrix


def histogram_intersection(M, N):
    M_samples, M_features = M.shape
    N_samples, N_features = N.shape

    K_int = np.zeros(shape=(M_samples, N_samples), dtype=np.float)
    for i in range(M_samples):
        for j in range(N_samples):
            K_int[i, j] = np.minimum(M[i, :], N[j, :]).sum()

    return K_int

def build_svm_histogram(train_features, test_features,labels_train, labels_test):


    parameters = [{'kernel': ['precomputed']
                      , 'C': [10, 1, 0.1, 0.2, 0.01, 0.002,0.001]}]

    clf = GridSearchCV(SVC(), parameters, n_jobs=4, cv=5, refit=True, return_train_score=True)

    kernel=histogram_intersection(np.array(train_features),np.array(labels_train))
    clf.fit(kernel,labels_train)


    predict = histogram_intersection(np.array(test_features),np.array( train_features))
    predictions = clf.predict(predict)

    accuracy_test = accuracy_score(labels_test,predictions,normalize=True)
    print('SVM test:')
    print(100*accuracy_test)

    print('Confusion Matrix')

    conf_matrix = confusion_matrix(labels_test, predictions)
    plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion matrix')
    plt.colorbar()
    classes = list(set(labels_test))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
    plt.savefig('conf2'+'.png')
    plt.close()




def build_svm_kernel(kernel,C, gamma, train_features, test_features,labels_train, labels_test):

    stdSlr= StandardScaler().fit(train_features)
    scaled_train= stdSlr.transform(train_features)
    scaled_test = stdSlr.transform(test_features)

    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma).fit(scaled_train, labels_train)

    predicted = clf.predict(scaled_test)

    accuracy = accuracy_score(labels_test,predicted, normalize=True)

    print('SVM test:')
    print(accuracy)

def build_svm_kernel_crossvalidation(train_features, test_features,labels_train, labels_test):


    parameters=[{'kernel':['rbf' ,'linear' ,'poly' ,'sigmoid' ]
                  ,'gamma':[0.01,0.001,0.002,0.003]
                   ,'C': [10,1, 0.1 ,0.2, 0.001, 0.0001]}]

    clf = GridSearchCV( SVC(), parameters,n_jobs=4, cv=5,refit=True,return_train_score=True)

    clf.fit(train_features, labels_train)
    
    results = pd.DataFrame(list(clf.cv_results_['mean_train_score']))
    results.to_csv('result_table.csv', index=False)

    print('Results')
    print(clf.cv_results_)
    print('Best parameters:')
    print(clf.best_params_)
    best_clf=clf.best_estimator_
    print('Best results')
    print(clf.best_score_)

    """clf = SVC(C=clf.best_params_['C'],
              kernel=clf.best_params_['kernel'],
              gamma=clf.best_params_['gamma'])
    clf.fit(train_features, labels_train)"""
   
    # Accuracy test

    #Using best_estimator
    predicted = best_clf.predict(test_features)

    print('MLP+SVM accuracy')
    print(100*accuracy_score(labels_test, predicted, normalize=True))

    conf_matrix = confusion_matrix(labels_test, predicted)
    plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion matrix')
    plt.colorbar()
    classes = list(set(labels_test))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
    plt.savefig('conf1'+'.png')
    plt.close()



