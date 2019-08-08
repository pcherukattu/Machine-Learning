#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from itertools import count
from sklearn.cross_validation import PredefinedSplit
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from time import time
from sklearn.metrics.classification import accuracy_score
clf=SVC(kernel="rbf",C= 10.0)
t0=time()

clf.fit(features_train,labels_train)
print"training time:",round(time()-t0,3),"s"
t1=time()
pred=clf.predict(features_test)
print"predicting time:",round(time()-t1,3),"s"
accuracy=accuracy_score(pred,labels_test)
print"accuracy is ",accuracy
print pred[10]," ",pred[26]," ",pred[50]

count=0
for x in range(len(pred)-1):
    if pred[x]==1:
        count=count+1
     
        
print count
#########################################################


