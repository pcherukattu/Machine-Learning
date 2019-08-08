#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from sklearn.metrics.scorer import recall_scorer
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
print data
labels, features = targetFeatureSplit(data)
print features
print "labels ",labels



### your code goes here 
from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(features,labels)
print classifier.score(features,labels)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split( features,labels, test_size=0.3, random_state=42)
classifier.fit(X_train,y_train)
pred=classifier.predict(X_test,y_test)
print "predication ",pred
print "length ", len(y_test)
k=0

no_poi_test=0
for k in range(len(y_test)):
    if y_test[k]==1.0:
        no_poi_test=no_poi_test+1
print "no_poi_test  ",no_poi_test         
true_positive=0        
for k in range(len(y_test)):
    if pred[k]==1.:
        if float(pred[k])==y_test[k]:
            true_positive=true_positive+1
        
    
print "True Positive ",true_positive  
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "Precision Score ",precision_score(y_test,pred)
print "Recall Score ",recall_score(y_test,pred)              
        
      
print classifier.score(X_test,y_test)

print no_poi_test


