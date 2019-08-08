#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
##from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.decomposition.pca import RandomizedPCA
from scipy.cluster.vq import whiten
from sklearn.metrics.scorer import accuracy_scorer
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','expenses','total_payments','total_stock_value','restricted_stock'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

data_dict.pop("TOTAL",0)
my_dataset = data_dict
print my_dataset

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.pipeline import make_pipeline,FeatureUnion
from sklearn.feature_selection import SelectKBest,f_classif
##from sklearn.decomposition import PCA

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics.classification import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
clf = svm.SVC(kernel='rbf',C=10000.0)
clf=GaussianNB()
pca=RandomizedPCA(n_components=4)
filter=SelectKBest(f_classif,k=4)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
pipe_clf=make_pipeline(pca,filter,clf)
pipe_clf.fit(features_train,labels_train)
print pipe_clf
features_test=pca.transform(features_test)
features_test=filter.transform(features_test)   
##combined_features=FeatureUnion([('pca',pca),('selectkbest',filter)])
##features_test=combined_features.fit(features_test,labels_test).transform(features_test)
print features_test
print filter.get_support()
prediction=clf.predict(features_test)
print "f1_score is ",f1_score(labels_test,prediction,average='macro')

##clf.fit(features_train,labels_train)
##pred=clf.predict(features_test)
##accuracy_new=accuracy_score(pred,labels_test)
##print accuracy_new
##print pipe_clf.named_steps["pca"].components_


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)