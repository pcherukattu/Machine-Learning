#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
from audioop import reverse
from sklearn.preprocessing.data import MinMaxScaler
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


from sklearn.preprocessing import MinMaxScaler
import numpy
### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0) 
feature_2 ="exercised_stock_options"
feature_1 = "salary"
poi  = "poi"
max_exercised_stock_options=0
min_exercised_stock_options=float("Inf")
for key in data_dict:
    if data_dict[key]["exercised_stock_options"]> 0 and data_dict[key]["exercised_stock_options"]!="NaN":
        if data_dict[key]["exercised_stock_options"] > max_exercised_stock_options:
            max_exercised_stock_options=data_dict[key]["exercised_stock_options"]
        if data_dict[key]["exercised_stock_options"]< min_exercised_stock_options:
            min_exercised_stock_options=data_dict[key]["exercised_stock_options"]
            
               
print min_exercised_stock_options, max_exercised_stock_options         


features_list = [poi,feature_2,feature_1]
data=featureFormat(data_dict, features_list,remove_any_zeroes=True)
poi, finance_features = targetFeatureSplit( data )
finance_features=numpy.reshape(numpy.array(finance_features),(len(finance_features),2))

##finance_features=sorted(finance_features,key=lambda x:x[0],reverse=True)
print finance_features
limit=len(finance_features)
scaler=MinMaxScaler()
###salaries=numpy.array([finance_features[0],[1000000.],finance_features[limit-1]])

rescaled_weight=scaler.fit_transform(finance_features)
print "The rescaled weight",rescaled_weight


data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3="total_payments"
poi  = "poi"
features_list = [poi, feature_1,feature_2,feature_3]
data = featureFormat(data_dict, features_list, remove_any_zeroes=True)
poi, finance_features = targetFeatureSplit( data )
print "Last ",finance_features





### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 ,f3 in finance_features:
    plt.scatter( f1, f2,f3 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300)
pred=kmeans.fit_predict(finance_features)




### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters2.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
