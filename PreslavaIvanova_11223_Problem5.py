# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:51:59 2019

@author: Preslava Ivanova, 11223
"""
#PART I. - Importing data
import matplotlib.pyplot as plt
import pandas

from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors 

#Looking at the data
data = pandas.read_csv("/Users/pivanova/Documents/Projects/PreslavaIvanova_11223_zad5/Problem 5/DiamGT.csv",delimiter=',',header=0)
print(data.head())
print(data.dtypes) #quantity variables are depth, table, price,x,y,z

#Defining the main data set
X = data.drop(['TVarGamma','Unnamed: 0'],axis=1) #removing columns we don't need
y = data.values[:,8]
print(X)
print(y)

#Standerdizing data
from sklearn.preprocessing import StandardScaler
X_new=StandardScaler().fit_transform(X)

#Encoding target variable
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y=label_encoder.fit_transform(y).astype('int')

#PART II. Applying KNN method for classification
# 2.1 - First method
myList = list(range(1,20))
neighbors = list(filter(lambda x: x % 2 != 0, myList))

#Creating empty lists to hold score values
cv_scores = []
cv_k=[]
cv_train_ind = []
cv_test_ind = []

#Defining the function using k-Fold
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=False)
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_k.append(k)
    sm=0
    for train_index, test_index in kf.split(X):
        knn.fit(X.iloc[train_index],y[train_index])
        sc_test  = knn.score(X.iloc[test_index],y[test_index])
        y_pred = knn.predict(X.iloc[test_index])
        if sm < sc_test :
            sm=sc_test
            train_maxindex = train_index
            test_maxindex =  test_index
    
    cv_train_ind.append(train_maxindex)
    cv_test_ind.append(test_maxindex)
    scores = knn.score(X, y) 
    cv_scores.append(scores)


#Definng missclassification error
MSE = [1 - x for x in cv_scores]

#Finding optimal values for k
optimal_k = neighbors[MSE.index(min(MSE))]
optimal_MSE = cv_scores[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
print("The optimal value of MSE        is %6.3f" % optimal_MSE)
#The optimal number of neighbors is 1
#The optimal value of MSE        is  0.59

#Testing the model with optimal k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X.iloc[cv_train_ind[MSE.index(min(MSE))]],y[cv_train_ind[MSE.index(min(MSE))]])

#Printing the score and a confusion matrix
y_p = knn.predict(X)
print('knn.score for the full set', knn.score(X, y))
from sklearn.metrics import confusion_matrix
print('confussion matrix for the full set',confusion_matrix(y, y_p))
#('knn.score for the full set', 0.9304783092324805)

#2.2. - Second method
from sklearn.model_selection import train_test_split
#Splitting the data set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3) 

#Fitting the model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
train_score = knn.score(X_train, y_train)
test_score = knn.score(X_test, y_test)
print('knn test_score =',test_score)
print('knn train_score =',train_score)
#('knn test_score=', 0.35156346557903845)
#('knn train_score=', 0.4738598442714127)

#Printing a confusion matrix - predicting values for the full data set
from sklearn.metrics import confusion_matrix
knn_pred = knn.predict(X)
print ('\n confussion matrix for the test set :\n',confusion_matrix(y,knn_pred))
print ('knn.score for the full set :', knn.score(X, y))
#('knn.score for the full set :', 0.4371709306637004)
#The model needs improvement, because of the low values

#Using kFold to improve the score
kf = KFold(n_splits=10, shuffle=False)
k=0
sm=0
for train_index, test_index in kf.split(X):
    #print('k=',k)
    knn.fit(X.iloc[train_index],y[train_index])
    score_test = knn.score(X.iloc[test_index], y[test_index])
    print('score_train =',knn.score(X.iloc[train_index], y[train_index])) 
    print('score_test =',score_test)
    if sm < score_test : 
        print('k=',k)
        sm=score_test
        train_minindex = train_index
        test_minindex =  test_index
        
    k+=1
    print

#Finding the highest score
knn.fit(X.iloc[train_minindex],y[train_minindex])
knn_predkf = knn.predict(X.iloc[test_minindex])
print('SCORE on test set: ',knn.score(X.iloc[test_minindex],
                                         y[test_minindex]))
#('SCORE on test set: ', 0.3350018539117538)

#Score of the full set
knn_scorekf = knn.score(X, y)
print('SCORE on full set: ',knn.score(X,y) ) 
#('SCORE on full set: ', 0.46375602521319986)

#PART III. Using Logistic regression with PCA
from sklearn.decomposition import PCA
target_names = data.TVarGamma.unique()
pca = PCA(n_components=2) 
X_r = pca.fit(X).transform(X) 

# Percentage of variance explained for each components 
print('explained variance ratio (first two components): %s' 
% str(pca.explained_variance_ratio_)) 
#[9.99999514e-01 3.27544496e-07 1.12004725e-07] - first 2 components explain
# just a little part of the data, so we can conclude that PCA is not suitable
# in this example
plt.figure() 
colors = ['navy', 'turquoise','darkorange'] 
lw = 0.5 
for color, i, target_name in zip(colors, [0, 1, 2], target_names): 
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name) 

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA of dataset') 
plt.show()

