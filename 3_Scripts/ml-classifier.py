# importing required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
#zzzzzzzzz reading csv file and extracting class column to y. 

#bankdata = pd.read_csv("bill_authentication.csv")
#bankdata = pd.read_csv("./Input/uk-2007-05-obvious.csv")
#bankdata = pd.read_csv("./Input/uk-2007-05-link.csv")
#bankdata = pd.read_csv("./Input/uk-2007-05-lbt.csv")
bankdata = pd.read_csv("./Input/uk-2007-05-content.csv")

#bankdata = pd.read_csv("./Input/link-train-gr.csv")
#bankdata = pd.read_csv("./Input/lbt-train-gr.csv")
#bankdata = pd.read_csv("./Input/content-train-gr.csv")

#testdata = pd.read_csv("./Input/uk-2007-05-obvious-test.csv") 
print(bankdata.shape)
#print(bankdata.head() )


#data preprocessing
X = bankdata.drop('class', axis=1)
#X_o = bankdata.drop('class', axis=1)
#X_o2 = X_o.drop('hostid', axis=1)
#X = X_o2.drop('assessmentscore', axis=1)






#index = np.array([1, 2])
#X = X_o[:, index]
#print(X[:5])

#X_test2 = testdata.drop('hostid', axis=1)

#X = X_o.drop('hostid','assesmentscore',axis=2)
#y = bankdata['Class']
y = bankdata['class']

print(X[:5])
#print("Printing y")
#print(y)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y) 

#Train Modddel
from sklearn.svm import SVC  
#svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='poly', degree=2, gamma='scale')
#Gausian
#svclassifier = SVC(kernel='rbf', gamma='scale')
#svclassifier = SVC(kernel='rbf', gamma=.01)
#svclassifier = SVC(kernel='sigmoid', gamma='scale') 
#svclassifier = SVC(kernel='linear')

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
   # knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)



# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
#plt.show()

# 10-fold cross-validation with the best KNN model
#svclassifier = KNeighborsClassifier(n_neighbors=4) #obvious-6,link////lbt-8,10,20,content-4
#svclassifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
from sklearn.linear_model import LogisticRegression
#svclassifier = LogisticRegression(solver='saga', max_iter=50000)#content-200,lbt-1000, link, obvious-50000,saga



from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

#svclassifier = GaussianNB()
#svclassifier = MultinomialNB()

from sklearn.ensemble import RandomForestClassifier
#svclassifier = RandomForestClassifier(random_state=1)

##*************Build      
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)
#y_pred2 = svclassifier.predict(X_test2)







print(y_pred[:5])

#print(y_pred2[:5])

#Evalutae Algooorithm
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))

#accuracy
from sklearn.metrics import accuracy_score

# Evaluate accuracy
print(accuracy_score(y_test,y_pred))

#kkkkkkkkkk-fold vvvvvvliadation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(svclassifier, X, y, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())




#Plot Model
## The line / model
#plt.scatter(y_test, y_pred)
#plt.xlabel("Real Values")
##plt.ylabel("Predictions")
#plt.show()


