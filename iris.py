import pandas as pd
import matplotlib.pyplot as plt

#load iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

#To see features of iris
iris.feature_names

#Creating data frame for easy visualization
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()

#Adding target no in front of species
df['target'] = iris.target
df.head()

#To see species of target no 1
df[df.target==1].head()

#To replace target no by name of species
df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

#Sepal length vs Sepal Width (Setosa vs Versicolor)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')

#Petal length vs Pepal Width (Setosa vs Versicolor)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')

#Train Using Support Vector Machine (SVM)
from sklearn.model_selection import train_test_split

X = df.drop(['target','flower_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


len(X_train)
len(X_test)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

model.score(X_test, y_test)

model.predict([[4.8,3.0,1.5,0.3]])

#Regularization (C)
model_C = SVC(C=10)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)

#Gamma
model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
model_g.score(X_test, y_test)

#Kernel
model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)

model_linear_kernal.score(X_test, y_test)



