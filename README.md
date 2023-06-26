# iris-classification
#import liabraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snsi
#read a csv file
iris_df=pd.read_csv("/Iris.csv")
iris_df.head()
# Drop the column Id of the dataset
iris_df=iris_df.drop(['Id'], axis=1)
iris_df.head()
# Change the names of the columns
iris_df.columns
iris_df.columns=['sepal_length','sepal_width','petal_length','petal_width','species']
# Top 5 records of the dataset
iris_df.head()
# Basic informations about the columns
iris_df.info()
# Count of species column of the dataset
species_counts=iris_df['species'].value_counts()
print("Count of the species column of the dataset : ")
print(species_counts)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# Make the data X and Y
X=iris_df.drop('species',axis=1)
X.head()
type(X)
Y=iris_df['species']
# Split the data into training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.24,random_state=20)
X_train.head()
X_train.shape
type(X_train)
Y_train.head()
Y_train.shape
type(Y_train)
# Prediction
y_pred=model.predict(X_test)
y_pred
print(X_test.shape)
print(y_pred.shape)
# Training accuracy
train_accuracy=model.score(X_train,Y_train)
print("The training accuracy is",train_accuracy)
# Test accuracy
test_accuracy=model.score(X_test,Y_test)
print("The testing accuracy is",test_accuracy)
