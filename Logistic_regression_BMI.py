
# Scenario 🏥
# A hospital wants to predict whether patients are at risk of developing diabetes based on their BMI (Body Mass Index). They collect data from 10 patients,
#  recording BMI values and whether the patient was diagnosed with diabetes (1 = diabetes, 0 = no diabetes).

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("/Users/ayush/Desktop/machine_learning/BMI_dataset - Sheet1.csv")
X=df[["BMI"]]
y=df["Diabetes"]
# train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


model=LogisticRegression()
model.fit(X_train,y_train)

y_predict=model.predict(X_test)


print("Accuracy",accuracy_score(y_test,y_predict))









