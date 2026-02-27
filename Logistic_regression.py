import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# create simple dataset
# driver ages and whther they filed a claim(1=claim filed , 0=no claim)

X=np.array([[18],[20],[22],[25],[28],[30],[35],[40],[45],[50]])
y=np.array([1,1,1,0,0,0,0,0,1,1])


# split into train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# train logistic regression model
model=LogisticRegression()
model.fit(X_train,y_train)


y_pred=model.predict(X_test)

# evaluate accuracy
print("Accuracy",accuracy_score(y_test,y_pred))



# Step 7 : Visualise decision boundary
plt.scatter(X,y,color="blue",label="Data points")
x_range=np.linspace(15,55,200).reshape(-1,1)
y_prob=model.predict_proba(x_range)[:,1]
plt.plot(x_range,y_prob,color="red",label="Logistic curve")
plt.xlabel("Driver age")
plt.ylabel("Probability of filing claim")
plt.legend()
plt.show()