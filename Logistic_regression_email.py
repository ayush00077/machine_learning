# Scenario 📧
# An email service provider wants to build a simple model to detect whether an incoming email is spam
#  or not spam. They decide to use three features:
# - Word Count – total number of words in the email.
# - Has Link – whether the email contains a hyperlink (1 = yes, 0 = no).
# - Caps Ratio – proportion of words written in ALL CAPS.
# They collect a small dataset of emails with these features and label them as spam (1) or not spam (0).

# Question for Students
# Using logistic regression:
# - Train a model to classify emails as spam or not spam.
# - Evaluate the model’s accuracy on a test set.
# - Interpret the predicted probabilities — what does it mean if an email has a 70% probability of being spam?
# - Test the model on a new email

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Dataset
# Toy spam dataset
# Features : [word_Count,has_link,caps_ratio]
X = np.array([
    [50, 1, 0.8],   # SPAM
    [200, 0, 0.1],  # Not spam
    [30, 1, 0.9],   # SPAM
    [180, 0, 0.05], # Not spam
    [10, 1, 0.95],  # SPAM
    [220, 0, 0.08], # Not spam
])
y=np.array([1,0,1,0,1,0]) # 1=spam , 0= not spam

# Split data
X_train,x_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train the model
model=LogisticRegression()
model.fit(X_train,y_train)

# Predicting the data
y_pred=model.predict(x_test)
proba=model.predict_proba(x_test)[:,1]

print("Predictions : ",y_pred)
print("Probabilities : ",proba.round(2))
print("Accuracy : ",accuracy_score(y_test,y_pred))

# try a new email
new_email=[[15,1,0.88]]
print("New Email Predicition",model.predict(new_email)[0])



