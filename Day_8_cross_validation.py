# Scenario-Based Question
# A university wants to build a predictive model to estimate
#  student grades based on four factors:
# - Study hours per week
# - Attendance percentage
# - Previous exam score
# - Average sleep hours
# They collect data from 200 students and decide to use Ridge
#  Regression for prediction. To evaluate the model, they apply different cross-validation strategies:
# - Basic K-Fold CV (5 folds, shuffled) to check the stability of the model’s R² scores.
# - Multi-metric evaluation using both R² and Mean Squared Error (MSE), comparing training and validation
# scores.
# - Stratified K-Fold CV (for a separate classification task predicting pass/fail using Logistic
#                         Regression).



from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np

# Step 1: Create synthetic dataset
# 1000 samples, 5 features
X, y = make_regression(
    n_samples=1000,
    n_features=5,
    noise=15,
    random_state=42
)

# Step 2: Initialize model
model = LinearRegression()

# Step 3: Define K-Fold
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Step 4: Perform cross-validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring='r2'
)

# Step 5: Print results
print("R2 scores for each fold:", scores)
print("Average R2 score:", np.mean(scores))

if scores.std()<0.05:
    print("model is stable")

else:
    print("model performance varies across folds so require investigation")    