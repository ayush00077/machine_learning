# Scenario: Predicting Patient Recovery Time
# A hospital research team wants to build a model to predict patient recovery time (in days) after
#  surgery based on several factors such as:
# - Age of the patient
# - Number of hours of post-surgery physiotherapy per week
# - Pre-existing health conditions (numeric severity score)
# - Length of hospital stay (days)
# - Average sleep hours during recovery
# They collect data from 1,000 patients and decide to use Linear Regression.
# To evaluate the model, they apply 5-Fold Cross-Validation with R² as the performance metric.


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