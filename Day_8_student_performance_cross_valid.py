
# Scenario: Predicting Student Exam Performance
# A university research team wants to build a model to predict student exam scores (out of 100) based on several factors such as:
# - Number of study hours per week
# - Attendance percentage in lectures
# - Prior GPA (Grade Point Average)
# - Participation in group projects (numeric engagement score)
# - Average sleep hours during exam preparation
# They collect data from 800 students across different departments and decide to use Linear Regression.
# To evaluate the model, they apply 5-Fold Cross-Validation with R² as the performance metric.


from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np

# data:-800 students, 5 features:
# study hours, attendance, GPA, group activity score, sleep hours

X, y = make_regression(
    n_samples=800,
    n_features=5,
    noise=8,
    random_state=42
)


model = LinearRegression()

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


r2_scores = cross_val_score(model,X,y,cv=kfold,scoring='r2')

print("R² score for each fold:", r2_scores)
print("Average R² score:", np.mean(r2_scores))