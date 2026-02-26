# Scenario: Predicting Employee Salary Based on Multiple Factors

# A company wants to predict employee salary based on several important factors:

# Years of Experience

# Education Level (1 = Bachelor, 2 = Master, 3 = PhD)

# Number of Skills Known

# Performance Rating (1 to 5)

# Since salary depends on multiple variables, the company uses Multiple Linear Regression.

# Salary=b0​+b1​(Experience)+b2​(EducationLevel)+b3​(Skills)+b4​(Performance)


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("/Users/ayush/Desktop/machine_learning/multil_salary_pred - Sheet1.csv")

print("Dataset:")
print(df.head())

# Define features and target
X = df[["Experience_years","Education_Level","Skills_Count","Performance_Rating"]]
y = df["Salary_lpa"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict on test data
y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test.values, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# Predict new employee
new_employee = pd.DataFrame({
    "Experience_years": [5],
    "Education_Level": [2],
    "Skills_Count": [6],
    "Performance_Rating": [4]
})

predicted_salary = model.predict(new_employee)

print(f"\nPredicted Salary: {predicted_salary[0]:.2f} lakhs")