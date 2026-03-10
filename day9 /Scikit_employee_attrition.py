# 📖 Scenario: Employee Attrition Prediction 💼
# A company wants to predict whether an employee is likely to stay or leave based on their personal and workplace information. The dataset contains the following features:
# - age → Age of the employee (numeric)
# - years_experience → Number of years of work experience (numeric)
# - department → Department (HR, IT, Sales) (categorical)
# - education → Education level (Graduate, Postgraduate) (categorical)
# - attrition → Target variable (1 = Leaves, 0 = Stays)
# The data science team decides to build a machine learning pipeline that:
# - Standardizes numeric features (age, years_experience).
# - Converts categorical features (department, education) into numerical format using One-Hot Encoding.
# - Combines preprocessing and model training into a single pipeline.
# - Uses Logistic Regression to predict employee attrition.

import pandas as pd

# Example Employee Dataset
data = {
    'age': [25, 30, 28, 35, 40],
    'years_experience': [2, 8, 5, 10, 12],
    'department': ['HR', 'IT', 'Sales', 'IT', 'HR'],
    'education': ['Graduate', 'Postgraduate', 'Graduate', 'Postgraduate', 'Graduate'],
    'attrition': [1, 0, 1, 0, 0]   # 1 = Leaves, 0 = Stays
}

df = pd.DataFrame(data)

# Define Features (X) and Target (y)
X = df[['age', 'years_experience', 'department', 'education']]
y = df['attrition']


# Import sklearn tools
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# Define preprocessing steps
numeric_features = ['age', 'years_experience']
categorical_features = ['department', 'education']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Full Pipeline (Preprocessing + Model)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the model
pipeline.fit(X, y)

# New employee data for prediction
new_employee = pd.DataFrame({
    'age': [29],
    'years_experience': [4],
    'department': ['Sales'],
    'education': ['Graduate']
})

# Make prediction
prediction = pipeline.predict(new_employee)

print("Prediction:", prediction)

if prediction[0] == 1:
    print("The employee is likely to LEAVE.")
else:
    print("The employee is likely to STAY.")