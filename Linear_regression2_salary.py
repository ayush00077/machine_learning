

# Step 1 : Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2 : Create dataset
df=pd.read_csv("/Users/ayush/Desktop/machine_learning/salary_lpa - Sheet1.csv")

# df = pd.DataFrame(data)

print("Dataset:")
print(df)


# Step 3 : Define features and target
X = df[["Experience_years"]]
y = df["Salary_lpa"]  # Make it 1D

# Step 4 : Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5 : Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6 : Model parameters
print("\nSlope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Step 7 : Predict on test data
y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test.values, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# Step 8 : Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# Step 9 : Predict new house price
new_salary = np.array([[10]])
predicted_salary = model.predict(new_salary)

print(f"\nPredicted salary for 15 years: {predicted_salary[0]:.2f} lakhs")

# Step 10 : Visualization
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), linewidth=2, label="Regression Line")
plt.xlabel("Experience")
plt.ylabel("salary")
plt.title("salary_predictore")
plt.legend()
plt.show()