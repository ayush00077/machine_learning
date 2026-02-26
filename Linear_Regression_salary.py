import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df=pd.read_csv('salary_lpa.csv')
df.describe()

# Step 3 :define features and targets
x=df[['Experience_years']]
y=df[['Salary_lpa']]

# Step 4 : Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
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
    print(f"Actual: {actual[0]:.2f}, Predicted: {pred[0]:.2f}")

# Step 8 : Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# Step 9 : Predict new Salary
new_salary = np.array([[10]])
predicted_salary = model.predict(new_salary)

print(f"\nPredicted salary for new experience: {predicted_salary[0][0]:.2f} lakhs")

# Step 10 : Visualization
plt.scatter(x, y, label="Actual Data")
plt.plot(x, model.predict(x), linewidth=2, label="Regression Line")
plt.xlabel("Experience years")
plt.ylabel("Salary")
plt.title("New Salary Prediction")
plt.legend()
plt.show()