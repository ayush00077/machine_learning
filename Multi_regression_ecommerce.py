import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df=pd.read_csv("/Users/ayush/Desktop/machine_learning/Delievery_dataset - Sheet1 (1).csv")

# printing database head
print(df.head())

# defining fatures and targets
X=df[["Distance_km","Items","Traffic_Level","Processing_Time_hr"]]
y=df["Delivery_Time_hr"]


# train,test,split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# training the model after this step slope and intercept gets assigned
model = LinearRegression()
model.fit(X_train, y_train)

# so now we are feeding input data that we had at beggining 20% to test for values

y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test.values, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)


# Predict new delivery time
new_time = pd.DataFrame({
    "Distance_km": [5],
    "Items": [2],
    "Traffic_Level": [6],
    "Processing_Time_hr": [4]
})

new_salary = model.predict(new_time)

print(f"\nPredicted time: {new_salary[0]:.2f} hours")
