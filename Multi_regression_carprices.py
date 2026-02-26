

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = {
"Engine_Size": [1.2,1.5,1.8,2.0,2.2,1.3,1.6,2.4,2.0,1.4,1.7,2.5,1.8,2.2,1.5],
"Mileage": [90,70,60,50,40,85,65,30,45,80,55,25,50,35,75],
"Age": [8,6,5,4,3,7,6,2,4,7,5,1,3,2,6],
"Horsepower": [80,95,110,130,150,85,100,180,140,90,115,200,125,160,105],
"Price": [3.5,5,6,8,10,4,5.5,14,9,4.5,6.5,16,8.5,12,5.2]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df.head())

x = df[["Engine_Size","Mileage","Age","Horsepower"]]
y = df["Price"]

x_train,x_test,y_train,y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test.values, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)



new_price = pd.DataFrame({
    "Engine_Size": [2.5],
    "Mileage": [90],
    "Age": [10],
    "Horsepower": [120]
})

predicted_price = model.predict(new_price)

print(f"\nPredicted the car: {predicted_price[0]:.2f} prices")