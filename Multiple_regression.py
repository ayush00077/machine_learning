#Multilinear Regression Scenario

# Scenario: Predicting House Price Based on Multiple Factors
# A real estate company wants to predict the price of a house based on several important features:
# Area (square feet)
# Number of bedrooms
# Age of the house (years)
# Distance from city center (km)
# Since multiple factors influence price, we use Multiple Linear Regression.
# Price=b0+b1(Area)+b2(Bedrooms)+b3(Age)+b4(Distance)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

#Step:3 Create dataset

data = {
    "Area_sqft": [800,1000,1200,1500,1800,2000,2200,2500,900,1600,1400,2100],
    "Bedrooms": [2,2,3,3,4,4,4,5,2,3,3,4],
    "Age_years": [15,10,8,5,4,3,2,1,12,6,7,3],
    "Distance_km": [12,10,8,6,5,4,3,2,11,7,9,4],
    "Price_lakhs": [40,50,62,75,90,105,120,140,45,80,70,110]
}

df=pd.DataFrame(data)

print("Dataset:")
print(df)

#Step 3 : define features and targets
X=df[['Area_sqft','Bedrooms','Age_years','Distance_km']]
y=df["Price_lakhs"]

# Step 4 : Split Dataset
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2, random_state=42
)

# Step 5 : Train Model
model=LinearRegression()
model.fit(X_train,y_train)

# Step 6 : Predict on test data
y_pred=model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test.values, y_pred):
    print(f"Actual: {actual:.2f} ,Predicted :{pred:.2f}")

#Step 8 : Evaluate Model
print("\nMean Absolute Error: ", mean_absolute_error(y_test,y_pred))
print("R2 Score: ", r2_score(y_test,y_pred))

# Step 9 : Predict Price of new house
#Example :1700 sqft, 3 bedrooms, 5 years old, 6 km from city
new_house_data={
    "Area_sqft":[1700],
    "Bedrooms":[3],
    "Age_years":[5],
    "Distance_km":[6]
}

new_house_df=pd.DataFrame(new_house_data)
predicted_price=model.predict(new_house_df)

print(f"\nPredicted house Price : {predicted_price[0]:.2f} lakhs")