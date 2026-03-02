# Scenario: Restaurant Ordering System
# A restaurant wants to build a machine learning model to predict order preparation time.
# They collect data about each order, including:
# - Meal Type: Breakfast, Lunch, Dinner
# - Spice Level: Mild, Medium, Hot
# Since models can’t directly work with text labels, the restaurant uses Label Encoding to 
# convert these categories into numbers.

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data=pd.DataFrame({
    'Meal_Type':['Breakfast', 'Lunch', 'Dinner'],
    'Spice_Level':['Mild', 'Medium', 'Hot']
})


# Step 2: Initialize LabelEncoders
le_meal = LabelEncoder()
data['Meal_Type_Encoded'] = le_meal.fit_transform(data['Meal_Type'])

le_spice = LabelEncoder()
data['Spice_Level_Encoded'] = le_spice.fit_transform(data['Spice_Level'])

# Step 4: View encoding mappings
print("Meal Type Mapping:", dict(zip(le_meal.classes_, le_meal.transform(le_meal.classes_))))
print("Spice Level Mapping:", dict(zip(le_spice.classes_, le_spice.transform(le_spice.classes_))))

# Step 5: Final encoded dataset
print("\nEncoded DataFrame:")
print(data)
