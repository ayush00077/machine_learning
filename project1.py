import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("/Users/ayush/Desktop/machine_learning/housing_dataset - Sheet1.csv")

# Features and target
x = df[['size', 'bedrooms', 'bathrooms', 'age', 'location']]
y = df['price']

# Train-validation split
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# Check sizes
print(f"Training set: {len(x_train)} samples")
print(f"Validation set: {len(x_val)} samples")