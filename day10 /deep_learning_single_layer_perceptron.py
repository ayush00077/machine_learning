# Scenario:
# We want to classify whether a restaurant order is “Large” (1) or “Small” (0) based on the number
#  of items ordered.
# - If items ≥ 3 → Large order
# - If items < 3 → Small order

import numpy as np

X = np.array([[2],[4],[3],[5]])
y = np.array([0,0,1,1])

weights = np.random.rand(1)
bias = np.random.rand(1)
learning_rate = 0.1

# step 3: Activation Function

def activation(z):
    return 1 if z >= 0 else 0

for epoch in range(10):
    print(f"Epoch {epoch+1}")
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        y_pred = activation(z)

        # Error
        error = y[i] - y_pred

        # Update rule
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

        print(f"Input: {X[i]}, Predicted: {y_pred}, Actual: {y[i]}, Error: {error}")

print("\nFinal Weights:", weights)
print("Final Bias:", bias)

# Step 5: Test the perceptron
test_items = np.array([3, 5])
for h in test_items:
    z = np.dot(h, weights) + bias
    print(f"Items: {h}, Prediction: {activation(z)}")