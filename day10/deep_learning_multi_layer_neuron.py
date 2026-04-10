
# Scenario 1: Online Course Completion Prediction
# We want to classify whether a student will complete an online course (1) or drop out (0) based on two features:

# Videos Watched – number of course videos the student watched

# Time Spent on Platform – total minutes spent on the learning platform

# Students who watch more videos and spend more time learning are more likely to complete the course.

# Example training data:

# Videos Watched	Time on Platform (min)	Complete Course
# 2	                 15	                      0
# 3	                 20                    	  0
# 8	                 60	                      1
# 9	                 75	                      1
# Because the relationship between engagement and completion may not be perfectly linear, we use a multi-layer neural network with a hidden layer to learn the pattern.

import numpy as np

# Step 1: Training Data
# [Videos Watched, Time on Platform]
X = np.array([
    [2, 15],
    [3, 20],
    [8, 60],
    [9, 75]
])

# Output: 0 = Dropout, 1 = Complete
y = np.array([[0], [0], [1], [1]])

# Step 2: Initialize weights and bias
np.random.seed(42)

weights_input_hidden = np.random.rand(2, 2)
bias_hidden = np.random.rand(1, 2)

weights_hidden_output = np.random.rand(2, 1)
bias_output = np.random.rand(1, 1)

learning_rate = 0.1

# Step 3: Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

# Step 4: Training the Neural Network
for epoch in range(1000):

    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Error calculation
    error = y - predicted_output

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate


# Step 5: Testing with new students
test_students = np.array([
    [7, 55],   # likely to complete
    [2, 10]    # likely to drop out
])

hidden_test = sigmoid(np.dot(test_students, weights_input_hidden) + bias_hidden)
final_prediction = sigmoid(np.dot(hidden_test, weights_hidden_output) + bias_output)

print("Predictions (probability of completion):")
print(final_prediction)