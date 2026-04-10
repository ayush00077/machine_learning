
# Scenario Question

# A biomedical researcher is designing a convolutional neural network (CNN) to analyze microscopic grayscale images of blood cells (64×64 pixels). She begins with a single convolutional layer defined as follows:

# Input: 64×64 grayscale images (1 channel)
# Convolutional Layer: 16 filters, each of size 5×5
# Stride: 1
# Padding: Same
# Activation Function: ReLU
# The goal of this layer is to extract local structural features such as cell boundaries and texture variations, which will later help in classifying whether the blood cells are healthy or show signs of abnormality.


# import torch
# import torch.nn as nn

# # Convolutional Layer
# conv_layer = nn.Conv2d(
#     in_channels=1,      # grayscale input
#     out_channels=16,    # 16 filters
#     kernel_size=5,      # 5x5 filter
#     stride=1,
#     padding=2           # SAME padding
# )

# # Activation function
# relu = nn.ReLU()

# # Sample input image (batch_size, channels, height, width)
# x = torch.randn(1, 1, 64, 64)

# # Forward pass
# conv_out = conv_layer(x)
# activated_out = relu(conv_out)

# print("Output Shape:", activated_out.shape)

# # Inspect parameters
# print("Weight shape:", conv_layer.weight.shape)
# print("Bias shape:", conv_layer.bias.shape)

# # Total parameters
# total_params = 16*1*5*5 + 16
# print("Total learnable parameters:", total_params)


# TENSORFLOW CODE:-

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU
import numpy as np

# Convolutional Layer
conv_layer = Conv2D(
    filters=16,           # 16 filters
    kernel_size=(5,5),    # 5x5 filter
    strides=1,
    padding='same',       # SAME padding
    input_shape=(64,64,1)
)

# Activation function
relu = ReLU()

# Sample input image (batch, height, width, channels)
x = np.random.randn(1, 64, 64, 1).astype(np.float32)

# Forward pass
conv_out = conv_layer(x)
activated_out = relu(conv_out)

print("Output Shape:", activated_out.shape)

# Inspect parameters
weights = conv_layer.get_weights()

print("Weight shape:", weights[0].shape)
print("Bias shape:", weights[1].shape)

# Total parameters
total_params = 16*1*5*5 + 16
print("Total learnable parameters:", total_params)