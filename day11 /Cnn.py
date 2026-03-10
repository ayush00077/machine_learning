# Scenario Question
# A machine learning engineer is designing a convolutional neural network (CNN) to process grayscale images of handwritten digits
#  (28×28 pixels). She starts with a single convolutional layer defined as follows

# import torch
# import torch.nn as nn

# # Single Conv2D layer
# conv = nn.Conv2d(
#     in_channels  = 1,    # grayscale input
#     out_channels = 32,   # 32 filters -> 32 feature maps
#     kernel_size  = 3,    # 3x3 filter
#     stride       = 1,
#     padding      = 1     # SAME padding: output = same size
# )

# # Forward pass
# x   = torch.randn(1, 1, 28, 28) # (batch, C, H, W)
# out = conv(x)
# print(out.shape) # torch.Size([1, 32, 28, 28])

# # Inspect learnable params
# print('Weights:', conv.weight.shape) # (32, 1, 3, 3)
# print('Bias   :', conv.bias.shape)   # (32,)
# print('Total  :', 32*1*3*3 + 32)     # = 320


# TENSORFLOW CODE :-

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np

# Create Conv2D layer
conv = Conv2D(
    filters=32,          # 32 filters -> 32 feature maps
    kernel_size=(3,3),   # 3x3 filter
    strides=1,
    padding='same',      # SAME padding -> output size remains same
    input_shape=(28,28,1)
)

# Create dummy input (batch, height, width, channels)
x = np.random.randn(1, 28, 28, 1).astype(np.float32)

# Forward pass
out = conv(x)

print("Output shape:", out.shape)

# Inspect parameters
weights = conv.get_weights()

print("Weights shape:", weights[0].shape)  # kernel weights
print("Bias shape:", weights[1].shape)     # bias

# Total parameters
print("Total:", 32*1*3*3 + 32)


