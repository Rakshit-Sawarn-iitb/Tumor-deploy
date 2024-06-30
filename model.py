import timm
model = timm.create_model("xception", pretrained=True)
import torch.nn as nn
import torch.optim as optim
import torch
# Modify the first convolutional layer to accept single-channel input
original_conv = model.conv1
model.conv1 = nn.Conv2d(
    in_channels=1,  # Change to 1 for single-channel input
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias
)

# If you want to initialize the new single-channel weights using the mean of the RGB weights
with torch.no_grad():
    model.conv1.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)

# Check the number of input features to the final fully connected layer
in_features = model.fc.in_features

# Replace the final fully connected layer with a new one for binary classification
model.fc = nn.Linear(in_features, 4)