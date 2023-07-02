import torch
import torch.nn as nn

# Create a model with Batch Normalization
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.BatchNorm1d(10)
)

# Print the initial num_batches_tracked value
print(model[1].num_batches_tracked)  # Output: 0

# Set the model in training mode
model.train()

# Process a batch of input
inputs = torch.randn(16, 10)
outputs = model(inputs)

# Print the updated num_batches_tracked value
print(model[1].num_batches_tracked)  # Output: 1
