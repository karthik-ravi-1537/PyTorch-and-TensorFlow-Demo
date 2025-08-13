# PyTorch Quick Reference

## Essential Imports
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
```

## Tensor Operations
```python
# Create tensors
x = torch.tensor([1, 2, 3])
x = torch.zeros(3, 4)
x = torch.randn(2, 3)

# Device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)

# Basic operations
y = x + 1
z = torch.matmul(x, y)
```

## Neural Networks
```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Training Loop
```python
model = SimpleNet(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## Common Patterns
- Always call `optimizer.zero_grad()` before backward pass
- Use `model.train()` for training, `model.eval()` for inference
- Use `torch.no_grad()` context for inference to save memory
- Save models with `torch.save(model.state_dict(), 'model.pth')`