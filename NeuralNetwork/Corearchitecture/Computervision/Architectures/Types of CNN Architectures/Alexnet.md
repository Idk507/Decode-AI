**AlexNet** is a **deep convolutional neural network (CNN)** that **won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012** with a **top-5 error rate of 15.3%** — a massive leap over the previous year's best of 25.8%. It was developed by **Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton** and introduced in the paper:  
> **"ImageNet Classification with Deep Convolutional Neural Networks" (2012)**.

AlexNet proved that **deep CNNs could scale to large datasets** and achieve **superhuman performance** on complex visual tasks, kickstarting the modern deep learning revolution.

---

## Complete Guide to AlexNet: Architecture, Code, Training, & Everything

---

## 1. **Key Innovations of AlexNet**

| Innovation | Description |
|----------|-----------|
| **Deep Architecture** | 8 learnable layers (5 conv + 3 FC) — much deeper than LeNet |
| **ReLU Activation** | First major use of **ReLU** → faster training, avoids vanishing gradients |
| **Dropout** | Applied in fully connected layers to prevent overfitting |
| **Data Augmentation** | Random crops, flips, PCA color jittering |
| **GPU Training** | Trained on **2 GTX 580 GPUs** using **model parallelism** |
| **Local Response Normalization (LRN)** | Enhanced generalization (now rarely used) |
| **Overlapping Max Pooling** | Pooling with stride < kernel size |

---

## 2. **AlexNet Architecture (Detailed Breakdown)**

| Layer | Type | Input Size | Output Size | Kernel | Stride | Padding | Parameters |
|-------|------|------------|-------------|--------|--------|---------|------------|
| **Input** | Image | 3 × 227 × 227 | - | - | - | - | - |
| **C1** | Conv | 3×227×227 | **96×55×55** | 11×11 | 4 | 0 | (11×11×3×96) = **34,944** |
| **ReLU1** | ReLU | 96×55×55 | 96×55×55 | - | - | - | 0 |
| **Norm1** | LRN | 96×55×55 | 96×55×55 | - | - | - | 0 |
| **Pool1** | MaxPool | 96×55×55 | **96×27×27** | 3×3 | 2 | 0 | 0 |
| **C2** | Conv | 96×27×27 | **256×27×27** | 5×5 | 1 | 2 | (5×5×96×256) = **614,400** |
| **ReLU2** | ReLU | 256×27×27 | 256×27×27 | - | - | - | 0 |
| **Norm2** | LRN | 256×27×27 | 256×27×27 | - | - | - | 0 |
| **Pool2** | MaxPool | 256×27×27 | **256×13×13** | 3×3 | 2 | 0 | 0 |
| **C3** | Conv | 256×13×13 | **384×13×13** | 3×3 | 1 | 1 | (3×3×256×384) = **884,736** |
| **ReLU3** | ReLU | 384×13×13 | 384×13×13 | - | - | - | 0 |
| **C4** | Conv | 384×13×13 | **384×13×13** | 3×3 | 1 | 1 | (3×3×384×384) = **1,327,104** |
| **ReLU4** | ReLU | 384×13×13 | 384×13×13 | - | - | - | 0 |
| **C5** | Conv | 384×13×13 | **256×13×13** | 3×3 | 1 | 1 | (3×3×384×256) = **884,736** |
| **ReLU5** | ReLU | 256×13×13 | 256×13×13 | - | - | - | 0 |
| **Pool5** | MaxPool | 256×13×13 | **256×6×6** | 3×3 | 2 | 0 | 0 |
| **FC6** | Fully Connected | 256×6×6 = 9216 | **4096** | - | - | - | (9216×4096) = **37,748,736** |
| **ReLU6** | ReLU | 4096 | 4096 | - | - | - | 0 |
| **Dropout6** | Dropout (p=0.5) | 4096 | 4096 | - | - | - | 0 |
| **FC7** | Fully Connected | 4096 | **4096** | - | - | - | (4096×4096) = **16,777,216** |
| **ReLU7** | ReLU | 4096 | 4096 | - | - | - | 0 |
| **Dropout7** | Dropout (p=0.5) | 4096 | 4096 | - | - | - | 0 |
| **FC8** | Fully Connected | 4096 | **1000** (ImageNet classes) | - | - | - | (4096×1000) = **4,096,000** |
| **Softmax** | Softmax | 1000 | 1000 | - | - | - | 0 |

> **Total Parameters**: **~60.9 million**  
> **Input Size**: **227×227×3** (not 224×224 — common misconception due to rounding in diagrams)

---

## 3. **Why 227×227 and not 224×224?**

- Start with 224×224 → after first conv (11×11, stride 4):  
  `(224 - 11)/4 + 1 = 54.25` → not integer!
- But `(227 - 11)/4 + 1 = 55` → integer → valid.

So **227×227** ensures clean integer dimensions throughout.

---

## 4. **Full PyTorch Implementation (Modernized)**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_classes = 1000
batch_size = 64
learning_rate = 0.01
num_epochs = 90  # AlexNet trained for ~90 epochs

# Data Augmentation & Preprocessing
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Note: Use ImageNet or CIFAR-10/100 for demo
# Here we use CIFAR-10 (resize to 227)
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# AlexNet Model
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x

# For CIFAR-10, change output to 10
model = AlexNet(num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# Learning rate scheduler (step decay as in paper)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training function
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item():.4f}')

        scheduler.step()
        print(f'Epoch {epoch+1} - Avg Loss: {running_loss/len(train_loader):.4f}')

# Test function
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

# Run
print("Training AlexNet on CIFAR-10...")
train()
print("Testing...")
test()
```

---

## 5. **Training Details (Original Paper)**

| Hyperparameter | Value |
|---------------|-------|
| **Optimizer** | SGD with momentum = 0.9 |
| **Initial LR** | 0.01 |
| **LR Policy** | Divide by 10 when validation error stops improving |
| **Weight Decay** | 0.0005 |
| **Batch Size** | 128 |
| **Epochs** | ~90 |
| **Weight Init** | Gaussian (mean=0, std=0.01) for conv, constant 1 for biases in layers 2,4,5,FC |
| **Bias Init** | 0 for layer 1,3; 1 for others |

---

## 6. **Data Augmentation (Critical!)**

```python
# Random crops from 256×256
# Horizontal flips
# PCA Color Augmentation: add multiples of principal components
```

> Without augmentation, AlexNet **overfits heavily**.

---

## 7. **Local Response Normalization (LRN)**

```python
# Rarely used today, but in AlexNet:
nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
```

Now replaced by **Batch Normalization**.

---

## 8. **Model Parallelism (Original Training)**

- Trained on **2 GPUs**
- **Split model across GPUs**:
  - GPU 1: Top half of layers
  - GPU 2: Bottom half
  - Communication only at certain layers (C2, C4, C5, FC)

---

## 9. **Performance**

| Model | Top-1 Error | Top-5 Error |
|-------|-------------|-------------|
| **AlexNet (2012)** | 37.5% | **15.3%** |
| Previous best (2011) | - | 25.8% |
| Human (estimated) | - | ~5% |

---

## 10. **Modern Perspective: Why AlexNet is Outdated**

| Feature | AlexNet | Modern (2025) |
|-------|--------|-------------|
| Depth | 8 layers | 100+ (ResNet, ViT) |
| Parameters | 60M | 25M–400M |
| Normalization | LRN | BatchNorm / LayerNorm |
| Pooling | Overlapping | Non-overlapping / Global Avg |
| Optimization | SGD | AdamW |
| Architecture | Sequential | Residual, Attention |

Still, **AlexNet started it all**.

---

## 11. **Visual Architecture (Text Diagram)**

```
Input (227×227×3)
│
├─► Conv1 (11×11, 96, s=4) → 55×55×96
│     ReLU + LRN + MaxPool(3×3,s=2) → 27×27×96
│
├─► Conv2 (5×5, 256, p=2) → 27×27×256
│     ReLU + LRN + MaxPool(3×3,s=2) → 13×13×256
│
├─► Conv3 (3×3, 384, p=1) → 13×13×384 → ReLU
│
├─► Conv4 (3×3, 384, p=1) → 13×13×384 → ReLU
│
├─► Conv5 (3×3, 256, p=1) → 13×13×256 → ReLU + MaxPool → 6×6×256
│
├─► FC6 (9216 → 4096) → ReLU + Dropout
│
├─► FC7 (4096 → 4096) → ReLU + Dropout
│
└─► FC8 (4096 → 1000) → Softmax
```

---

## 12. **Parameter Count Chart (Chart.js)**

<img width="890" height="475" alt="image" src="https://github.com/user-attachments/assets/32110b79-4060-434d-bb67-30aae63460c1" />


> **FC layers dominate** (~95% of parameters).

---

## 13. **Pretrained AlexNet**

```python
import torchvision.models as models
model = models.alexnet(pretrained=True)  # Loads ImageNet weights
```

Use for transfer learning on smaller datasets.

---

## 14. **Summary**

| Aspect | Details |
|------|--------|
| **Year** | 2012 |
| **Authors** | Krizhevsky, Sutskever, Hinton |
| **Breakthrough** | Won ILSVRC 2012 by 10% margin |
| **Key Ideas** | ReLU, Dropout, Data Aug, GPU training |
| **Layers** | 5 Conv + 3 FC |
| **Input** | 227×227×3 |
| **Params** | ~60.9M |
| **Legacy** | Sparked the deep learning boom |

---

## Run the Code

```bash
pip install torch torchvision
python alexnet.py
```

> On CIFAR-10, expect **~80–85% accuracy** after 50 epochs.

---

