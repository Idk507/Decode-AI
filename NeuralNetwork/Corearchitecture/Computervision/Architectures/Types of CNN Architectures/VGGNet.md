# **VGGNet: Complete Guide (Architecture, Code, Training, Analysis)**

**VGGNet** (Visual Geometry Group Network) is a **deep convolutional neural network** developed by **Simonyan & Zisserman** from the University of Oxford in 2014. It **placed 2nd in ILSVRC 2014** (ImageNet) with **7.3% top-5 error**, just behind GoogLeNet (6.7%).

> **Paper**: *"Very Deep Convolutional Networks for Large-Scale Image Recognition"* (2014)  
> **Key Idea**: **Deeper networks with small 3×3 filters** outperform wider networks with large filters.

---

## **1. Core Innovations of VGGNet**

| Innovation | Description |
|----------|-------------|
| **Small 3×3 Filters** | All conv layers use **3×3 kernels** (receptive field equivalent to larger filters when stacked) |
| **Very Deep** | 16–19 weight layers (vs AlexNet’s 8) |
| **Uniform Architecture** | Consistent design: stack of conv → ReLU → pool |
| **1×1 Convolutions** | Used in some variants for dimensionality reduction |
| **No LRN** | Dropped Local Response Normalization (from AlexNet) |
| **Pretrained Models** | Widely used for **transfer learning** |

---

## **2. VGG Variants (A, B, C, D, E)**

| Config | A | A-LRN | B | C | D (VGG16) | E (VGG19) |
|--------|---|---|---|---|----------|----------|
| **Layers** | 11 | 11 | 13 | 16 | **16** | **19** |
| **Conv 3×3** | 8 | 8 | 10 | 13 | **13** | **16** |
| **FC Layers** | 3 | 3 | 3 | 3 | 3 | 3 |
| **Params** | 133M | 133M | 133M | 138M | **138M** | **143M** |

> **VGG16** and **VGG19** are the most popular.

---

## **3. VGG16 Architecture (Most Common)**

| Layer | Type | Input | Output | Kernel | Stride | Padding | #Filters | Params |
|-------|------|-------|--------|--------|--------|---------|----------|--------|
| Input | - | - | 224×224×3 | - | - | - | - | - |
| **Block 1** |
| conv1_1 | Conv | 224×224×3 | 224×224×64 | 3×3 | 1 | 1 | 64 | 1,728 |
| conv1_2 | Conv | 224×224×64 | 224×224×64 | 3×3 | 1 | 1 | 64 | 36,864 |
| pool1 | MaxPool | 224×224×64 | **112×112×64** | 2×2 | 2 | 0 | - | 0 |
| **Block 2** |
| conv2_1 | Conv | 112×112×64 | 112×112×128 | 3×3 | 1 | 1 | 128 | 73,728 |
| conv2_2 | Conv | 112×112×128 | 112×112×128 | 3×3 | 1 | 1 | 128 | 147,456 |
| pool2 | MaxPool | 112×112×128 | **56×56×128** | 2×2 | 2 | 0 | - | 0 |
| **Block 3** |
| conv3_1 | Conv | 56×56×128 | 56×56×256 | 3×3 | 1 | 1 | 256 | 294,912 |
| conv3_2 | Conv | 56×56×256 | 56×56×256 | 3×3 | 1 | 1 | 256 | 589,824 |
| conv3_3 | Conv | 56×56×256 | 56×56×256 | 3×3 | 1 | 1 | 256 | 589,824 |
| pool3 | MaxPool | 56×56×256 | **28×28×256** | 2×2 | 2 | 0 | - | 0 |
| **Block 4** |
| conv4_1 | Conv | 28×28×256 | 28×28×512 | 3×3 | 1 | 1 | 512 | 1,179,648 |
| conv4_2 | Conv | 28×28×512 | 28×28×512 | 3×3 | 1 | 1 | 512 | 2,359,296 |
| conv4_3 | Conv | 28×28×512 | 28×28×512 | 3×3 | 1 | 1 | 512 | 2,359,296 |
| pool4 | MaxPool | 28×28×512 | **14×14×512** | 2×2 | 2 | 0 | - | 0 |
| **Block 5** |
| conv5_1 | Conv | 14×14×512 | 14×14×512 | 3×3 | 1 | 1 | 512 | 2,359,296 |
| conv5_2 | Conv | 14×14×512 | 14×14×512 | 3×3 | 1 | 1 | 512 | 2,359,296 |
| conv5_3 | Conv | 14×14×512 | 14×14×512 | 3×3 | 1 | 1 | 512 | 2,359,296 |
| pool5 | MaxPool | 14×14×512 | **7×7×512** | 2×2 | 2 | 0 | - | 0 |
| **FC Layers** |
| fc6 | FC | 7×7×512 = 25088 | 4096 | - | - | - | - | 102,764,544 |
| fc7 | FC | 4096 | 4096 | - | - | - | - | 16,781,312 |
| fc8 | FC | 4096 | **1000** | - | - | - | - | 4,097,000 |
| **Softmax** | - | 1000 | 1000 | - | - | - | - | 0 |

> **Total Parameters**: **~138 million**  
> **Input Size**: **224×224×3**  
> **Receptive Field**: 3×3 conv stacked → same as 5×5, 7×7, etc., but **more non-linearities**

---

## **Why 3×3 Filters?**

| Stack | Receptive Field | #Params (vs 7×7) | Non-linearities |
|-------|------------------|------------------|-----------------|
| 1× (7×7) | 7×7 | 49×C | 1 |
| 3× (3×3) | 7×7 | **27×C** | **3** |

→ **Fewer parameters + more ReLUs = better generalization**

---

## **4. Full PyTorch Implementation (VGG16)**

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
num_classes = 10  # CIFAR-10
batch_size = 64
learning_rate = 0.01
num_epochs = 50
weight_decay = 5e-4

# Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),  # VGG expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# VGG16 Configuration
cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),  # Modern addition
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]  # Not in original
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Model
model = VGG('D', num_classes=num_classes).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Training
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item():.4f}')

        scheduler.step()
        print(f'Epoch {epoch+1} - Avg Loss: {running_loss/len(train_loader):.4f}')

# Test
def test():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

# Run
print("Training VGG16 on CIFAR-10...")
train()
print("Testing...")
test()
```

---

## **5. Training Details (Original Paper)**

| Hyperparameter | Value |
|---------------|-------|
| **Optimizer** | SGD + momentum (0.9) |
| **LR** | 0.01 → reduce by 10× when val acc plateaus |
| **Weight Decay** | 5×10⁻⁴ |
| **Batch Size** | 256 |
| **Init** | Random from N(0, 0.01) |
| **Data Aug** | Random crop, flip |
| **Scale Jittering** | Train on 256–512, test on 224, 256, 384, etc. |

---

## **6. Pretrained VGG in PyTorch**

```python
import torchvision.models as models

# VGG16 with BatchNorm (modernized)
model = models.vgg16_bn(pretrained=True)

# VGG16 (original)
model = models.vgg16(pretrained=True)

# Change classifier for new task
model.classifier[6] = nn.Linear(4096, 10)  # for CIFAR-10
```

---

## **7. Performance**

| Model | Top-1 | Top-5 | Params |
|-------|-------|--------|--------|
| **VGG16** | 71.5% | **92.7%** | 138M |
| **VGG19** | 72.4% | 91.0% | 143M |
| **AlexNet** | 63.3% | 84.7% | 60M |

---

## **8. Pros & Cons**

| Pros | Cons |
|------|------|
| Simple, uniform design | **138M parameters** → huge |
| Excellent feature extractor | **Slow inference** |
| Great for transfer learning | FC layers dominate params |
| Small filters → efficient receptive field | No residual connections |

---

## **9. Parameter Distribution (Chart.js)**
<img width="983" height="562" alt="image" src="https://github.com/user-attachments/assets/eb1c802a-c231-45e7-bf1d-1efa6f8a1208" />


> **~90% of params in FC layers!**

---

## **10. Modern Fixes**

| Issue | Fix |
|------|-----|
| Too many params | Replace FC with **Global Average Pooling** (Network in Network, ResNet) |
| Slow | Use **depthwise separable conv** (MobileNet) |
| Vanishing gradients | Add **skip connections** (ResNet) |

---

## **11. Architecture Diagram (Text)**

```
Input (224×224×3)
│
├─► [64,64] → MaxPool → 112×112×64
├─► [128,128] → MaxPool → 56×56×128
├─► [256,256,256] → MaxPool → 28×28×256
├─► [512,512,512] → MaxPool → 14×14×512
├─► [512,512,512] → MaxPool → 7×7×512
│
├─► Flatten → 25088
├─► FC → 4096 → Dropout
├─► FC → 4096 → Dropout
└─► FC → 1000 → Softmax
```

---

## **12. Run the Code**

```bash
pip install torch torchvision
python vgg16.py
```

> Expected: **~92–94% on CIFAR-10** after 50 epochs.

---

## **13. Summary**

| Feature | Value |
|--------|-------|
| **Year** | 2014 |
| **Authors** | Simonyan & Zisserman |
| **Depth** | 16–19 layers |
| **Filter Size** | Only 3×3 |
| **Params** | 138M (VGG16) |
| **Input** | 224×224×3 |
| **Legacy** | Proved **depth > width** |

---



