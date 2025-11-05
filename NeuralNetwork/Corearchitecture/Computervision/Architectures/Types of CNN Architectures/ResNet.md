# **ResNet (Residual Network): Complete Guide**  
### *Winner of ILSVRC 2015 — Top-1 Error: 3.57%*  
> **Paper**: *"Deep Residual Learning for Image Recognition"* (He et al., 2015)  
> **Authors**: Microsoft Research (Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun)  
> **Key Idea**: **Residual Learning** — "Skip Connections" solve vanishing gradients

---

## **1. Core Innovation: Residual Block**

### **Problem with Deep Networks**
- **Vanishing/Exploding Gradients**
- **Degradation Problem**: Accuracy **decreases** with more layers (e.g., 56-layer > 20-layer)

### **Solution: Residual Connection**
```python
y = F(x) + x   # instead of y = F(x)
```
- **Identity Shortcut**: `x` is added directly to output
- **Residual Block**: Learn `F(x) = y - x` (the *residual*)

> **"Let the network learn the difference, not the whole mapping."**

---

## **2. Residual Block Types**

| Type | Structure | Used In |
|------|---------|--------|
| **Basic Block** | 3×3 → 3×3 | ResNet-18, ResNet-34 |
| **Bottleneck Block** | 1×1 → 3×3 → 1×1 | ResNet-50, 101, 152 |

### **Bottleneck Design (for deeper nets)**
```
Input
│
├─ 1×1 conv (reduce dim)
├─ 3×3 conv
└─ 1×1 conv (restore dim)
│
+──► Identity (or 1×1 projection if dims differ)
│
Output
```

→ **Fewer parameters**, **faster**

---

## **3. ResNet Variants**

| Model | Layers | Params | Top-1 (ImageNet) | Top-5 |
|-------|--------|--------|------------------|-------|
| **ResNet-18** | 18 | 11.7M | 69.8% | 89.1% |
| **ResNet-34** | 34 | 21.8M | 73.3% | 91.4% |
| **ResNet-50** | 50 | **25.6M** | **76.1%** | **93.0%** |
| **ResNet-101** | 101 | 44.5M | 77.4% | 93.6% |
| **ResNet-152** | 152 | 60.2M | **78.3%** | **94.1%** |

> **ResNet-50** is the **sweet spot** (accuracy vs speed)

---

## **4. ResNet-50 Architecture (Detailed)**

| Layer | Output Size | Block | Repeat | Stride |
|-------|-------------|-------|--------|--------|
| conv1 | 112×112 | 7×7, 64 | 1 | 2 |
| pool1 | 56×56 | 3×3 max pool | 1 | 2 |
| **conv2_x** | 56×56 | Bottleneck | **3** | 1 |
| **conv3_x** | 28×28 | Bottleneck | **4** | 2 (first) |
| **conv4_x** | 14×14 | Bottleneck | **6** | 2 (first) |
| **conv5_x** | 7×7 | Bottleneck | **3** | 2 (first) |
| pool | 1×1 | Global Avg Pool | - | - |
| fc | 1000 | Fully Connected | - | - |

> **Total Blocks**: 3 + 4 + 6 + 3 = **16** → 16 × 3 = **48** conv layers + conv1 + fc = **50 layers**

---

## **5. Full PyTorch Implementation (ResNet-50)**

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
batch_size = 128
learning_rate = 0.1
num_epochs = 100
weight_decay = 1e-4

# Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Bottleneck Block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ResNet-50
def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

model = resnet50(num_classes=num_classes).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

# Training
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
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
        print(f'Epoch {epoch+1} - Avg Loss: {running_loss/len(trainloader):.4f}')

# Test
def test():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

# Run
print("Training ResNet-50 on CIFAR-10...")
train()
print("Testing...")
test()
```


---

## 🧠 The Main Idea of ResNet

> “Instead of learning the output directly, let’s make the network learn the *difference* between the input and the output.”

That “difference” is called a **residual**.
Hence the name — **Residual Network (ResNet).**

---

## ⚙️ The Problem Before ResNet

When CNNs became deeper (like 20, 50, 100+ layers):

* Training got **harder**
* Gradients **vanished or exploded**
* Deeper networks sometimes performed **worse** than shallow ones 😩

This was weird — adding more layers should increase power, right?
But deeper networks couldn’t easily learn identity mappings.

---

## 🪄 The Solution — Skip Connection

ResNet introduced a simple trick:
**“Skip” one or more layers by adding the input directly to the output.**

### Diagram of a Residual Block

```
Input (x)
   │
[Conv → BN → ReLU → Conv → BN]
   │
 + └──────────────┐
   ↓              ↓
  Add ----------> Output
   ↓
  ReLU
```

### Mathematically:

[
y = F(x) + x
]

where

* ( F(x) ): output of a few conv layers (the residual)
* ( x ): the original input (shortcut connection)

---

## 🧮 Why Does This Help?

If the optimal function is **identity** (i.e. do nothing, ( y = x )),
normal networks must learn that through weights — hard!
But with ResNet:
[
F(x) = 0 \implies y = x
]
so the network just sets residual to zero → super easy for gradients to flow! ✅

This solves:

* **Vanishing gradient** problem
* **Degradation** (when deeper networks perform worse)

---

## 🏗️ A Simple Residual Block (ResNet-18 or 34)

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If dimensions change, use a 1x1 conv to match shortcut shape
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = self.relu(out)
        return out
```

---

## 🧩 Two Types of Blocks

| Type                 | Used In             | Key Difference                   |
| -------------------- | ------------------- | -------------------------------- |
| **Basic Block**      | ResNet-18, 34       | Two 3×3 convs                    |
| **Bottleneck Block** | ResNet-50, 101, 152 | 1×1 → 3×3 → 1×1 (for efficiency) |

---

## 🧮 Math Behind “Adding” in Residuals

If both ( F(x) ) and ( x ) have shape `[B, C, H, W]`,
the addition is **element-wise**:

[
Y_{b,c,h,w} = F_{b,c,h,w} + X_{b,c,h,w}
]

Each pixel and channel adds up directly — no extra learnable weights.

If shapes differ (like channels change), a 1×1 conv “projects” ( x ) to match.

---

## 🔥 Why ResNet Was a Breakthrough

| Benefit                  | What it Means                                                  |
| ------------------------ | -------------------------------------------------------------- |
| Easier optimization      | Gradients flow through skip path directly                      |
| Deeper networks possible | ResNet trained 152+ layers successfully                        |
| Reusability              | Idea reused in Transformers, UNet, GANs, etc.                  |
| Simpler math             | Learning residuals (differences) instead of absolute functions |

---

## 🧠 Example Intuition

Imagine you’re learning how to draw a face.
You already have a sketch (the input).
You don’t start over — you **refine** it by adding small improvements (residuals).
That’s what ResNet does!

---

## ✅ Quick Summary

| Concept             | Meaning                                      |
| ------------------- | -------------------------------------------- |
| **Residual Block**  | Two conv layers + skip connection            |
| **Skip Connection** | Adds input to output, helps gradient flow    |
| **Math**            | ( y = F(x) + x )                             |
| **Why it works**    | Learns easier: refine input, not recreate it |
| **Result**          | Super deep networks that still train easily  |

---

---

## **6. Pretrained ResNet**

```python
import torchvision.models as models

# ResNet-50 (pretrained on ImageNet)
model = models.resnet50(pretrained=True)

# Change final layer
model.fc = nn.Linear(2048, 10)  # for CIFAR-10
```

---

## **7. Performance Comparison**

| Model | Params | Top-5 Error | Speed |
|-------|--------|-------------|-------|
| **ResNet-50** | 25.6M | **7.0%** | Fast |
| VGG16 | 138M | 7.3% | Slow |
| GoogLeNet | 6.8M | 6.7% | Medium |
| AlexNet | 60M | 15.3% | Fast |

---

## **8. Residual Block Visualization**

```text
Input
│
├───► [1×1, 64] ──► [3×3, 64] ──► [1×1, 256] ──►
│                                              │
│                                              +──► Add
│                                              │
└──────────────────────────────────────────────► Identity
```

---

## **9. Parameter Distribution (Chart.js)**
<img width="990" height="546" alt="image" src="https://github.com/user-attachments/assets/38a83b82-0ead-4850-84cc-3da804250e8c" />

---

## **10. Why ResNet Works**

| Benefit | Explanation |
|-------|-----------|
| **No Degradation** | 152-layer > 34-layer |
| **Easy Optimization** | Gradients flow through shortcuts |
| **Identity Mapping** | If extra layers not needed → learn `F(x)=0` |
| **Modular** | Stack blocks easily |

---

## **11. ResNet Variants**

| Variant | Change |
|-------|--------|
| **ResNet-v2** | Pre-activation (BN-ReLU-Conv) |
| **ResNeXt** | Cardinality (grouped conv) |
| **Wide ResNet** | Wider layers |
| **ResNet-D** | Downsample with avg pool + stride |

---

## **12. Summary**

| Feature | Value |
|--------|-------|
| **Year** | 2015 |
| **Winner** | ILSVRC 2015 |
| **Key Idea** | **Residual Learning** |
| **Block** | Bottleneck (1×1, 3×3, 1×1) |
| **Params** | 25.6M (ResNet-50) |
| **Depth** | Up to 152 layers |
| **Legacy** | **Backbone of modern CV** |

---

## **Run the Code**

```bash
pip install torch torchvision
python resnet50.py
```

> Expected: **~94–95% on CIFAR-10**

---

**ResNet is the foundation of modern deep learning.**  
Used in:
- Object detection (Faster R-CNN)
- Segmentation (Mask R-CNN)
- GANs
- Transformers (ViT uses ResNet backbones)

---

- **Comparison with DenseNet**

Let me know — I’ll build it!
