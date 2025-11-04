# **GoogLeNet (Inception v1): Complete Guide**  
### *Winner of ILSVRC 2014 — Top-5 Error: 6.67%*  
> **Paper**: *"Going Deeper with Convolutions"* (Szegedy et al., 2015)  
> **Authors**: Google (Christian Szegedy, Wei Liu, et al.)  
> **Key Idea**: **Inception Module** — multi-scale feature extraction in parallel

---

## **1. Core Innovations of GoogLeNet**

| Innovation | Description |
|----------|-------------|
| **Inception Module** | Parallel 1×1, 3×3, 5×5 conv + pooling → captures multi-scale features |
| **1×1 Convolutions** | **Dimensionality reduction** → reduces computation |
| **Global Average Pooling** | Replaces FC layers → **huge parameter reduction** |
| **22 Layers Deep** | But **~12× fewer parameters than AlexNet** |
| **Auxiliary Classifiers** | 2 side branches for **gradient flow during training** |
| **No Fully Connected Layers** | Only **~6M parameters** vs AlexNet’s 60M |

Sure! Let’s make it very simple to understand the **Inception Module** in **GoogLeNet** 😊

---

## 🎯 What Problem Did Inception Solve?

Before Inception, CNNs used one filter size at a time (like only 3×3).
But what if some image features are tiny, while others are large?

✅ We want the network to **look at the image in multiple ways**
→ small details + big shapes
✅ Without making the model extremely heavy

---

## 🧠 What Is an Inception Module?

Inception module **applies multiple filter sizes in parallel**:

* 1×1 filters → capture simple/local info + reduce channels
* 3×3 filters → capture medium features
* 5×5 filters → capture bigger patterns
* Max pooling → highlights strongest features

Then, it **concatenates** all the outputs together.

---

## 🍱 A Simple Analogy

Think of an image like a **scene**…

| Filter  | What it notices                   |
| ------- | --------------------------------- |
| 1×1     | Color/local pixel detail          |
| 3×3     | Edges, corners, medium textures   |
| 5×5     | Bigger shapes like faces, objects |
| Pooling | Most important/highest signals    |

Instead of choosing one, Inception says:

> “Why not use all — and let the network decide what’s important?”

---

## 🔍 A Very Simple Diagram

```
        Input
          |
 ------------------------------------------------
 |      |         |          |                  |
1x1   3x3       5x5    Max Pooling + 1x1 conv
 |      |         |          |
 ------------------------------
          Concatenate
                 |
              Output
```

---

## 🧩 Key Trick: 1×1 Convolution

Large filters (like 5×5) are expensive!
So by first applying **1×1 Conv**, we reduce the number of channels → fewer parameters.

Example:
Instead of using 5×5 filters on 256 channels → expensive
Use 1×1 conv to compress 256 → 32 channels → huge savings!

➡️ Efficient + powerful ✔️

---

## ✅ Why Inception Was a Big Deal

| Benefit                      | What it means                           |
| ---------------------------- | --------------------------------------- |
| Multi-scale feature learning | Detects small + large patterns together |
| Memory + compute efficient   | Faster, lighter than very deep CNNs     |
| Improved accuracy            | Helped GoogLeNet win ImageNet 2014      |

---

## 🔑 One-Sentence Summary

> Inception module looks at the same image through multiple filter sizes at once and merges the best information, making GoogLeNet both smart and efficient.

---

---

## **2. GoogLeNet (Inception v1) Architecture**

| Layer | Type | Output Size | #1×1 | #3×3 reduce | #3×3 | #5×5 reduce | #5×5 | Pool proj | Total Params |
|-------|------|-------------|------|-------------|------|-------------|------|-----------|--------------|
| Input | - | 224×224×3 | - | - | - | - | - | - | - |
| conv1 | 7×7, s=2 | 112×112×64 | - | - | - | - | - | - | 9,408 |
| pool1 | 3×3 max, s=2 | 56×56×64 | - | - | - | - | - | - | 0 |
| conv2_red | 1×1 | 56×56×64 | 64 | - | - | - | - | - | 4,096 |
| conv2 | 3×3 | 56×56×192 | - | - | 192 | - | - | - | 307,392 |
| pool2 | 3×3 max, s=2 | 28×28×192 | - | - | - | - | - | - | 0 |
| **Inception (3a)** | | 28×28×256 | 64 | 96 | 128 | 16 | 32 | 32 | **159,936** |
| **Inception (3b)** | | 28×28×480 | 128 | 128 | 192 | 32 | 96 | 64 | **376,320** |
| pool3 | 3×3 max, s=2 | 14×14×480 | - | - | - | - | - | - | 0 |
| **Inception (4a)** | | 14×14×512 | 192 | 96 | 208 | 16 | 48 | 64 | **364,544** |
| **Aux Classifier 1** | | 1000 | - | - | - | - | - | - | **~1M** |
| **Inception (4b)** | | 14×14×512 | 160 | 112 | 224 | 24 | 64 | 64 | **437,952** |
| **Inception (4c)** | | 14×14×512 | 128 | 128 | 256 | 24 | 64 | 64 | **462,592** |
| **Inception (4d)** | | 14×14×528 | 112 | 144 | 288 | 32 | 64 | 64 | **580,544** |
| **Aux Classifier 2** | | 1000 | - | - | - | - | - | - | **~1M** |
| **Inception (4e)** | | 14×14×832 | 256 | 160 | 320 | 32 | 128 | 128 | **840,896** |
| pool4 | 3×3 max, s=2 | 7×7×832 | - | - | - | - | - | - | 0 |
| **Inception (5a)** | | 7×7×832 | 256 | 160 | 320 | 32 | 128 | 128 | **1,071,936** |
| **Inception (5b)** | | 7×7×1024 | 384 | 192 | 384 | 48 | 128 | 128 | **1,382,400** |
| pool5 | 7×7 avg | 1×1×1024 | - | - | - | - | - | - | 0 |
| dropout | 40% | 1×1×1024 | - | - | - | - | - | - | 0 |
| linear | FC | 1×1×1000 | - | - | - | - | - | - | 1,024,000 |
| softmax | | 1000 | - | - | - | - | - | - | 0 |

> **Total Parameters**: **~6.8 million**  
> **Depth**: **22 layers** (with pooling)  
> **Input**: **224×224×3**

---

## **3. The Inception Module (Core Idea)**

```
Input
│
├── 1×1 conv → (cheap features)
├── 1×1 conv → 3×3 conv
├── 1×1 conv → 5×5 conv
└── 3×3 max pool → 1×1 conv
│
└─► Concatenate along channel dimension
```

### **Why 1×1 Convolutions?**
- **Reduce depth** before expensive 3×3/5×5 → **bottleneck**
- Example:  
  Input: 192 channels → 1×1 conv → 32 channels → 5×5 conv → 32 channels  
  vs  
  192 → 5×5 → 32 = **25× more computation**

---

## **4. Auxiliary Classifiers (Training Only)**

Two side branches:
- After **Inception (4a)** and **(4d)**
- Structure:
  - AvgPool 5×5, s=3 → 4×4
  - 1×1 conv → 128 filters
  - FC → 1024
  - Dropout 70%
  - FC → 1000 (softmax)

**Purpose**: Combat vanishing gradients in deep nets  
**Weight**: Loss = `main_loss + 0.3 * (aux1_loss + aux2_loss)`  
**Removed during inference**

---

## **5. Full PyTorch Implementation (GoogLeNet)**

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
learning_rate = 0.05
num_epochs = 100
aux_weight = 0.3  # Only during training

# Data
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

# Inception Module
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], 1)

# Auxiliary Classifier
class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# GoogLeNet
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)
            self.aux2 = AuxClassifier(528, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)

        aux1 = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)

        aux2 = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.e4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.aux_logits and self.training:
            return x, aux1, aux2
        return x

# Model
model = GoogLeNet(num_classes=num_classes, aux_logits=True).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            if model.aux_logits:
                outputs, aux1, aux2 = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux1, labels)
                loss3 = criterion(aux2, labels)
                loss = loss1 + aux_weight * (loss2 + loss3)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item():.4f}')

        scheduler.step()

# Test
def test():
    model.eval()
    model.aux_logits = False
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
print("Training GoogLeNet on CIFAR-10...")
train()
print("Testing...")
test()
```

---

## **6. Pretrained GoogLeNet**

```python
import torchvision.models as models
model = models.googlenet(pretrained=True)
```

---

## **7. Performance**

| Model | Params | Top-5 Error (ImageNet) |
|-------|--------|-----------------------|
| **GoogLeNet** | **6.8M** | **6.67%** |
| VGG16 | 138M | 7.3% |
| AlexNet | 60M | 15.3% |

---

## **8. Parameter Distribution (Chart.js)**

<img width="965" height="539" alt="image" src="https://github.com/user-attachments/assets/8bb42de8-e06f-4845-998b-41c13853d0ed" />


---

## **9. Pros & Cons**

| Pros | Cons |
|------|------|
| Only **6.8M params** | Complex architecture |
| Multi-scale features | Auxiliary losses add complexity |
| Global Avg Pooling | Hard to modify |
| State-of-the-art in 2014 | Slower than VGG (many small layers) |

---

## **10. Evolution**

| Version | Year | Key Change |
|--------|------|-----------|
| **Inception v1** | 2014 | 1×1, 3×3, 5×5 |
| **v2/v3** | 2015 | Factorized 5×5 → 3×3, BatchNorm |
| **v4** | 2016 | Residual connections |
| **Inception-ResNet** | 2016 | ResNet + Inception |

---

## **11. Summary**

| Feature | Value |
|--------|-------|
| **Year** | 2014 |
| **Winner** | ILSVRC 2014 |
| **Params** | **6.8M** |
| **Depth** | 22 layers |
| **Key Module** | Inception |
| **Innovation** | 1×1 conv + Global Avg Pool |
| **Legacy** | Efficient deep networks |

---

**GoogLeNet proved: "Deeper doesn’t mean more parameters."**

raining**

Happy coding!
