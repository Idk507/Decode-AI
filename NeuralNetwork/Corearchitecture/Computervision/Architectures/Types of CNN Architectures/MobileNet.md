# **MobileNet (v1 → v3): Complete Guide**  
### *Lightweight, Efficient, and Deployable CNNs for Mobile & Edge Devices*

---

## **Overview**

| Version | Year | Key Innovation | Top-1 (ImageNet) | Params | FLOPs |
|--------|------|----------------|------------------|--------|-------|
| **MobileNet-v1** | 2017 | **Depthwise Separable Conv** | 70.6% | **4.2M** | 0.57B |
| **MobileNet-v2** | 2018 | **Inverted Residuals + Linear Bottleneck** | 72.0% | **3.4M** | 0.30B |
| **MobileNet-v3** | 2019 | **Neural Architecture Search (NAS) + h-swish + Squeeze-and-Excitation** | **75.2%** | **5.4M** | 0.22B |

> **MobileNet = Accuracy × Efficiency**  
> **Designed for**: Phones, IoT, embedded systems

---

## **1. MobileNet-v1: Depthwise Separable Convolution**

### **Standard Conv**
```
3×3×C_in×C_out → 9 × C_in × C_out ops
```

### **Depthwise Separable = 2 Steps**

| Step | Operation | Cost |
|------|---------|------|
| **Depthwise** | 3×3 per channel | `9 × C_in` |
| **Pointwise** | 1×1 to mix | `C_in × C_out` |

> **Total**: `9×C_in + C_in×C_out` → **~8–9× fewer ops**

### **Architecture (v1)**

| Layer | Type | Output | Stride |
|------|------|--------|--------|
| conv1 | 3×3, 32 | 112×112×32 | 2 |
| **DW Block ×13** | DW + PW | ↓ | 1 or 2 |
| Avg Pool | 7×7 | 1×1×1024 | |
| FC | | 1000 | |

> **Width Multiplier (α)**: Scale channels (e.g., α=0.5 → 1M params)  
> **Resolution Multiplier (ρ)**: Input size (e.g., 128×128)

---

## **2. MobileNet-v2: Inverted Residuals + Linear Bottleneck**

### **Problems with v1**
- **ReLU on low-dim → information loss**
- **No skip connections**

### **Inverted Residual Block**

```text
Input (low-dim)
│
├─► 1×1 conv → expand (6×) → ReLU6
├─► 3×3 depthwise → ReLU6
└─► 1×1 conv → compress → **NO ReLU** (linear)
│
+──► Skip (if stride=1 & same dim)
```

> **Expansion → Depthwise → Projection (linear)**

### **Why Linear Bottleneck?**
- ReLU on low-dim tensors → **destroys information**
- Keep output **linear** → preserves features

### **Architecture (v2)**

| Layer | t (expand) | c (out) | n (repeat) | s |
|------|------------|---------|------------|---|
| conv1 | - | 32 | 1 | 2 |
| bottleneck | 1 | 16 | 1 | 1 |
| bottleneck | 6 | 24 | 2 | 2 |
| bottleneck | 6 | 32 | 3 | 2 |
| bottleneck | 6 | 64 | 4 | 2 |
| bottleneck | 6 | 96 | 3 | 1 |
| bottleneck | 6 | 160 | 3 | 2 |
| bottleneck | 6 | 320 | 1 | 1 |
| conv2 | - | 1280 | 1 | 1 |
| AvgPool + FC | | 1000 | | |

> **Fewer params (3.4M)**, **faster**, **better accuracy**

---

## **3. MobileNet-v3: NAS + h-swish + SE**

### **Neural Architecture Search (NAS)**
- Used **MnasNet-style platform-aware NAS**
- Optimize for **latency**, not just FLOPs


---

# 🧠 Why MobileNet Exists

After **Xception**, people realized:

> Depthwise separable convolutions are super efficient —
> so why not build a *whole CNN* around them?

That’s what **MobileNet** does.
It’s designed to run **on phones, embedded devices, and edge systems** — fast, light, and low power 🔋.

---

# 🚀 MOBILE NET v1 (2017)

### 🧩 The Core Idea — “Depthwise Separable Convolution”

Instead of using normal convolution, every conv layer is split into:
1️⃣ **Depthwise convolution:** looks at each channel separately
2️⃣ **Pointwise (1×1) convolution:** combines them

This is **exactly like Xception**, but used systematically.

---

### ⚙️ Structure

```
Input
↓
Conv 3×3 (normal)
↓
Depthwise Conv + Pointwise Conv → ReLU6
↓
Depthwise Conv + Pointwise Conv → ReLU6
↓
... repeat ...
↓
AvgPool + FC → Softmax
```

---

### 💡 Two Hyperparameters

| Parameter                 | Meaning                    | Example               |
| ------------------------- | -------------------------- | --------------------- |
| α (width multiplier)      | Shrinks number of channels | 0.75× or 0.5× smaller |
| ρ (resolution multiplier) | Shrinks input size         | 224×224 → 160×160     |

✅ Trade accuracy for speed & size
✅ Used to tune model for devices

---

### 🧮 Efficiency

| Operation | Normal Conv | Depthwise Separable | Reduction   |
| --------- | ----------- | ------------------- | ----------- |
| Params    | (k×k×M×N)   | (k×k×M + M×N)       | ~9× fewer   |
| FLOPs     | High        | Very low            | 8–9× faster |

---

### 🔋 Results

| Model            | Accuracy (ImageNet) | Params | Speed  |
| ---------------- | ------------------- | ------ | ------ |
| VGG16            | 71%                 | 138M   | Slow   |
| **MobileNet-v1** | 70–71%              | 4.2M   | ⚡ Fast |

✅ Runs easily on mobile CPUs
✅ Simple, lightweight backbone

---

# ⚙️ MOBILE NET v2 (2018)

Google refined the design to fix some v1 limitations.

---

## 🧩 Problem in v1

Depthwise convs were great — but as we made the network deeper, features became **too compressed** (information lost).

---

## 💡 Solution: “Inverted Residuals + Linear Bottleneck”

Let’s break that down simply 👇

---

### 1️⃣ Normal Residual Block (ResNet style)

```
Input → Conv → Conv → Add(Input)
```

### 2️⃣ MobileNet-v2 Block (Inverted Residual)

```
Input → 1×1 (expand channels) → Depthwise Conv → 1×1 (project down) → Add(Input)
```

So it first **expands**, then **compresses** — the *opposite* of ResNet (hence “inverted”).

---

### 🧮 Linear Bottleneck

Normally, CNNs end layers with ReLU activation.
But when compressing (reducing channels), **ReLU kills information**.
So v2 keeps the **bottleneck layer linear** (no ReLU there).

That’s why it’s called **Linear Bottleneck** — preserves fine details.

---

### 🧠 Intuitive Example

Imagine squeezing toothpaste (features) out of a tube (bottleneck).
If you apply a ReLU (cutting negative values), half the paste is gone!
So MobileNet-v2 doesn’t squeeze too hard and avoids losing info.

---

### 🧩 MobileNet-v2 Block

```
Input (small channels)
↓
1×1 Conv → Expand (6× channels)
↓
Depthwise Conv
↓
1×1 Conv → Project back (Linear)
↓
Add skip connection (if same size)
```

---

### ✅ Why It’s Better

| Feature      | MobileNet-v1        | MobileNet-v2         |
| ------------ | ------------------- | -------------------- |
| Block type   | Depthwise separable | Inverted residual    |
| ReLU usage   | Everywhere          | Linear at bottleneck |
| Feature loss | High                | Preserves details    |
| Speed        | Fast                | Fast                 |
| Accuracy     | ~71%                | ~74%                 |

---

# 🚀 MOBILE NET v3 (2019)

MobileNet-v3 is the **best** and **smartest** version.
It was designed using **Neural Architecture Search (NAS)** — meaning Google’s AI *automatically discovered the best design*.

---

## 💡 New Additions

1️⃣ **SE Blocks (Squeeze-and-Excitation)**
→ Adds a small attention mechanism
→ Learns which channels are important
→ "Channel attention"

2️⃣ **Swish / h-swish Activation**
→ Smoother than ReLU
→ Helps gradients flow better

[
h_swish(x) = x * ReLU6(x+3) / 6
]

3️⃣ **Smarter block design**
→ Uses both v1 and v2 ideas
→ Mix of 3×3 and 5×5 depthwise convs
→ SE + Linear bottleneck in every block

---

### 🧩 Block Example (Simplified)

```
Input
↓
1×1 Conv → Expand
↓
Depthwise Conv (3×3 or 5×5)
↓
SE Block (channel attention)
↓
1×1 Conv → Project (Linear)
↓
h-swish Activation
```

---

### 🧠 Two Versions

| Model                  | Optimized For     | Used In                  |
| ---------------------- | ----------------- | ------------------------ |
| MobileNet-v3 **Small** | Low-power devices | Phones, microcontrollers |
| MobileNet-v3 **Large** | High-performance  | Cloud + Mobile GPUs      |

---

### 📊 Performance

| Model                  | Params | Top-1 Acc  | Speed       |
| ---------------------- | ------ | ---------- | ----------- |
| MobileNet-v1           | 4.2M   | 71%        | ⚡ Fast      |
| MobileNet-v2           | 3.4M   | 74%        | ⚡⚡ Faster   |
| **MobileNet-v3 Large** | 5.4M   | **76–78%** | ⚡⚡⚡ Fastest |

✅ Combines Xception-like efficiency
✅ Adds ResNet-style skip connections
✅ Adds SE and better activations

---

# 🧩 Evolution Summary

| Version | Core Idea                              | Key Feature      | Analogy                   |
| ------- | -------------------------------------- | ---------------- | ------------------------- |
| **v1**  | Depthwise separable conv               | Simplicity       | “Lightweight Inception”   |
| **v2**  | Inverted residuals + Linear bottleneck | Better info flow | “ResNet-style efficiency” |
| **v3**  | NAS + SE + h-swish                     | Smart & adaptive | “AI-optimized hybrid”     |

---

# 🧠 Simple Analogy Recap

| Model            | You Can Think Of It As          |
| ---------------- | ------------------------------- |
| **Inception**    | Looking at image in many ways   |
| **Xception**     | Doing it smartly — per channel  |
| **MobileNet-v1** | Doing it smartly *and fast*     |
| **MobileNet-v2** | Fast, but avoids losing details |
| **MobileNet-v3** | Fast + smart + AI-tuned         |

---

# ✅ TL;DR Summary

| Version | Main Innovation                        | Why It Matters           |
| ------- | -------------------------------------- | ------------------------ |
| **v1**  | Depthwise separable conv               | Huge speed-up            |
| **v2**  | Inverted residuals + linear bottleneck | Preserve info            |
| **v3**  | SE + h-swish + NAS                     | Smarter, higher accuracy |

---




### **Key Innovations**

| Innovation | Description |
|----------|-----------|
| **h-swish** | `x * ReLU6(x+3)/6` → better than ReLU6 |
| **Squeeze-and-Excitation (SE)** | Channel attention |
| **Hard-swish & Hard-sigmoid** | Quantization-friendly |
| **Redesigned Layers** | Remove expensive early layers |
| **Last Stage Redesign** | 1×1 → 5×5 → SE |

### **Architecture (v3-Large)**

| Layer | Operator | Exp | Out | SE | NL | s |
|------|----------|-----|-----|----|----|---|
| conv | conv2d | - | 16 | - | H | 2 |
| bneck | bneck, 3×3 | 16 | 16 | - | R | 1 |
| bneck | bneck, 3×3 | 64 | 24 | - | R | 2 |
| bneck | bneck, 3×3 | 72 | 24 | - | R | 1 |
| bneck | bneck, 5×5 | 72 | 40 | Yes | R | 2 |
| bneck | bneck, 5×5 | 120 | 40 | Yes | R | 1 |
| bneck | bneck, 5×5 | 240 | 80 | - | H | 2 |
| bneck | bneck, 3×3 | 200 | 80 | - | H | 1 |
| bneck | bneck, 3×3 | 184 | 80 | - | H | 1 |
| bneck | bneck, 3×3 | 184 | 80 | - | H | 1 |
| bneck | bneck, 3×3 | 480 | 112 | Yes | H | 1 |
| bneck | bneck, 5×5 | 672 | 160 | Yes | H | 2 |
| bneck | bneck, 5×5 | 960 | 160 | Yes | H | 1 |
| conv | 1×1 | - | 960 | - | H | 1 |
| pool + conv | | 1280 | | | |
| conv | 1×1 | - | k | - | - | |

> **NL**: `R` = ReLU, `H` = h-swish

---

## **4. Full PyTorch Code: MobileNet-v3 (Small & Large)**

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
num_classes = 10
batch_size = 128
learning_rate = 0.05
num_epochs = 100

# Data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# h-swish
class HSwish(nn.Module):
    def forward(self, x):
        return x * nn.functional.relu6(x + 3, inplace=True) / 6

# SE Module
class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        return x * self.fc(x)

# MobileNet-v3 Block
class MBConv(nn.Module):
    def __init__(self, in_c, out_c, exp_c, kernel, stride, se, nl):
        super().__init__()
        self.use_se = se
        self.use_skip = stride == 1 and in_c == out_c

        # Expand
        if exp_c != in_c:
            self.expand = nn.Sequential(
                nn.Conv2d(in_c, exp_c, 1, bias=False),
                nn.BatchNorm2d(exp_c),
                HSwish() if nl == 'H' else nn.ReLU6(inplace=True)
            )
        else:
            self.expand = None

        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(exp_c, exp_c, kernel, stride, kernel//2, groups=exp_c, bias=False),
            nn.BatchNorm2d(exp_c),
            HSwish() if nl == 'H' else nn.ReLU6(inplace=True)
        )

        # SE
        if self.use_se:
            self.se = SEModule(exp_c)

        # Project
        self.project = nn.Sequential(
            nn.Conv2d(exp_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        out = x
        if self.expand:
            out = self.expand(out)
        out = self.depthwise(out)
        if self.use_se:
            out = self.se(out)
        out = self.project(out)
        if self.use_skip:
            out = out + x
        return out

# MobileNet-v3
class MobileNetV3(nn.Module):
    def __init__(self, mode='large', num_classes=10):
        super().__init__()
        configs = {
            'small': [
                # k, exp, out, se, nl, s
                [3, 16, 16, True, 'R', 2],
                [3, 72, 24, False, 'R', 2],
                [3, 88, 24, False, 'R', 1],
                [5, 96, 40, True, 'H', 2],
                [5, 240, 40, True, 'H', 1],
                [5, 240, 40, True, 'H', 1],
                [5, 120, 48, True, 'H', 1],
                [5, 144, 48, True, 'H', 1],
                [5, 288, 96, True, 'H', 2],
                [5, 576, 96, True, 'H', 1],
                [5, 576, 96, True, 'H', 1],
            ],
            'large': [
                [3, 16, 16, False, 'R', 1],
                [3, 64, 24, False, 'R', 2],
                [3, 72, 24, False, 'R', 1],
                [5, 72, 40, True, 'R', 2],
                [5, 120, 40, True, 'R', 1],
                [5, 120, 40, True, 'R', 1],
                [3, 240, 80, False, 'H', 2],
                [3, 200, 80, False, 'H', 1],
                [3, 184, 80, False, 'H', 1],
                [3, 184, 80, False, 'H', 1],
                [3, 480, 112, True, 'H', 1],
                [3, 672, 112, True, 'H', 1],
                [5, 672, 160, True, 'H', 2],
                [5, 960, 160, True, 'H', 1],
                [5, 960, 160, True, 'H', 1],
            ]
        }

        cfg = configs[mode]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            HSwish()
        )

        in_c = 16
        self.blocks = nn.ModuleList()
        for k, exp, out, se, nl, s in cfg:
            self.blocks.append(MBConv(in_c, out, exp, k, s, se, nl))
            in_c = out

        self.head = nn.Sequential(
            nn.Conv2d(in_c, 960 if mode == 'large' else 576, 1, bias=False),
            nn.BatchNorm2d(960 if mode == 'large' else 576),
            HSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960 if mode == 'large' else 576, 1280 if mode == 'large' else 1024, 1),
            HSwish(),
            nn.Conv2d(1280 if mode == 'large' else 1024, num_classes, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x.squeeze(-1).squeeze(-1)

# Model
model = MobileNetV3(mode='large', num_classes=num_classes).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training & Test
def train():
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

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

print("Training MobileNet-v3 Large...")
train()
print("Testing...")
test()
```

---

## **5. Pretrained Models (PyTorch)**

```python
import torchvision.models as models

# MobileNet-v2
model = models.mobilenet_v2(pretrained=True)

# MobileNet-v3
model = models.mobilenet_v3_large(pretrained=True)
model = models.mobilenet_v3_small(pretrained=True)
```

---

## **6. Performance Summary**

| Model | Params | FLOPs | Top-1 | Latency (ms) |
|------|--------|-------|-------|--------------|
| **v1** | 4.2M | 0.57B | 70.6% | ~20 |
| **v2** | 3.4M | 0.30B | 72.0% | ~15 |
| **v3-Large** | 5.4M | 0.22B | **75.2%** | **~12** |
| **v3-Small** | 2.9M | 0.06B | 67.4% | ~8 |

---

## **7. Efficiency Chart (Chart.js)**
<img width="1014" height="557" alt="image" src="https://github.com/user-attachments/assets/2a1156b8-7a9c-4b15-8df4-f06cecfdc5d0" />

---

## **8. Summary**

| Version | Best For | Key Idea |
|--------|---------|---------|
| **v1** | First mobile CNN | Depthwise Separable |
| **v2** | Efficiency + Accuracy | Inverted Residuals |
| **v3** | State-of-the-art mobile | NAS + h-swish + SE |

---

## **Run Code**

```bash
pip install torch torchvision
python mobilenet_v3.py
```

> Expected: **~95%+ on CIFAR-10**

---

**MobileNet is the backbone of on-device AI**  
Used in:
- **TensorFlow Lite**
- **Edge TPU**
- **Android Neural Networks API**
- **Object detection (SSD)**
- **Segmentation**

---
