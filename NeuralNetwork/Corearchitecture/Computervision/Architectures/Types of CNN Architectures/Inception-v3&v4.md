
### *Inception Evolution — From GoogLeNet to State-of-the-Art Efficiency*

---

## **Overview**

| Model | Year | Key Innovations | Top-5 Error (ImageNet) | Params |
|------|------|------------------|------------------------|--------|
| **Inception-v1 (GoogLeNet)** | 2014 | 1×1, 3×3, 5×5 parallel | 6.67% | 6.8M |
| **Inception-v2/v3** | 2015 | Factorized convolutions, BatchNorm, Label Smoothing | **5.6%** | ~25M |
| **Inception-v4** | 2016 | Residual + Inception, Stem redesign | **4.9%** | ~43M |

> **Inception-v3** = *Efficient + Accurate*  
> **Inception-v4** = *Inception + Residual = Best of both worlds*

---

## **1. Inception-v3: Key Innovations**

| Innovation | Description |
|----------|-----------|
| **Factorized 7×7** | Replace 7×7 → two 3×3 → **75% fewer params** |
| **Asymmetric Factorization** | 5×5 → 1×5 + 5×1 → **33% cheaper** |
| **Efficient Grid Reduction** | Parallel pooling + stride conv → no info loss |
| **Batch Normalization** | After every conv |
| **Label Smoothing** | Regularization: `y = 0.9 * true + 0.1 / K` |
| **Auxiliary Classifier** | Removed (not needed with BN) |

---

## **2. Inception-v3 Architecture**

| Layer | Type | Output | Details |
|------|------|--------|-------|
| **Stem** | | 35×35×384 | 3×3 → 3×3 → 3×3 + parallel paths |
| **Inception-A** ×4 | | 35×35×384 | 1×1, 3×3, 5×5 |
| **Reduction-A** | | 17×17×1024 | Grid reduction |
| **Inception-B** ×7 | | 17×17×1024 | Asymmetric + factorized |
| **Reduction-B** | | 8×8×1536 | Pool + stride |
| **Inception-C** ×3 | | 8×8×1536 | 1×1, 1×3+3×1 |
| **Global Avg Pool** | | 1×1×2048 | |
| **FC + Softmax** | | 1000 | |

> **Total**: **42 layers**, **~25M params**

---

## **3. Inception-v4: Residual + Inception**

### **Inception-ResNet Block**
```text
Input
│
├── 1×1
├── 1×1 → 3×3
├── 1×1 → 3×3 → 3×3
│
└─► Concat → 1×1 (scale) → + Input
```

### **Inception-v4 vs v3**

| Feature | v3 | v4 |
|-------|----|----|
| Residual Connections | No | Yes |
| Stem | Complex | Cleaner |
| Depth | 42 | 75 |
| Params | 25M | 43M |
| Top-5 Error | 5.6% | **4.9%** |

---

## **4. Full PyTorch Code: Inception-v3 (from `torchvision`)**

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
batch_size = 64
learning_rate = 0.045  # As in paper
num_epochs = 100
label_smoothing = 0.1

# Data (CIFAR-10 → 224×224)
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

# Load Inception-v3
model = torchvision.models.inception_v3(pretrained=False, aux_logits=True, num_classes=num_classes)
model = model.to(device)

# Label Smoothing Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        n_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            if model.aux_logits:
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
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

print("Training Inception-v3 on CIFAR-10...")
train()
print("Testing...")
test()
```
Excellent — now you’re entering the **evolved Inception family**, where Google took the original Inception (from GoogLeNet, 2014) and **made it faster, deeper, and smarter** using clever factorization and architectural tweaks.

Let’s go step-by-step 👇
You’ll understand both **Inception-v3** and **Inception-v4** — simply, logically, and with enough depth to connect to real implementation or interview discussion.

---

# 🧠 Why Inception Needed to Evolve

Original **GoogLeNet (Inception-v1)** was brilliant — it captured **multi-scale features** efficiently.
But as models got deeper:

* Computation grew heavy
* Some filters (like 5×5) became too expensive
* Gradient flow could still improve

So, Inception-v2 → v3 → v4 were **progressive refinements** to fix these issues.

---

# 🚀 INCEPTION-v2 / v3 — “Factorization” and Efficiency

## 🎯 Main Goal:

Make convolutions **faster and more efficient**
without losing accuracy or depth.

---

## 1️⃣ **Factorization of Convolutions**

Instead of a 5×5 convolution (expensive), they replaced it with two 3×3 convolutions.

### Why?

Parameter cost:

* 5×5 with (C_{in} = C_{out} = 256):
  (256 × 256 × 5 × 5 = 1.6M)
* Two 3×3 layers:
  (256 × 256 × 3 × 3 × 2 = 1.18M)

✅ **Fewer parameters, same receptive field**

---

## 2️⃣ **Asymmetric Convolutions**

Instead of a 3×3, use **1×3 followed by 3×1**.

[
3×3 \Rightarrow (1×3) + (3×1)
]

### Why?

* 3×3: 9 parameters per input-output channel pair
* 1×3 + 3×1: 6 total parameters
  ✅ 33% fewer parameters
  ✅ More nonlinearity (ReLU between them)

---

## 3️⃣ **Batch Normalization Everywhere**

Every conv layer is followed by BN → ReLU
→ stabilizes and speeds up training.

---

## 4️⃣ **Auxiliary Classifiers = Regularization**

Same as v1, but improved with BN.

---

## 🧩 Inception-v3 Block Types

Inception-v3 organizes the network into **three block families**:

| Block       | Purpose                           | Example Filter Sizes       |
| ----------- | --------------------------------- | -------------------------- |
| Inception-A | Normal multi-scale block          | 1×1, 3×3, 5×5 (factorized) |
| Inception-B | Reduces spatial size (downsample) | 3×3 stride 2               |
| Inception-C | Deeper, uses asymmetric convs     | 1×3, 3×1                   |

---

## 🏗️ Simplified Architecture Overview (Inception-v3)

| Stage               | Block Type  | Output Size (approx.) |
| ------------------- | ----------- | --------------------- |
| Stem                | Conv + Pool | 149×149               |
| Inception-A × 3     | —           | 35×35                 |
| Reduction-A         | Downsample  | 17×17                 |
| Inception-B × 5     | —           | 17×17                 |
| Reduction-B         | Downsample  | 8×8                   |
| Inception-C × 2     | —           | 8×8                   |
| Global AvgPool + FC | —           | 1×1 → Classes         |

---

## 🧮 Mathematical Summary

For each Inception-v3 module:
[
y = \text{concat}(f_{1×1}(x), f_{1×3→3×1}(x), f_{3×3→3×3}(x), f_{pool→1×1}(x))
]
where each ( f ) branch uses BN + ReLU between layers.

---

## ✅ Result

* **~28M parameters**
* **Factorized convolutions**
* **Better gradient flow**
* **State-of-the-art accuracy (ImageNet 2015)**

---

# 🧩 INCEPTION-v4 — When Inception Meets ResNet (2016)

Now Google said —

> “What if we combine the *multi-scale idea of Inception* with the *residual idea of ResNet*?”

Hence two major architectures were born:

* **Inception-v4** — pure Inception, deeper and cleaner
* **Inception-ResNet-v2** — hybrid Inception + residual connections

---

## 🧠 Key Innovations in Inception-v4

1️⃣ **Modular Design (4 building blocks)**

| Block         | Function                       |
| ------------- | ------------------------------ |
| Stem          | Initial convolutions + pooling |
| Inception-A   | Multi-scale feature extraction |
| Inception-B   | Factorized 7×7 convs           |
| Inception-C   | High-dimensional feature mix   |
| Reduction-A/B | Downsampling blocks            |

---

2️⃣ **Deeper and More Symmetrical**

* v4 has **more Inception blocks** than v3
* Every block follows a clean, repeated pattern
* Used **“same” padding** for stable spatial sizes

---

3️⃣ **Stronger Regularization**

* Label smoothing
* Dropout
* BatchNorm
  → makes deeper networks trainable

---

## 🧮 Inception-v4 Block Details

| Block       | Core Design                        |
| ----------- | ---------------------------------- |
| Inception-A | Uses 1×1, 3×3, 5×5 branches        |
| Reduction-A | Mix of stride=2 convs and pooling  |
| Inception-B | Factorized 7×7 convs (1×7 + 7×1)   |
| Reduction-B | Another downsample stage           |
| Inception-C | Final stage with multiple 1×3, 3×1 |

---

## 🧩 Inception-v4 Architecture Summary

| Stage           | Type        | Output Size |
| --------------- | ----------- | ----------- |
| Stem            | Conv + Pool | 149×149     |
| Inception-A × 4 | —           | 35×35       |
| Reduction-A     | ↓           | 17×17       |
| Inception-B × 7 | —           | 17×17       |
| Reduction-B     | ↓           | 8×8         |
| Inception-C × 3 | —           | 8×8         |
| Global AvgPool  | —           | 1×1         |
| FC              | —           | num_classes |

---

## ⚖️ Inception-v4 vs Inception-ResNet-v2

| Feature           | Inception-v4   | Inception-ResNet-v2       |
| ----------------- | -------------- | ------------------------- |
| Skip Connections  | ❌ No           | ✅ Yes (Residual style)    |
| Speed             | Slower         | Faster                    |
| Parameters        | Higher         | Fewer                     |
| Gradient Flow     | Good           | Excellent                 |
| Accuracy          | Very high      | Slightly higher           |
| Design Philosophy | Pure Inception | Hybrid Inception + ResNet |

---

# 🧮 Factorization Example (Math)

Let’s say we have a 7×7 conv on 128 channels.
Original cost:
[
7×7×128×128 = 802,816
]

Factorized as:
[
1×7 + 7×1
\Rightarrow (1×7×128×128) + (7×1×128×128) = 2×114,688 = 229,376
]

✅ ~3.5× fewer parameters
✅ Same receptive field
✅ Extra nonlinearity

That’s the **core mathematical innovation** of Inception-v3/v4.

---

# 🧩 Summary Table — Evolution of Inception

| Version          | Year | Key Idea               | Improvement            |
| ---------------- | ---- | ---------------------- | ---------------------- |
| v1 (GoogLeNet)   | 2014 | Multi-scale filters    | Efficient CNN          |
| v2               | 2015 | Factorized 5×5 → 2×3×3 | Faster, fewer params   |
| v3               | 2015 | Asymmetric convs, BN   | More accurate          |
| v4               | 2016 | Modular deep design    | Deeper, more efficient |
| Inception-ResNet | 2016 | Add residuals          | Combines best of both  |

---

# 🧠 Intuition Recap

| Analogy          | Meaning                                                     |
| ---------------- | ----------------------------------------------------------- |
| Inception-v1     | “Look at everything — small & large patterns.”              |
| Inception-v3     | “Do it faster with smaller filters.”                        |
| Inception-v4     | “Stack it deeper, more organized.”                          |
| Inception-ResNet | “Do all that, but with shortcut safety (skip connections).” |

---

---

# 🧩 **Inception-v3 — Modular Design**

Inception-v3 uses three main block types:

* **Inception-A** → captures multi-scale local features (35×35)
* **Inception-B** → deeper middle layers with asymmetric 1×7, 7×1 convs (17×17)
* **Inception-C** → fine-grained, high-level patterns (8×8)

---

## 🧠 1️⃣ Inception-A Block (Factorized 5×5 → Two 3×3)

```python
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=1),
            nn.Conv2d(48, 64, kernel_size=5, padding=2)  # factorized 5x5 in original
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.Conv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)
```

### ➕ Shape Example:

Input: `35×35×192`
Output channels: 64+64+96+32 = **256**
Output: `35×35×256`

**Why it works:** combines local (1×1), medium (3×3), and large (5×5) receptive fields efficiently.

---

## 🧠 2️⃣ Inception-B Block (Asymmetric Factorization)

```python
class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 192, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=(1,7), padding=(0,3)),
            nn.Conv2d(128, 192, kernel_size=(7,1), padding=(3,0))
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=(7,1), padding=(3,0)),
            nn.Conv2d(128, 128, kernel_size=(1,7), padding=(0,3)),
            nn.Conv2d(128, 192, kernel_size=(7,1), padding=(3,0)),
            nn.Conv2d(192, 192, kernel_size=(1,7), padding=(0,3))
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)
        ], dim=1)
```

### ➕ Shape Example:

Input: `17×17×768`
Output: `17×17×768` (each branch ~192 channels)

**Why asymmetric convs?**
`(1×7)` + `(7×1)` = same receptive field as `7×7`, but
✅ fewer params
✅ more nonlinearities
✅ faster training

---

## 🧠 3️⃣ Reduction Blocks (Downsampling)

To move from one grid size to the next (like 35×35 → 17×17):

```python
class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.Conv2d(96, 96, kernel_size=3, stride=2)
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch1(x), self.branch2(x), self.branch3(x)
        ], dim=1)
```

This replaces the need for a simple pooling layer, preserving multi-path richness even when reducing spatial size.

---

## 🏗️ Full Inception-v3 Outline

```
Input (299x299x3)
↓
Stem (Conv + Pool)
↓
Inception-A × 3
↓
Reduction-A
↓
Inception-B × 5
↓
Reduction-B
↓
Inception-C × 2
↓
AvgPool → FC (1000 classes)
```

✅ ~28M parameters
✅ High accuracy
✅ Efficient training

---

# 🧩 **Inception-v4 — Modular, Deeper, Cleaner**

Now let’s see how **Inception-v4** expanded this concept.

---

## 🧠 1️⃣ Stem Block (Initial Convolutions)

```python
class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),   # 149x149
            nn.Conv2d(32, 32, kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2),                   # 73x73
            nn.Conv2d(64, 80, kernel_size=1),
            nn.Conv2d(80, 192, kernel_size=3),
            nn.MaxPool2d(3, stride=2)                    # 35x35
        )

    def forward(self, x):
        return self.stem(x)
```

This gives us a solid 35×35×192 output — the same spatial start point as Inception-A.

---

## 🧩 2️⃣ Inception-A, Reduction-A, Inception-B, Reduction-B, Inception-C Blocks

Very similar logic to v3 but:

* **More filters per branch**
* **More layers per block**
* **Consistent “same padding”**
* **Better normalization and structure**

Each block follows the same concept:

* Multi-branch (1×1, 3×3, 5×5) convs
* Parallel pooling + 1×1 projection
* Concatenation
* Downsampling via reduction blocks

---

## 🧮 Example Block (Inception-C)

```python
class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 256, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1),
            nn.Conv2d(384, 256, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(384, 256, kernel_size=(3,1), padding=(1,0))
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1),
            nn.Conv2d(384, 448, kernel_size=(3,1), padding=(1,0)),
            nn.Conv2d(448, 512, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(512, 256, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(512, 256, kernel_size=(3,1), padding=(1,0))
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 256, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
```

---

## 🧩 Inception-v4 Summary Flow

| Stage          | Type        | Output Size | Description              |
| -------------- | ----------- | ----------- | ------------------------ |
| Stem           | Conv + Pool | 35×35       | Initial                  |
| Inception-A ×4 | —           | 35×35       | Early feature extraction |
| Reduction-A    | ↓           | 17×17       | Downsample               |
| Inception-B ×7 | —           | 17×17       | Deeper multi-scale       |
| Reduction-B    | ↓           | 8×8         | Downsample               |
| Inception-C ×3 | —           | 8×8         | Final deep features      |
| AvgPool + FC   | —           | 1×1         | Classifier               |

✅ Deeper (42 layers)
✅ More balanced design
✅ Extremely accurate on ImageNet

---

# ⚖️ **v3 vs v4 — Side-by-Side**

| Feature          | Inception-v3          | Inception-v4                      |
| ---------------- | --------------------- | --------------------------------- |
| Factorization    | 3×3, 1×7, 7×1         | Extended (up to 1×7×1×3 combos)   |
| Normalization    | BN                    | BN + Better initialization        |
| Depth            | ~48 layers            | ~75 layers                        |
| Reduction design | Simpler               | Two dedicated reduction blocks    |
| Accuracy         | ~78% top-1 (ImageNet) | ~80%+                             |
| Speed            | Faster                | Slightly slower but more accurate |

---

# 🧮 Math Summary — Factorization Efficiency

| Kernel | Original Params | Factorized Params | Ratio      |
| ------ | --------------- | ----------------- | ---------- |
| 5×5    | 25              | 9+9 = 18          | 1.4× fewer |
| 7×7    | 49              | 7+7 = 14          | 3.5× fewer |

So factorization saves parameters and increases nonlinearity (extra BN + ReLU).

---

# 🎯 Final Takeaway

| Concept              | Meaning                                            |
| -------------------- | -------------------------------------------------- |
| **Inception-v1**     | Multi-scale filters                                |
| **Inception-v2/v3**  | Factorization for speed & depth                    |
| **Inception-v4**     | Deep, modular, clean with heavy reuse              |
| **Inception-ResNet** | Inception + skip connections = best of both worlds |

---



---

## **5. Inception-v3 Factorization Examples**

### **5×5 → 1×5 + 5×1**
```text
Input → 1×5 conv → 5×1 conv → Output
```
→ **33% cheaper**

### **7×7 → 1×7 + 7×1 → two 3×3**
```text
7×7 = 49 ops
1×7 + 7×1 = 14 ops → two 3×3 = 18 ops
```

---

## **6. Inception-v4 / Inception-ResNet-v2**

> **Not in `torchvision` by default**, but available via **timm** or custom impl.

```bash
pip install timm
```

```python
import timm

# Inception-v4
model = timm.create_model('inception_v4', pretrained=True, num_classes=10)

# Inception-ResNet-v2
model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=10)
```

---

## **7. Performance Comparison**

| Model | Params | Top-5 Error | Speed |
|------|--------|-------------|-------|
| **Inception-v3** | 25M | **5.6%** | Fast |
| **Inception-v4** | 43M | **4.9%** | Medium |
| **ResNet-152** | 60M | 5.7% | Slow |
| **DenseNet-201** | 20M | 6.2% | Slow |

---

## **8. Inception Module Evolution**

```text
v1: 1×1, 3×3, 5×5
v2: 5×5 → 3×3×2
v3: 5×5 → 1×5+5×1, 7×7 → 3×3×2
v4: + Residual connections
```

---

## **9. Parameter Efficiency (Chart.js)**

```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["Inception-v1", "v3", "v4", "ResNet-50"],
    "datasets": [{
      "label": "Parameters (M)",
      "data": [6.8, 25, 43, 25.6],
      "backgroundColor": ["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0"]
    }]
  },
  "options": {
    "plugins": { "title": { "display": true, "text": "Inception Family" } }
  }
}
```

---

## **10. Pros & Cons**

| Model | Pros | Cons |
|------|------|------|
| **v3** | Efficient, fast, accurate | No residual |
| **v4** | Best accuracy | More params, complex |

---

## **11. Summary**

| Feature | Inception-v3 | Inception-v4 |
|--------|--------------|-------------|
| **Year** | 2015 | 2016 |
| **Key** | Factorization, BN | Residual + Inception |
| **Params** | 25M | 43M |
| **Top-5** | 5.6% | **4.9%** |
| **Use Case** | Mobile, speed | Max accuracy |

---

## **Run Code**

```bash
pip install torch torchvision timm
python inception_v3.py
```

> Expected: **~96%+ on CIFAR-10**

---

**Inception-v3 is the gold standard for efficient deep CNNs.**  
Used in:
- MobileNet inspiration
- Object detection (SSD)
- Medical imaging
- Transfer learning

---
