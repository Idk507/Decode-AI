
---

## 🔥 What is SqueezeNet?

**SqueezeNet** (Iandola et al., 2016) is a **lightweight CNN** architecture that achieves **AlexNet-level accuracy** on ImageNet, but with **50× fewer parameters** (only ~1.2 million parameters vs. 60 million in AlexNet!).

**Goal:**

> Create a small, fast CNN that still performs competitively for image classification — ideal for deployment on mobile and embedded devices.

---

## 🎯 Design Philosophy

The authors focused on *model compression without accuracy loss*.
They achieved it using **three main strategies**:

1. **Replace 3×3 filters with 1×1 filters** wherever possible
   → Fewer parameters (since 1×1 has 9× fewer weights than 3×3).

2. **Reduce input channels to 3×3 filters**
   → Each 3×3 filter gets fewer input channels → fewer total weights.

3. **Delay downsampling** (pooling)
   → Keep large activation maps in early layers → more spatial information preserved.

---

Input → 1×1 Convs (Squeeze) → 1×1 + 3×3 Convs (Expand) → Concat


## 🧩 The Core Building Block — The “Fire Module”

SqueezeNet’s magic comes from the **Fire module**.

Each **Fire module** has:

* **Squeeze layer**: uses 1×1 filters to reduce (squeeze) the number of input channels.
* **Expand layer**: mixes 1×1 and 3×3 filters to “expand” features again.

```
Input
  ↓
[Squeeze] 1×1 conv → fewer channels
  ↓
[Expand]
   ├─ 1×1 conv (half of expansion)
   └─ 3×3 conv (other half)
  ↓
Concatenate outputs (along channel axis)
  ↓
Output
```

So a Fire module looks like this:

```
            → 1x1 conv →
Input → 1x1 conv →          → concat → Output
            → 3x3 conv →
```

---

## 🔢 Mathematical Understanding

Let’s define:

<img width="832" height="441" alt="image" src="https://github.com/user-attachments/assets/c6dea052-6da9-4e14-9b1b-14782c91f0ca" />


---

## 🏗️ Full SqueezeNet Architecture

Here’s the layer-by-layer breakdown of **SqueezeNet v1.0** (the original version):

| Stage | Layer         | Type                                 | Details          |
| ----- | ------------- | ------------------------------------ | ---------------- |
| 1     | Conv1         | 7×7 conv, stride 2                   | 96 filters       |
|       | MaxPool1      | 3×3, stride 2                        |                  |
| 2     | Fire2         | Squeeze=16, Expand(1×1=64, 3×3=64)   |                  |
| 3     | Fire3         | Squeeze=16, Expand(1×1=64, 3×3=64)   |                  |
| 4     | Fire4         | Squeeze=32, Expand(1×1=128, 3×3=128) |                  |
|       | MaxPool4      | 3×3, stride 2                        |                  |
| 5     | Fire5         | Squeeze=32, Expand(1×1=128, 3×3=128) |                  |
| 6     | Fire6         | Squeeze=48, Expand(1×1=192, 3×3=192) |                  |
| 7     | Fire7         | Squeeze=48, Expand(1×1=192, 3×3=192) |                  |
| 8     | Fire8         | Squeeze=64, Expand(1×1=256, 3×3=256) |                  |
|       | MaxPool8      | 3×3, stride 2                        |                  |
| 9     | Fire9         | Squeeze=64, Expand(1×1=256, 3×3=256) |                  |
| 10    | Conv10        | 1×1 conv                             | #classes filters |
| 11    | GlobalAvgPool | 13×13 → 1×1                          |                  |
| 12    | Softmax       | —                                    | classification   |

---

## 🧠 Why It’s Efficient

### 1. Smaller filters

1×1 filters drastically reduce parameters.

### 2. Channel squeezing

Fewer input channels to 3×3 filters → fewer multiplications.

### 3. Late pooling

Downsampling is postponed → large feature maps longer → better accuracy.

### 4. Global Average Pooling

No fully connected layers → drastically reduces parameters.

---

## 📐 Parameter Comparison

| Model           | Parameters  | Size (MB) | Accuracy (Top-1, ImageNet) |
| --------------- | ----------- | --------- | -------------------------- |
| AlexNet         | 60 million  | ~240 MB   | 57%                        |
| SqueezeNet      | 1.2 million | ~4.8 MB   | 57%                        |
| MobileNetV2     | 3.4 million | ~14 MB    | 71%                        |
| EfficientNet-B0 | 5.3 million | ~20 MB    | 77%                        |

---

## ⚙️ PyTorch Implementation (Simple and Readable)

Let’s build the **Fire module** and the **SqueezeNet model**.

```python
# squeeznet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Fire Module ----
class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        # Squeeze layer
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # Expand layers
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        out1 = self.expand1x1(x)
        out3 = self.expand3x3(x)
        return self.expand_activation(torch.cat([out1, out3], dim=1))


# ---- SqueezeNet ----
class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            Fire(512, 64, 256, 256)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
```

✅ You can run this directly:

```python
model = SqueezeNet(num_classes=10)
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(y.shape)   # torch.Size([1, 10])
```

---

## 🧮 Training Setup (Example)

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 📊 Key Insights

| Concept              | Explanation                                                       |
| -------------------- | ----------------------------------------------------------------- |
| **1×1 Conv filters** | Drastically reduce parameters                                     |
| **Squeeze ratio**    | Number of squeeze channels / expand channels (~0.125–0.5 typical) |
| **No FC layers**     | Use global average pooling                                        |
| **ReLU everywhere**  | Simple, fast nonlinearities                                       |
| **Dropout**          | Adds regularization for small model                               |

---

## ⚖️ SqueezeNet vs Other Compact CNNs

| Model               | Params (M) | Accuracy | Notes                        |
| ------------------- | ---------- | -------- | ---------------------------- |
| **SqueezeNet**      | 1.2        | 57%      | Smallest                     |
| **MobileNetV2**     | 3.4        | 71%      | Depthwise separable convs    |
| **ShuffleNet**      | 1.4        | 69%      | Group conv + channel shuffle |
| **EfficientNet-B0** | 5.3        | 77%      | Compound scaling             |
| **ResNet-18**       | 11.7       | 69%      | Baseline comparison          |

---

## 🧠 Summary

**SqueezeNet = “AlexNet accuracy with 50× fewer parameters.”**

**Core idea:** Fire modules (1×1 squeeze + 1×1 & 3×3 expand).
**Benefits:**

* Compact and fast.
* Easy to train from scratch.
* Excellent for edge devices (mobile, Raspberry Pi, etc.).
* Used as a backbone for lightweight detection/segmentation networks.

---

## 🚀 Extensions

* **SqueezeNet v1.1** — smaller filters (3×3 instead of 7×7 in conv1), faster.
* **SqueezeNext** — deeper but more efficient variant.
* **SqueezeSeg** — SqueezeNet adapted for LiDAR point-cloud segmentation.
* **Tiny-YOLO + SqueezeNet** — for real-time object detection.

---

## 💡 TL;DR

| Concept            | Summary                                      |
| ------------------ | -------------------------------------------- |
| 🧱 Architecture    | Fire modules (1×1 squeeze + expand)          |
| ⚙️ Parameters      | ~1.2M                                        |
| 🧩 Filter Strategy | Replace 3×3 with 1×1; fewer input channels   |
| 🕒 Speed           | ~2–3× faster than AlexNet                    |
| 🧠 Accuracy        | Similar to AlexNet (~57% on ImageNet)        |
| 📱 Use case        | Lightweight classification / edge deployment |

---
