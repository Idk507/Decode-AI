# **EfficientNet (v1–v2): Complete Guide**  
### *State-of-the-Art Efficiency via Compound Scaling*  
> **Paper**: *"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"* (Tan & Le, 2019)  
> **Authors**: Google Brain (Mingxing Tan, Quoc V. Le)  
> **Key Idea**: **Uniformly scale depth, width, and resolution** using a **compound coefficient φ**

---

## **1. Core Innovation: Compound Scaling**

### **Traditional Scaling**
- **Depth only**: Deeper → better, but diminishing returns
- **Width only**: Wider → better, but slow
- **Resolution only**: Higher input → better, but expensive

### **EfficientNet: Scale All Three Uniformly**

```math
\begin{aligned}
\text{depth} &= \alpha^\phi \\
\text{width} &= \beta^\phi \\
\text{resolution} &= \gamma^\phi \\
\text{s.t. } \alpha \cdot \beta^2 \cdot \gamma^2 &\approx 2
\end{aligned}
```

> **φ**: Compound coefficient (controls total resources)  
> **α, β, γ**: Optimal ratios found via **Neural Architecture Search (NAS)**

---

## **2. EfficientNet Variants (B0 → B7)**

| Model | φ | Params | FLOPs | Top-1 (ImageNet) | Input Size |
|-------|----|--------|-------|------------------|------------|
| **B0** | 0 | **5.3M** | **0.39B** | **77.1%** | 224 |
| **B1** | 0.5 | 7.8M | 0.70B | 79.1% | 240 |
| **B2** | 1.0 | 9.2M | 1.0B | 80.1% | 260 |
| **B3** | 1.5 | 12M | 1.8B | 81.6% | 300 |
| **B4** | 2.0 | 19M | 4.2B | 82.9% | 380 |
| **B5** | 3.0 | 30M | 9.9B | 83.6% | 456 |
| **B6** | 4.0 | 43M | 19B | 84.0% | 528 |
| **B7** | 5.0 | 66M | 37B | **84.3%** | 600 |

> **B0 = Baseline** (found via NAS)  
> **B7 = 8.4× smaller, 6.1× faster than ResNet-152**

---

## **3. EfficientNet-B0 Architecture**

| Stage | Operator | Resolution | #Channels | #Layers |
|-------|----------|------------|-----------|---------|
| 1 | Conv3×3 | 224×224 | 32 | 1 |
| 2 | MBConv1, k3×3 | 112×112 | 16 | 1 |
| 3 | MBConv6, k3×3 | 112×112 | 24 | 2 |
| 4 | MBConv6, k5×5 | 56×56 | 40 | 2 |
| 5 | MBConv6, k3×3 | 28×28 | 80 | 3 |
| 6 | MBConv6, k5×5 | 14×14 | 112 | 3 |
| 7 | MBConv6, k5×5 | 14×14 | 192 | 4 |
| 8 | MBConv6, k3×3 | 7×7 | 320 | 1 |
| 9 | Conv1×1 & Pooling & FC | 7×7 | 1280 | 1 |

> **MBConv** = Mobile Inverted Bottleneck (from MobileNet-v2) + **SE**

Perfect — let’s slow down and explain that part very simply 👶✨

We’re talking about this formula used by **EfficientNet**:

[
\text{depth} = \alpha^\phi, \quad
\text{width} = \beta^\phi, \quad
\text{resolution} = \gamma^\phi
]

<img width="1336" height="557" alt="image" src="https://github.com/user-attachments/assets/1d5708a7-0386-41d2-8fe9-0e9fa2454e23" />

---


> “Instead of just making a network deeper or wider,
> what if we *scale all dimensions together* — depth, width, and resolution — in a balanced way?”



---

## 🧠 The Problem Before EfficientNet

Before 2019, we had amazing models:

* **ResNet** → deeper = better
* **Inception** → smarter branching
* **MobileNet** → smaller & faster

But everyone scaled models **manually**:

* Make it *deeper* → more accuracy but slower
* Make it *wider* → faster but overfits
* Make images *bigger* → better detail but expensive

❌ These scaling choices weren’t balanced.
Sometimes they helped, sometimes they broke performance.

---

## 💡 The Idea Behind EfficientNet

> EfficientNet finds a **balanced scaling rule** that grows a model *smoothly*
> in **depth**, **width**, and **resolution** — *all at once*.

So instead of guessing,
it learns a **compound scaling formula** from experiments.

---

### ⚙️ The Formula

[
\text{depth} = \alpha^\phi, \quad
\text{width} = \beta^\phi, \quad
\text{resolution} = \gamma^\phi
]

where

* φ (phi) = scaling factor (how big you want the model)
* α, β, γ = constants found by a small grid search

✅ This way, every time you make the model “bigger” (increase φ),
you make **all three dimensions grow proportionally**.

---

## 🧩 Example

Start with a **baseline** network called **EfficientNet-B0** (small one).
Then:

| Model | Scaling (φ) | What It Means         |
| ----- | ----------- | --------------------- |
| B0    | φ = 0       | baseline              |
| B1    | φ = 1       | slightly deeper/wider |
| B2    | φ = 2       | larger images         |
| B7    | φ = 7       | huge, very accurate   |

So EfficientNet-B7 is basically **B0 scaled up** properly, not a totally new design.

---

## 🧱 The Building Block — MBConv

EfficientNet doesn’t reinvent layers — it builds on **MobileNet-v2’s inverted residual block**.

Each block looks like this:

```
Input
↓
1×1 Conv (Expand)
↓
Depthwise Conv (3×3 or 5×5)
↓
Squeeze-and-Excite (attention)
↓
1×1 Conv (Project, Linear)
↓
Skip Connection
```

✅ Uses depthwise separable conv → light
✅ Adds SE (Squeeze-and-Excitation) → smarter channel attention
✅ Linear bottleneck → keeps information

---

## 🔋 So Why “Efficient”?

Because it’s:

* **Smaller** → fewer parameters
* **Smarter** → scales properly
* **Stronger** → better accuracy

---

## 🧩 Architecture of EfficientNet-B0

| Stage  | Operator            | Resolution | Channels | #Layers |
| ------ | ------------------- | ---------- | -------- | ------- |
| Stem   | Conv3×3             | 224×224    | 32       | 1       |
| Block1 | MBConv1, k3×3       | 112×112    | 16       | 1       |
| Block2 | MBConv6, k3×3       | 112×112    | 24       | 2       |
| Block3 | MBConv6, k5×5       | 56×56      | 40       | 2       |
| Block4 | MBConv6, k3×3       | 28×28      | 80       | 3       |
| Block5 | MBConv6, k5×5       | 14×14      | 112      | 3       |
| Block6 | MBConv6, k5×5       | 14×14      | 192      | 4       |
| Block7 | MBConv6, k3×3       | 7×7        | 320      | 1       |
| Head   | Conv1×1 + Pool + FC | 7×7        | 1280     | 1       |

---

## 🧮 Parameter Efficiency

| Model               | Params   | Accuracy (ImageNet) | FLOPs |
| ------------------- | -------- | ------------------- | ----- |
| ResNet-50           | 25M      | 76%                 | 4.1B  |
| **EfficientNet-B0** | **5.3M** | **77%**             | 0.39B |
| EfficientNet-B7     | 66M      | **84.4%**           | 37B   |

✅ B0 → small but strong
✅ B7 → huge and highly accurate

---

## 🔬 Inside the Squeeze-and-Excite Block

Think of SE (Squeeze-and-Excite) like an “attention gate” for channels.

Steps:
1️⃣ **Squeeze:** Global average pool → creates 1 value per channel
2️⃣ **Excite:** Two FC layers → learn how important each channel is
3️⃣ **Scale:** Multiply input channels by their learned importance weights

✅ Helps model focus on “what matters” in the feature maps.

---

## 🎨 Simple Analogy

Imagine you’re editing a photo:

* **Depth** → applying more filters (more processing steps)
* **Width** → using more color channels (R, G, B, etc.)
* **Resolution** → using higher quality image

EfficientNet learns the *best balance* of these three —
so you get a clear, beautiful photo **without wasting power** 🎯

---

## 🧩 EfficientNet Family

| Model | Input Size | Params | Top-1 Accuracy |
| ----- | ---------- | ------ | -------------- |
| B0    | 224×224    | 5.3M   | 77.1%          |
| B1    | 240×240    | 7.8M   | 79.1%          |
| B2    | 260×260    | 9.2M   | 80.3%          |
| B3    | 300×300    | 12M    | 81.7%          |
| B4    | 380×380    | 19M    | 83.0%          |
| B5    | 456×456    | 30M    | 83.7%          |
| B6    | 528×528    | 43M    | 84.0%          |
| B7    | 600×600    | 66M    | 84.4%          |

All use **the same architecture**, only scaled by φ (compound coefficient).

---

## ⚙️ EfficientNet-v2 (2021) — Quick Mention

Later, Google released **EfficientNet-v2**:

* Even faster
* Uses **Fused MBConv** (normal + depthwise conv combined)
* Better training efficiency
* Smaller for the same accuracy

It’s what’s used in **Vision Transformers** hybrid models too.

---

## 🧠 Summary (for Beginners)

| Concept              | Meaning                                              |
| -------------------- | ---------------------------------------------------- |
| **EfficientNet**     | Scales network width, depth, and resolution together |
| **MBConv Block**     | Reused from MobileNet-v2 (light + SE attention)      |
| **Compound Scaling** | A smart rule for growing models fairly               |
| **Result**           | Best accuracy with the fewest parameters             |

---



> **EfficientNet** is like a perfectly balanced recipe — it increases the number of layers, width, and image size *together* so the network grows efficiently without wasting computation.

---


## 🧩 Step 1 — What are these words?

| Symbol         | What it means                                                                           | Simple idea                           |
| -------------- | --------------------------------------------------------------------------------------- | ------------------------------------- |
| **Depth**      | How many layers the network has                                                         | “How tall the building is”            |
| **Width**      | How many filters per layer                                                              | “How wide each floor is”              |
| **Resolution** | The input image size                                                                    | “How detailed your photo is”          |
| **φ (phi)**    | How much you want to scale up                                                           | “How big you want to make your model” |
| **α, β, γ**    | Numbers that control how much depth, width, and resolution should grow when φ increases | “How much you stretch each side”      |

---
<img width="284" height="659" alt="image" src="https://github.com/user-attachments/assets/f56fd113-c6c7-44c4-9c0d-73a7ac6a4dc3" />


## 🧠 Step 2 — The problem it solves

Before EfficientNet, people used to grow models randomly:

* “Let’s double the layers!”
* “Let’s make the image bigger!”

But that often made training **slow** or **unstable** 😩

EfficientNet found that the **best way** to grow a network
is to **balance all three** — depth, width, and resolution — **together**, using this formula.

---

## 🧮 Step 3 — What does the formula do?

Let’s say you start with a **small model (B0)**.

If you want to make it bigger → increase φ by 1.

Then:

* Make it **α times deeper** (more layers)
* Make it **β times wider** (more filters per layer)
* Use **γ times larger images** (higher resolution)

So instead of only scaling one part (like just making more layers),
you grow **everything evenly**.

---

## 🔢 Step 4 — Example with simple numbers

Imagine:

* α = 1.2 (depth grows 20% each step)
* β = 1.1 (width grows 10% each step)
* γ = 1.15 (resolution grows 15% each step)

Now:

| Model | φ | Depth | Width | Resolution |
| ----- | - | ----- | ----- | ---------- |
| B0    | 0 | 1×    | 1×    | 1×         |
| B1    | 1 | 1.2×  | 1.1×  | 1.15×      |
| B2    | 2 | 1.44× | 1.21× | 1.32×      |
| B3    | 3 | 1.73× | 1.33× | 1.52×      |

So each “B-number” (B1, B2, B3, … B7)
is just the **same network scaled up smoothly** in *all three directions*.

---

## 🎨 Step 5 — Simple Analogy

Imagine you’re zooming in on a photo.

You could:

* Only make it wider → it stretches sideways 😕
* Only make it taller → it looks weird 😕
* But if you make it **bigger evenly in all directions** → it stays clear and balanced 😍

That’s exactly what EfficientNet’s formula does —
it *zooms in* on the network evenly, so it stays efficient and accurate.

---

## ✅ In One Sentence

> The formula tells EfficientNet how to **grow bigger networks in a balanced way**, by increasing their **depth, width, and image size together**, using simple multipliers (α, β, γ) controlled by one knob (φ).

---


---

## **4. MBConv Block (Core of EfficientNet)**

```text
Input
│
├─► 1×1 conv → expand (6×) → Swish
├─► 3×3 or 5×5 depthwise → Swish
├─► SE (Squeeze-and-Excitation)
└─► 1×1 conv → project → **DropConnect**
│
+──► Skip (if stride=1 & same dim)
```

> **Swish** = `x * sigmoid(x)` → better than ReLU  
> **SE** = Channel attention  
> **DropConnect** = Dropout on skip connection

---

## **5. Full PyTorch Code: EfficientNet-B0**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_classes = 10
batch_size = 128
learning_rate = 0.1
num_epochs = 100
drop_connect_rate = 0.2

# Data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
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

# Swish
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# SE Module
class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            Swish(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

# MBConv
class MBConv(nn.Module):
    def __init__(self, in_c, out_c, exp_c, kernel, stride, se_ratio, drop_rate):
        super().__init__()
        self.use_se = se_ratio > 0
        self.use_skip = stride == 1 and in_c == out_c

        # Expand
        if exp_c != in_c:
            self.expand = nn.Sequential(
                nn.Conv2d(in_c, exp_c, 1, bias=False),
                nn.BatchNorm2d(exp_c),
                Swish()
            )
        else:
            self.expand = None

        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(exp_c, exp_c, kernel, stride, kernel//2, groups=exp_c, bias=False),
            nn.BatchNorm2d(exp_c),
            Swish()
        )

        # SE
        if self.use_se:
            self.se = SEModule(exp_c, int(exp_c * se_ratio))

        # Project
        self.project = nn.Sequential(
            nn.Conv2d(exp_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        # DropConnect
        self.drop_rate = drop_rate

    def forward(self, x):
        identity = x
        if self.expand:
            x = self.expand(x)
        x = self.depthwise(x)
        if self.use_se:
            x = self.se(x)
        x = self.project(x)

        if self.use_skip:
            if self.training and self.drop_rate > 0:
                x = x + identity * (torch.rand(x.size(0), 1, 1, 1, device=x.device) >= self.drop_rate).float()
            else:
                x = x + identity
        return x

# EfficientNet
class EfficientNet(nn.Module):
    def __init__(self, phi=0, num_classes=10):
        super().__init__()
        # Scaling coefficients
        alpha, beta, gamma = 1.2, 1.1, 1.15
        depth_coeff = alpha ** phi
        width_coeff = beta ** phi
        resolution_coeff = gamma ** phi

        # Baseline B0 config
        cfg = [
            # k, exp, out, se, stride, repeat
            [3,  32,  16, 0.25, 1, 1],
            [3,  16,  24, 0,    2, 2],
            [5,  24,  40, 0.25, 2, 2],
            [3,  40,  80, 0.25, 2, 3],
            [5,  80, 112, 0.25, 1, 3],
            [5, 112, 192, 0.25, 2, 4],
            [3, 192, 320, 0.25, 1, 1],
        ]

        # Scale
        in_c = 32
        out_c = int(32 * width_coeff)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_c, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            Swish()
        )

        blocks = []
        total_layers = 0
        for k, exp, c, se, s, r in cfg:
            r = max(1, round(r * depth_coeff))
            exp = round(exp * width_coeff)
            c = round(c * width_coeff)
            for i in range(r):
                stride = s if i == 0 else 1
                drop_rate = drop_connect_rate * total_layers / sum([max(1, round(rr * depth_coeff)) for _,_,_,_,_,rr in cfg])
                blocks.append(MBConv(in_c, c, exp, k, stride, se, drop_rate))
                in_c = c
                total_layers += 1

        self.blocks = nn.Sequential(*blocks)

        final_c = round(1280 * width_coeff)
        self.head = nn.Sequential(
            nn.Conv2d(in_c, final_c, 1, bias=False),
            nn.BatchNorm2d(final_c),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_c, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# B0
model = EfficientNet(phi=0, num_classes=num_classes).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training
def train():
    model.train()
    for epoch in range(num_epochs):
        for images, labels in trainloader:
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

print("Training EfficientNet-B0...")
train()
print("Testing...")
test()
```

---

## **6. Pretrained Models (PyTorch)**

```python
import torchvision.models as models

# EfficientNet-B0 to B7
model = models.efficientnet_b0(pretrained=True)
model = models.efficientnet_b7(pretrained=True)
```

---

## **7. Performance Comparison**

| Model | Params | FLOPs | Top-1 | Speed |
|------|--------|-------|-------|-------|
| **EfficientNet-B0** | 5.3M | 0.39B | **77.1%** | Very Fast |
| **B7** | 66M | 37B | **84.3%** | Fast |
| ResNet-50 | 25M | 4.1B | 76.1% | Medium |
| MobileNet-v3 | 5.4M | 0.22B | 75.2% | Fast |

---

## **8. Scaling Chart (Chart.js)**

<img width="999" height="563" alt="image" src="https://github.com/user-attachments/assets/c9f202b0-24c9-4542-9257-c66d2dda79ba" />


---

## **9. EfficientNet-v2 (2021)**

| Improvement | Description |
|-----------|-----------|
| **Fused-MBConv** | Replace expand+depthwise with 3×3 conv |
| **Progressive Learning** | Train with smaller images → larger |
| **Smaller SE** | 1×1 instead of 1×1+FC |
| **Result** | **B0: 78.8%**, **S: 83.9%**, **3× faster training**

---

## **10. Summary**

| Feature | Value |
|--------|-------|
| **Year** | 2019 |
| **Breakthrough** | **Compound Scaling** |
| **Best Model** | **B7: 84.3%** |
| **Params** | **5.3M → 66M** |
| **Legacy** | **New standard for efficiency** |

---

## **Run Code**

```bash
pip install torch torchvision
python efficientnet.py
```

> Expected: **~96%+ on CIFAR-10**

---

**EfficientNet is the endgame of CNN scaling.**  
Used in:
- **Google Vision API**
- **TensorFlow Hub**
- **Edge AI**
- **Medical imaging**

---

