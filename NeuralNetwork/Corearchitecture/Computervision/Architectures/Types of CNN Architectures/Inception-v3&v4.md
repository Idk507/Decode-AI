
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
