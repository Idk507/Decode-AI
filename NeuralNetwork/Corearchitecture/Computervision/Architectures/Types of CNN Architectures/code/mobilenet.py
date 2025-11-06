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
