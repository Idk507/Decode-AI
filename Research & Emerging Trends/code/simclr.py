import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Define SimCLR model
class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return z

# InfoNCE loss
def info_nce_loss(z1, z2, temperature=0.5, batch_size=64):
    z = torch.cat([z1, z2], dim=0)  # Concatenate two views
    sim = torch.mm(z, z.T) / temperature  # Cosine similarity
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels, labels])  # Positive pairs
    loss = F.cross_entropy(sim, labels)
    return loss

# Data augmentations for SimCLR
def get_simclr_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model and optimizer
backbone = torchvision.models.resnet18(pretrained=False)
backbone.fc = nn.Identity()  # Remove classification head
model = SimCLR(backbone).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Pre-training with SimCLR
def train_simclr(model, train_loader, optimizer, epochs=10):
    transform = get_simclr_transforms()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in train_loader:
            # Apply two different augmentations
            x1 = torch.stack([transform(img) for img in images]).cuda()
            x2 = torch.stack([transform(img) for img in images]).cuda()
            
            optimizer.zero_grad()
            z1 = model(x1)
            z2 = model(x2)
            loss = info_nce_loss(z1, z2, batch_size=images.size(0))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# Linear probing for evaluation
def evaluate_linear_probe(backbone, train_loader, test_loader):
    # Train a linear classifier on frozen backbone
    classifier = nn.Linear(512, 10).cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train linear classifier
    backbone.eval()
    classifier.train()
    for epoch in range(5):
        for images, labels in train_loader:
            images = images.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                features = backbone(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate on test set
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            features = backbone(images)
            outputs = classifier(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Linear Probe Accuracy: {accuracy:.2f}%")

# Run training and evaluation
train_simclr(model, train_loader, optimizer, epochs=10)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
evaluate_linear_probe(model.backbone, train_loader, test_loader)
