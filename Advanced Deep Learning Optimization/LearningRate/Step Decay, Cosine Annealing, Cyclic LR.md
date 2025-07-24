
---

### **1. Step Decay**

#### **What is it?**
Step Decay reduces the learning rate by a fixed factor (e.g., divide by 10) at specific intervals (e.g., every 10 epochs). It’s like taking big steps at first and then smaller steps as you get closer to the goal.

#### **Why Use It?**
- **Simple and effective**: Easy to implement and works well for many tasks.
- **Stabilizes training**: Large learning rates early on speed up learning, while smaller rates later fine-tune the model.
- **When to use**: When you know roughly how many epochs you’ll train and want predictable drops in the learning rate.

#### **Math**
<img width="721" height="457" alt="image" src="https://github.com/user-attachments/assets/aa43a7ee-451e-41a1-9c1a-e692aca84400" />


#### **Pros and Cons**
- **Pros**: Simple, predictable, works well for many tasks.
- **Cons**: Abrupt drops can disrupt training if not timed well.

---

### **2. Cosine Annealing**

#### **What is it?**
Cosine Annealing reduces the learning rate smoothly following a cosine curve over a set number of epochs or iterations. It starts high, decreases to a minimum, and can optionally restart (used in “warm restarts”).

#### **Why Use It?**
- **Smooth transitions**: Avoids abrupt changes in learning rate, leading to stable training.
- **Good for fine-tuning**: The gradual decrease helps the model settle into a good minimum.
- **When to use**: For tasks like image classification or NLP, especially with large models like Transformers, or when you want to explore the loss landscape smoothly.

#### **Math**
<img width="900" height="511" alt="image" src="https://github.com/user-attachments/assets/33ca4c99-0b14-4f94-a195-5eea17389588" />

#### **Pros and Cons**
- **Pros**: Smooth decay, works well for deep networks, supports warm restarts for cycling.
- **Cons**: Requires tuning \( T_{\text{max}} \), may not adapt to training dynamics.

---

### **3. Cyclic Learning Rates (CLR)**

#### **What is it?**
Cyclic Learning Rates oscillate the learning rate between a minimum and maximum value in a repeating cycle (e.g., triangular or sinusoidal pattern). Instead of always decreasing, the learning rate goes up and down to explore the loss landscape.

#### **Why Use It?**
- **Escapes local minima**: Varying the learning rate helps the model jump out of bad spots in the loss landscape.
- **Faster convergence**: Can reduce the need for extensive learning rate tuning.
- **When to use**: For complex models or datasets where the loss landscape is rugged, or when you want to train faster with less hyperparameter tuning.

#### **Math**
<img width="960" height="650" alt="image" src="https://github.com/user-attachments/assets/54d0f49b-391f-4782-8356-fe347f21352e" />


#### **Pros and Cons**
- **Pros**: Reduces need for learning rate tuning, helps escape local minima, speeds up training.
- **Cons**: Requires tuning cycle length and bounds, can be unstable if bounds are poorly chosen.

---

### **Code Example in PyTorch**

Below is a PyTorch example implementing **Step Decay**, **Cosine Annealing**, and **Cyclic Learning Rates** for training a simple neural network on the MNIST dataset. I’ll use the same model as in your previous question for consistency but show all three schedulers.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

# 1. Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Load MNIST dataset
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. Initialize model, loss, optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 4. Choose a scheduler (uncomment one to use)
# Step Decay: Reduce lr by 0.1 every 5 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Cosine Annealing: Decay lr over 20 epochs
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)

# Cyclic LR: Cycle between 0.001 and 0.1 every 2000 steps
# scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=1000, mode='triangular')

# 5. Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # For CyclicLR, step the scheduler after each batch
        if isinstance(scheduler, lr_scheduler.CyclicLR):
            scheduler.step()

    # For StepLR and CosineAnnealingLR, step after each epoch
    if not isinstance(scheduler, lr_scheduler.CyclicLR):
        scheduler.step()

    # Print current learning rate and loss
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Learning Rate: {current_lr:.6f}")

print("Training finished!")
```

---

### **Explanation of the Code**

1. **Model**: A simple feedforward neural network (`SimpleNet`) for MNIST digit classification (28x28 images to 10 classes).
2. **Optimizer**: SGD with an initial learning rate of 0.1 and momentum of 0.9.
3. **Schedulers** (uncomment one to use):
   - **Step Decay**: Reduces learning rate by 0.1 every 5 epochs (`step_size=5`, `gamma=0.1`).
   - **Cosine Annealing**: Decays learning rate over 20 epochs (`T_max=20`) to a minimum of 0.001 (`eta_min=0.001`).
   - **Cyclic LR**: Cycles between 0.001 and 0.1 every 2000 steps (`step_size_up=1000`, `mode='triangular'`).
4. **Training Loop**:
   - Processes batches, computes loss, and updates weights.
   - For **CyclicLR**, the scheduler steps after each batch (since cycles are step-based).
   - For **StepLR** and **CosineAnnealingLR**, the scheduler steps after each epoch.
5. **Output**: Prints the epoch, average loss, and current learning rate.

---

### **Key Differences in the Code**
- **Step Decay**: Learning rate drops abruptly at fixed intervals (e.g., epoch 5, 10, 15).
- **Cosine Annealing**: Learning rate decreases smoothly, reaching the minimum at the end of the cycle (epoch 20).
- **Cyclic LR**: Learning rate oscillates every 2000 steps, creating a triangular pattern. Since MNIST has ~938 batches per epoch (60,000 images / 64 batch size), one cycle takes ~2 epochs.

---

### **When to Use Each Scheduler**
- **Step Decay**:
  - Use for: Standard tasks like image classification where you want predictable learning rate drops.
  - Example: Training CNNs on datasets like CIFAR-10 or MNIST.
  - Why: Simple to set up and works well when you can estimate training duration.
- **Cosine Annealing**:
  - Use for: Large models (e.g., Transformers) or tasks needing smooth learning rate changes.
  - Example: NLP tasks or fine-tuning pre-trained models.
  - Why: Smooth decay avoids abrupt changes, and warm restarts can help explore the loss landscape.
- **Cyclic LR**:
  - Use for: Complex models or datasets where you want faster training with less hyperparameter tuning.
  - Example: Training ResNets or when experimenting with new architectures.
  - Why: Oscillating rates help escape local minima and can reduce training time.

---

### **Visualizing the Learning Rate Schedules**

To help understand how the learning rates change, here’s a chart comparing the three schedulers over 20 epochs (assuming ~938 batches per epoch for Cyclic LR).

<img width="1006" height="563" alt="image" src="https://github.com/user-attachments/assets/1d023514-ced2-4236-9c51-ca61df84076f" />


**Chart Explanation**:
- **Step Decay** (blue): Drops from 0.1 to 0.01 at epoch 5, to 0.001 at epoch 10, etc.
- **Cosine Annealing** (orange): Smoothly decreases from 0.1 to 0.001 over 10 epochs, then restarts.
- **Cyclic LR** (green): Oscillates between 0.001 and 0.1 every ~2 epochs (2000 steps).

---

### **Tips for Using These Schedulers**
1. **Step Decay**:
   - Tune `step_size` and `gamma` based on your dataset and model.
   - Example: For small datasets, use smaller `step_size` (e.g., 5 epochs).
2. **Cosine Annealing**:
   - Set `T_max` to the total number of epochs or a fraction for warm restarts.
   - Use `eta_min` > 0 for stability in large models.
3. **Cyclic LR**:
   - Choose `base_lr` and `max_lr` based on a learning rate range test (see Leslie Smith’s CLR paper).
   - Set `step_size_up` to 2–10 times the number of batches per epoch.
4. **Experiment**: Test different schedulers on your dataset to see which converges fastest or achieves the best accuracy.

---

### **Conclusion**
- **Step Decay** is simple and effective for predictable learning rate drops, ideal for standard tasks.
- **Cosine Annealing** provides smooth decay, great for large models or fine-tuning.
- **Cyclic LR** oscillates the learning rate to explore the loss landscape, speeding up training and reducing tuning effort.

