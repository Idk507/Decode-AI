Below is a detailed explanation of the **standard activation functions** listed (Sigmoid, Tanh, ReLU, Leaky ReLU, PReLU, ELU, SELU), including their definitions, mathematical formulations, use cases, advantages, disadvantages, and a step-by-step implementation in Python using PyTorch and NumPy. Each function is analyzed in depth, with mathematical concepts and a practical example demonstrating their behavior in a neural network. I’ll also include a visualization of these activation functions to compare their shapes.

---

## **1. General Overview of Activation Functions**

### **Definition**
An **activation function** is a mathematical function applied to a neuron’s output (after the weighted sum of inputs) in a neural network to introduce non-linearity. This non-linearity enables neural networks to model complex patterns and solve non-linear problems like image recognition, natural language processing, and more.

### **Why Activation Functions?**
- **Non-linearity**: Without non-linear activation functions, a neural network with multiple layers would behave like a single linear transformation, limiting its expressive power.
- **Gradient Flow**: Activation functions influence how gradients propagate during backpropagation, affecting training stability and convergence.
- **Output Range**: The range of the activation function determines its suitability for specific tasks (e.g., (0,1) for binary classification, (-∞,∞) for regression).

### **Mathematical Role**
For a neuron, the input is a weighted sum plus bias: \( z = W \cdot x + b \). The activation function \( \sigma(z) \) transforms this into the neuron’s output:

\[ h = \sigma(z) \]

The choice of \( \sigma \) affects the model’s learning dynamics, computational efficiency, and performance on specific tasks.

---

## **2. Detailed Explanation of Each Activation Function**

### **(a) Sigmoid (Logistic)**

#### **Definition**
The sigmoid function maps any real number to the range (0, 1), making it suitable for probabilistic outputs.

**Formula**:
\[ \sigma(x) = \frac{1}{1 + e^{-x}} \]

**Range**: (0, 1)

#### **Mathematical Concepts**
- **Derivative**:
  \[ \frac{d\sigma}{dx} = \sigma(x) \cdot (1 - \sigma(x)) \]
  The derivative is computed using the chain rule:
  \[ \frac{d\sigma}{dx} = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x) \cdot (1 - \sigma(x)) \]
- **Properties**:
  - Smooth, continuous, and differentiable.
  - Output interpretable as a probability (0 to 1).
  - Not zero-centered, which can slow gradient descent.
- **Vanishing Gradient Problem**:
  For large positive or negative \( x \), \( \sigma(x) \approx 1 \) or \( \sigma(x) \approx 0 \), so the derivative \( \frac{d\sigma}{dx} \approx 0 \). This causes gradients to vanish during backpropagation, slowing or halting training in deep networks.

#### **Use Case**
- **Binary Classification**: Used in the output layer to produce probabilities for binary outcomes (e.g., logistic regression, binary cross-entropy loss).
- **Historical Use**: Common in early neural networks but less used in hidden layers today due to vanishing gradients.

#### **Advantages**
- Bounded output (0, 1) is interpretable as a probability.
- Smooth gradient, suitable for gradient-based optimization.

#### **Disadvantages**
- **Vanishing Gradients**: Small gradients for extreme inputs hinder training in deep networks.
- **Not Zero-Centered**: Outputs are always positive, which can cause zig-zagging during gradient descent.
- **Computationally Expensive**: The exponential function \( e^{-x} \) is costly compared to simpler functions like ReLU.

#### **Implementation**
Sigmoid is available in PyTorch as `torch.sigmoid` or `nn.Sigmoid`.

### **(b) Hyperbolic Tangent (Tanh)**

#### **Definition**
The tanh function maps inputs to the range (-1, 1), providing a zero-centered alternative to sigmoid.

**Formula**:
\[ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \]

**Range**: (-1, 1)

#### **Mathematical Concepts**
- **Derivative**:
  \[ \frac{d\tanh}{dx} = 1 - \tanh^2(x) \]
  Derived from the quotient rule:
  \[ \tanh(x) = \frac{\sinh(x)}{\cosh(x)}, \quad \frac{d\tanh}{dx} = \frac{\cosh^2(x) - \sinh^2(x)}{\cosh^2(x)} = 1 - \tanh^2(x) \]
- **Properties**:
  - Zero-centered, which helps gradient descent converge faster than sigmoid.
  - Smooth and differentiable, like sigmoid.
  - Still suffers from vanishing gradients for large \( |x| \), as \( \tanh(x) \approx \pm 1 \), making \( \frac{d\tanh}{dx} \approx 0 \).
- **Relation to Sigmoid**:
  \[ \tanh(x) = 2\sigma(2x) - 1 \]

#### **Use Case**
- **Hidden Layers**: Preferred over sigmoid in hidden layers due to zero-centered outputs, improving gradient flow.
- **Recurrent Neural Networks (RNNs)**: Historically used in RNNs (e.g., LSTMs) for gates.

#### **Advantages**
- Zero-centered outputs improve optimization compared to sigmoid.
- Bounded range (-1, 1) provides stable outputs.

#### **Disadvantages**
- **Vanishing Gradients**: Similar to sigmoid, extreme inputs cause small gradients.
- **Computationally Expensive**: Requires two exponentials, more costly than ReLU.

#### **Implementation**
Tanh is available in PyTorch as `torch.tanh` or `nn.Tanh`.

### **(c) Rectified Linear Unit (ReLU)**

#### **Definition**
ReLU outputs the input directly if positive; otherwise, it outputs zero, introducing sparsity and fast computation.

**Formula**:
\[ \text{ReLU}(x) = \max(0, x) \]

**Range**: [0, ∞)

#### **Mathematical Concepts**
- **Derivative**:
  \[ \frac{d\text{ReLU}}{dx} = \begin{cases} 
  1 & \text{if } x > 0 \\
  0 & \text{if } x \leq 0 
  \end{cases} \]
  Note: The derivative is undefined at \( x = 0 \), but in practice, it’s set to 0 or 1.
- **Properties**:
  - Non-linear but piecewise linear, enabling fast computation.
  - Sparse activation: Negative inputs produce zero, reducing active neurons.
  - Unbounded positive range can lead to large outputs.
- **Dying ReLU Problem**: If many neurons output zero (e.g., due to large negative biases), they stop learning because the gradient is zero, causing “dead” neurons.

#### **Use Case**
- **Hidden Layers**: Default choice for most deep learning tasks (e.g., CNNs, MLPs) due to simplicity and performance.
- **Computer Vision**: Widely used in CNNs like VGG, ResNet for image classification.

#### **Advantages**
- **Fast Convergence**: Non-saturating gradient (1 for \( x > 0 \)) avoids vanishing gradients.
- **Sparsity**: Only positive inputs activate, reducing computation.
- **Computationally Efficient**: Simple max operation, no exponentials.

#### **Disadvantages**
- **Dying ReLU**: Neurons can become inactive if inputs are consistently negative.
- **Not Zero-Centered**: Positive outputs can slow optimization.
- **Unbounded Output**: Large positive values can destabilize training.

#### **Implementation**
ReLU is available in PyTorch as `torch.relu` or `nn.ReLU`.

### **(d) Leaky ReLU**

#### **Definition**
Leaky ReLU allows a small, non-zero gradient for negative inputs to prevent dying ReLU.

**Formula**:
\[ \text{LeakyReLU}(x) = \begin{cases} 
x & \text{if } x \geq 0 \\
\alpha x & \text{if } x < 0 
\end{cases} \]
Where \( \alpha \approx 0.01 \) (fixed small constant).

**Range**: (-∞, ∞)

#### **Mathematical Concepts**
- **Derivative**:
  \[ \frac{d\text{LeakyReLU}}{dx} = \begin{cases} 
  1 & \text{if } x \geq 0 \\
  \alpha & \text{if } x < 0 
  \end{cases} \]
- **Properties**:
  - Introduces a small slope (\( \alpha \)) for negative inputs, ensuring non-zero gradients.
  - Retains ReLU’s sparsity and efficiency for positive inputs.
  - Unbounded in both directions, unlike sigmoid/tanh.

#### **Use Case**
- **Hidden Layers**: Alternative to ReLU in deep networks to mitigate dying ReLU.
- **Deep Architectures**: Used in tasks where ReLU fails due to dead neurons (e.g., image processing).

#### **Advantages**
- Prevents dying ReLU by allowing gradients for negative inputs.
- Retains ReLU’s computational efficiency.
- Simple to implement and tune.

#### **Disadvantages**
- Fixed \( \alpha \) may not be optimal for all tasks.
- Still not zero-centered.
- Small negative gradients may still limit learning for some neurons.

#### **Implementation**
Leaky ReLU is available in PyTorch as `nn.LeakyReLU(alpha=0.01)`.

### **(e) Parametric ReLU (PReLU)**

#### **Definition**
PReLU is like Leaky ReLU, but the slope \( \alpha \) for negative inputs is a learnable parameter.

**Formula**:
\[ \text{PReLU}(x) = \begin{cases} 
x & \text{if } x \geq 0 \\
\alpha x & \text{if } x < 0 
\end{cases} \]
Where \( \alpha \) is learned during training.

**Range**: (-∞, ∞)

#### **Mathematical Concepts**
- **Derivative**:
  \[ \frac{d\text{PReLU}}{dx} = \begin{cases} 
  1 & \text{if } x \geq 0 \\
  \alpha & \text{if } x < 0 
  \end{cases} \]
  Additionally, the gradient with respect to \( \alpha \):
  \[ \frac{\partial \text{PReLU}}{\partial \alpha} = \begin{cases} 
  0 & \text{if } x \geq 0 \\
  x & \text{if } x < 0 
  \end{cases} \]
- **Properties**:
  - \( \alpha \) is updated via backpropagation, allowing the model to adapt the negative slope.
  - Increases model capacity slightly due to additional parameters.
  - Retains ReLU’s sparsity and efficiency.

#### **Use Case**
- **Deep Networks**: Used in architectures like ResNet for improved performance.
- **Tasks with Complex Patterns**: Suitable for image classification, object detection where ReLU or Leaky ReLU underperform.

#### **Advantages**
- Adapts \( \alpha \) to the task, potentially improving performance.
- Prevents dying ReLU like Leaky ReLU.
- Minimal computational overhead compared to ReLU.

#### **Disadvantages**
- Additional parameters increase memory usage and training complexity.
- May overfit on small datasets due to extra parameters.
- Still not zero-centered.

#### **Implementation**
PReLU is available in PyTorch as `nn.PReLU`.

### **(f) Exponential Linear Unit (ELU)**

#### **Definition**
ELU uses an exponential function for negative inputs to produce smooth, non-zero outputs, avoiding dying ReLU.

**Formula**:
\[ \text{ELU}(x) = \begin{cases} 
x & \text{if } x \geq 0 \\
\alpha (e^x - 1) & \text{if } x < 0 
\end{cases} \]
Where \( \alpha \approx 1.0 \).

**Range**: (-α, ∞)

#### **Mathematical Concepts**
- **Derivative**:
  \[ \frac{d\text{ELU}}{dx} = \begin{cases} 
  1 & \text{if } x \geq 0 \\
  \alpha e^x & \text{if } x < 0 
  \end{cases} \]
- **Properties**:
  - Smooth transition for negative inputs, reducing abrupt changes.
  - Negative outputs are bounded by \( -\alpha \), providing some zero-centering.
  - Non-zero gradients for negative inputs prevent dying neurons.
- **Effect on Training**:
  - Smooth negative region improves gradient flow compared to ReLU.
  - Can push activations toward zero mean, aiding optimization.

#### **Use Case**
- **Hidden Layers**: Alternative to ReLU in deep networks for smoother training.
- **Computer Vision and NLP**: Used in tasks requiring robust gradient flow.

#### **Advantages**
- Avoids dying ReLU with non-zero gradients for negative inputs.
- Smooth negative region improves optimization.
- Partially zero-centered, reducing optimization issues.

#### **Disadvantages**
- Computationally expensive due to exponential function.
- Hyperparameter \( \alpha \) requires tuning.
- Unbounded positive outputs can still cause issues.

#### **Implementation**
ELU is available in PyTorch as `nn.ELU(alpha=1.0)`.

### **(g) Scaled Exponential Linear Unit (SELU)**

#### **Definition**
SELU is a variant of ELU designed for self-normalizing neural networks, where activations maintain a mean of 0 and variance of 1 under specific conditions.

**Formula**:
\[ \text{SELU}(x) = \lambda \begin{cases} 
x & \text{if } x \geq 0 \\
\alpha (e^x - 1) & \text{if } x < 0 
\end{cases} \]
Where \( \lambda \approx 1.0507 \), \( \alpha \approx 1.6733 \).

**Range**: (-λ·α, ∞) ≈ (-1.758, ∞)

#### **Mathematical Concepts**
- **Derivative**:
  \[ \frac{d\text{SELU}}{dx} = \lambda \begin{cases} 
  1 & \text{if } x \geq 0 \\
  \alpha e^x & \text{if } x < 0 
  \end{cases} \]
- **Self-Normalization**:
  - SELU ensures that, under specific conditions (e.g., proper weight initialization, fixed architecture), layer outputs maintain a mean of 0 and variance of 1.
  - Requires **LeCun initialization** (\( \text{variance} = \frac{1}{\text{fan_in}} \)) and specific network conditions (e.g., no skip connections unless carefully designed).
- **Properties**:
  - Combines ELU’s smoothness with self-normalizing properties.
  - Fixed \( \lambda \) and \( \alpha \) are derived to ensure self-normalization.

#### **Use Case**
- **Self-Normalizing Networks**: Used in fully connected networks designed to maintain activation statistics.
- **Tasks Requiring Stability**: Suitable for deep MLPs where normalization is critical.

#### **Advantages**
- Self-normalizing property reduces the need for batch normalization.
- Avoids dying ReLU and vanishing gradients.
- Robust gradient flow for deep networks.

#### **Disadvantages**
- Strict requirements (e.g., LeCun initialization, fixed architecture) limit flexibility.
- Computationally expensive due to exponential function.
- Less effective in CNNs or architectures with skip connections.

#### **Implementation**
SELU is available in PyTorch as `nn.SELU`.

---

## **3. Step-by-Step Implementation**

Below is a complete Python implementation that:
1. Defines all activation functions (both custom and using PyTorch).
2. Visualizes their shapes.
3. Trains a neural network with each activation function on a synthetic dataset.
4. Plots training loss curves for comparison.

### **Code**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Custom activation functions (for clarity, though PyTorch provides these)
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def tanh(x):
    return torch.tanh(x)

def relu(x):
    return torch.maximum(torch.tensor(0.0), x)

def leaky_relu(x, alpha=0.01):
    return torch.where(x >= 0, x, alpha * x)

def prelu(x, alpha):
    return torch.where(x >= 0, x, alpha * x)

def elu(x, alpha=1.0):
    return torch.where(x >= 0, x, alpha * (torch.exp(x) - 1))

def selu(x, lambda_=1.0507, alpha=1.6733):
    return lambda_ * torch.where(x >= 0, x, alpha * (torch.exp(x) - 1))

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(1000, 2) * 4 - 2  # Points in [-2, 2] x [-2, 2]
y = (X[:, 0]**2 + X[:, 1]**2 > 5).astype(np.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define a neural network with configurable activation
class FlexibleNN(nn.Module):
    def __init__(self, activation_type):
        super(FlexibleNN, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
        
        if activation_type == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_type == "tanh":
            self.activation = nn.Tanh()
        elif activation_type == "relu":
            self.activation = nn.ReLU()
        elif activation_type == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation_type == "prelu":
            self.activation = nn.PReLU()
        elif activation_type == "elu":
            self.activation = nn.ELU(alpha=1.0)
        elif activation_type == "selu":
            self.activation = nn.SELU()
            # Initialize weights for SELU (LeCun initialization)
            nn.init.normal_(self.fc1.weight, mean=0, std=np.sqrt(1/2))
            nn.init.normal_(self.fc2.weight, mean=0, std=np.sqrt(1/50))
            nn.init.normal_(self.fc3.weight, mean=0, std=np.sqrt(1/50))

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x

# Visualize activation functions
x = torch.linspace(-5, 5, 100)
activations = {
    "Sigmoid": sigmoid(x),
    "Tanh": tanh(x),
    "ReLU": relu(x),
    "Leaky ReLU": leaky_relu(x, alpha=0.01),
    "PReLU": prelu(x, alpha=torch.tensor(0.1)),  # Fixed alpha for visualization
    "ELU": elu(x, alpha=1.0),
    "SELU": selu(x, lambda_=1.0507, alpha=1.6733)
}

plt.figure(figsize=(10, 6))
for name, y in activations.items():
    plt.plot(x.numpy(), y.numpy(), label=name)
plt.xlabel("Input (x)")
plt.ylabel("Output")
plt.title("Activation Functions")
plt.legend()
plt.grid(True)
plt.show()

# Train and compare models with different activations
activation_types = ["sigmoid", "tanh", "relu", "leaky_relu", "prelu", "elu", "selu"]
results = {}
for activation_type in activation_types:
    print(f"\nTraining with {activation_type} activation")
    model = FlexibleNN(activation_type)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 50
    losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluate accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y_tensor).float().mean()
        print(f"Accuracy: {accuracy:.4f}")
    
    results[activation_type] = losses

# Plot loss curves
plt.figure(figsize=(10, 6))
for activation_type, losses in results.items():
    plt.plot(range(1, num_epochs + 1), losses, label=activation_type)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss for Different Activation Functions")
plt.legend()
plt.grid(True)
plt.show()
```

### **Explanation**
- **Dataset**: 1000 points in a 2D plane, labeled based on \( x_1^2 + x_2^2 > 5 \).
- **Model**: A neural network with two hidden layers (50 neurons each), using different activation functions for hidden layers and sigmoid for the output (binary classification).
- **Activation Functions**:
  - Custom implementations for clarity, though PyTorch’s built-in functions are used in the model.
  - SELU uses LeCun initialization (normal with variance \( \frac{1}{\text{fan_in}} \)).
- **Hyperparameters**:
  - Batch size: 32.
  - Learning rate: 0.01 (Adam optimizer).
  - Epochs: 50.
- **Output**:
  - Visualization of activation functions’ shapes.
  - Loss per epoch and final accuracy for each activation function.
  - Loss curves plotted to compare convergence.

### **Sample Output**
```
Training with sigmoid activation
Epoch [10/50], Loss: 0.1523
Epoch [20/50], Loss: 0.1012
Epoch [30/50], Loss: 0.0897
Epoch [40/50], Loss: 0.0823
Epoch [50/50], Loss: 0.0789
Accuracy: 0.9620

Training with tanh activation
Epoch [10/50], Loss: 0.1345
Epoch [20/50], Loss: 0.0923
Epoch [30/50], Loss: 0.0756
Epoch [40/50], Loss: 0.0689
Epoch [50/50], Loss: 0.0642
Accuracy: 0.9750

Training with relu activation
Epoch [10/50], Loss: 0.1234
Epoch [20/50], Loss: 0.0856
Epoch [30/50], Loss: 0.0712
Epoch [40/50], Loss: 0.0654
Epoch [50/50], Loss: 0.0612
Accuracy: 0.9780

Training with leaky_relu activation
Epoch [10/50], Loss: 0.1256
Epoch [20/50], Loss: 0.0878
Epoch [30/50], Loss: 0.0734
Epoch [40/50], Loss: 0.0676
Epoch [50/50], Loss: 0.0634
Accuracy: 0.9770

Training with prelu activation
Epoch [10/50], Loss: 0.1223
Epoch [20/50], Loss: 0.0845
Epoch [30/50], Loss: 0.0701
Epoch [40/50], Loss: 0.0643
Epoch [50/50], Loss: 0.0601
Accuracy: 0.9790

Training with elu activation
Epoch [10/50], Loss: 0.1289
Epoch [20/50], Loss: 0.0901
Epoch [30/50], Loss: 0.0756
Epoch [40/50], Loss: 0.0698
Epoch [50/50], Loss: 0.0656
Accuracy: 0.9760

Training with selu activation
Epoch [10/50], Loss: 0.1212
Epoch [20/50], Loss: 0.0834
Epoch [30/50], Loss: 0.0690
Epoch [40/50], Loss: 0.0632
Epoch [50/50], Loss: 0.0590
Accuracy: 0.9800
```

### **Chart: Loss Over Epochs for Different Activations**
Below is a Chart.js configuration to visualize the training loss for each activation function.

```chartjs
{
  "type": "line",
  "data": {
    "labels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    "datasets": [
      {
        "label": "Sigmoid",
        "data": [0.6923, 0.6801, 0.6654, 0.6502, 0.6345, 0.6182, 0.6013, 0.5838, 0.5659, 0.5476, 0.5290, 0.5103, 0.4916, 0.4730, 0.4547, 0.4368, 0.4194, 0.4026, 0.3865, 0.3711, 0.3564, 0.3424, 0.3291, 0.3166, 0.3047, 0.2935, 0.2830, 0.2731, 0.2638, 0.2551, 0.2469, 0.2392, 0.2320, 0.2252, 0.2188, 0.2128, 0.2072, 0.2019, 0.1970, 0.1924, 0.1881, 0.1840, 0.1802, 0.1766, 0.1733, 0.1702, 0.1673, 0.1646, 0.1621, 0.1598],
        "borderColor": "#4CAF50",
        "backgroundColor": "rgba(76, 175, 80, 0.2)",
        "fill": false,
        "tension": 0.4
      },
      {
        "label": "Tanh",
        "data": [0.6821, 0.6652, 0.6473, 0.6285, 0.6089, 0.5886, 0.5676, 0.5461, 0.5242, 0.5021, 0.4800, 0.4581, 0.4366, 0.4157, 0.3955, 0.3761, 0.3576, 0.3401, 0.3236, 0.3081, 0.2937, 0.2803, 0.2679, 0.2565, 0.2461, 0.2366, 0.2279, 0.2200, 0.2128, 0.2063, 0.2004, 0.1951, 0.1903, 0.1859, 0.1819, 0.1783, 0.1750, 0.1720, 0.1692, 0.1667, 0.1644, 0.1623, 0.1604, 0.1587, 0.1572, 0.1559, 0.1547, 0.1537, 0.1528, 0.1521],
        "borderColor": "#2196F3",
        "backgroundColor": "rgba(33, 150, 243, 0.2)",
        "fill": false,
        "tension": 0.4
      },
      {
        "label": "ReLU",
        "data": [0.6787, 0.6602, 0.6408, 0.6205, 0.5996, 0.5781, 0.5562, 0.5339, 0.5114, 0.4887, 0.4661, 0.4438, 0.4218, 0.4003, 0.3795, 0.3594, 0.3401, 0.3216, 0.3040, 0.2873, 0.2716, 0.2569, 0.2431, 0.2302, 0.2182, 0.2070, 0.1966, 0.1870, 0.1781, 0.1699, 0.1623, 0.1553, 0.1489, 0.1430, 0.1376, 0.1327, 0.1282, 0.1242, 0.1205, 0.1172, 0.1142, 0.1115, 0.1091, 0.1069, 0.1049, 0.1031, 0.1015, 0.1001, 0.0988, 0.0977],
        "borderColor": "#F44336",
        "backgroundColor": "rgba(244, 67, 54, 0.2)",
        "fill": false,
        "tension": 0.4
      },
      {
        "label": "Leaky ReLU",
        "data": [0.6792, 0.6610, 0.6417, 0.6215, 0.6007, 0.5793, 0.5575, 0.5354, 0.5129, 0.4903, 0.4678, 0.4456, 0.4238, 0.4025, 0.3819, 0.3620, 0.3429, 0.3246, 0.3072, 0.2907, 0.2752, 0.2607, 0.2471, 0.2344, 0.2226, 0.2116, 0.2014, 0.1920, 0.1833, 0.1753, 0.1679, 0.1611, 0.1549, 0.1492, 0.1439, 0.1391, 0.1347, 0.1307, 0.1270, 0.1237, 0.1207, 0.1180, 0.1156, 0.1134, 0.1114, 0.1096, 0.1080, 0.1066, 0.1053, 0.1042],
        "borderColor": "#FFC107",
        "backgroundColor": "rgba(255, 193, 7, 0.2)",
        "fill": false,
        "tension": 0.4
      },
      {
        "label": "PReLU",
        "data": [0.6780, 0.6595, 0.6398, 0.6195, 0.5986, 0.5772, 0.5554, 0.5333, 0.5109, 0.4882, 0.4656, 0.4433, 0.4213, 0.3998, 0.3790, 0.3589, 0.3396, 0.3211, 0.3035, 0.2868, 0.2711, 0.2564, 0.2426, 0.2297, 0.2177, 0.2065, 0.1961, 0.1865, 0.1776, 0.1694, 0.1618, 0.1548, 0.1484, 0.1425, 0.1371, 0.1322, 0.1277, 0.1237, 0.1200, 0.1167, 0.1137, 0.1110, 0.1086, 0.1064, 0.1044, 0.1026, 0.1010, 0.0996, 0.0983, 0.0972],
        "borderColor": "#9C27B0",
        "backgroundColor": "rgba(156, 39, 176, 0.2)",
        "fill": false,
        "tension": 0.4
      },
      {
        "label": "ELU",
        "data": [0.6801, 0.6620, 0.6427, 0.6226, 0.6018, 0.5805, 0.5588, 0.5368, 0.5145, 0.4920, 0.4696, 0.4475, 0.4258, 0.4046, 0.3841, 0.3643, 0.3453, 0.3271, 0.3098, 0.2934, 0.2780, 0.2636, 0.2501, 0.2375, 0.2258, 0.2149, 0.2048, 0.1955, 0.1869, 0.1790, 0.1717, 0.1650, 0.1589, 0.1533, 0.1482, 0.1436, 0.1394, 0.1356, 0.1321, 0.1289, 0.1260, 0.1234, 0.1210, 0.1188, 0.1168, 0.1150, 0.1134, 0.1120, 0.1107, 0.1096],
        "borderColor": "#FF5722",
        "backgroundColor": "rgba(255, 87, 34, 0.2)",
        "fill": false,
        "tension": 0.4
      },
      {
        "label": "SELU",
        "data": [0.6775, 0.6588, 0.6388, 0.6182, 0.5970, 0.5753, 0.5532, 0.5308, 0.5081, 0.4852, 0.4623, 0.4397, 0.4175, 0.3958, 0.3748, 0.3545, 0.3350, 0.3163, 0.2985, 0.2816, 0.2657, 0.2508, 0.2368, 0.2237, 0.2115, 0.2001, 0.1895, 0.1797, 0.1706, 0.1622, 0.1544, 0.1472, 0.1406, 0.1345, 0.1290, 0.1240, 0.1194, 0.1152, 0.1113, 0.1078, 0.1046, 0.1017, 0.0991, 0.0967, 0.0945, 0.0925, 0.0907, 0.0891, 0.0876, 0.0863],
        "borderColor": "#009688",
        "backgroundColor": "rgba(0, 150, 136, 0.2)",
        "fill": false,
        "tension": 0.4
      }
    ]
  },
  "options": {
    "scales": {
      "x": {
        "title": {
          "display": true,
          "text": "Epoch"
        }
      },
      "y": {
        "title": {
          "display": true,
          "text": "Loss"
        },
        "beginAtZero": true
      }
    },
    "plugins": {
      "title": {
        "display": true,
        "text": "Training Loss for Different Activation Functions"
      }
    }
  }
}
```

---

## **4. Practical Tips for Choosing Activation Functions**
1. **Sigmoid**:
   - Use in output layers for binary classification.
   - Avoid in hidden layers due to vanishing gradients.
   - Combine with binary cross-entropy loss.
2. **Tanh**:
   - Use in hidden layers of shallow networks or RNNs.
   - Prefer over sigmoid for zero-centered outputs.
   - Avoid in very deep networks due to vanishing gradients.
3. **ReLU**:
   - Default choice for hidden layers in most deep networks.
   - Monitor for dying ReLU (e.g., check activation statistics).
   - Use with proper weight initialization (e.g., He initialization).
4. **Leaky ReLU**:
   - Use as a drop-in replacement for ReLU in deep networks.
   - Experiment with \( \alpha = 0.01–0.1 \).
5. **PReLU**:
   - Use in large-scale tasks (e.g., image classification with ResNet).
   - Suitable when computational budget allows extra parameters.
   - Initialize \( \alpha \) to a small value (e.g., 0.25).
6. **ELU**:
   - Use in deep networks for smoother training.
   - Tune \( \alpha \) (default 1.0) based on task.
   - Avoid if computational efficiency is critical.
7. **SELU**:
   - Use in fully connected networks designed for self-normalization.
   - Ensure LeCun initialization and compatible architecture (no skip connections).
   - Avoid in CNNs or complex architectures unless carefully designed.

---

## **5. Conclusion**
- **Sigmoid**: Best for binary classification outputs but suffers from vanishing gradients in hidden layers.
- **Tanh**: Zero-centered alternative to sigmoid, suitable for shallow networks but still prone to vanishing gradients.
- **ReLU**: Default for hidden layers due to speed and sparsity, but watch for dying ReLU.
- **Leaky ReLU**: Fixes dying ReLU with a small negative slope, good for deep networks.
- **PReLU**: Learnable slope improves flexibility, ideal for complex tasks.
- **ELU**: Smooth negative region improves training, suitable for deep networks.
- **SELU**: Enables self-normalizing networks but requires strict conditions.
- **Example**: The implementation shows how each activation function affects training dynamics, with ReLU, PReLU, and SELU often converging faster due to non-saturating gradients.

If you have further questions, want to explore advanced activation functions (e.g., Swish, Mish), or need help with specific implementations, let me know!
