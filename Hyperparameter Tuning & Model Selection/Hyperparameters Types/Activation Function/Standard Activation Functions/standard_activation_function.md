## Table of Contents
1. [Sigmoid (Logistic) Activation Function](#1-sigmoid-logistic-activation-function)
2. [Hyperbolic Tangent (Tanh) Activation Function](#2-hyperbolic-tangent-tanh-activation-function)
3. [Rectified Linear Unit (ReLU) Activation Function](#3-rectified-linear-unit-relu-activation-function)
4. [Leaky ReLU Activation Function](#4-leaky-relu-activation-function)
5. [Parametric ReLU (PReLU) Activation Function](#5-parametric-relu-prelu-activation-function)
6. [Exponential Linear Unit (ELU) Activation Function](#6-exponential-linear-unit-elu-activation-function)
7. [Scaled Exponential Linear Unit (SELU) Activation Function](#7-scaled-exponential-linear-unit-selu-activation-function)
8. [Summary and Comparison](#8-summary-and-comparison)



##  Summary and Comparison

| **Activation Function** | **Range**              | **Use Case**                              | **Advantages**                              | **Problems**                              |
|-------------------------|------------------------|-------------------------------------------|---------------------------------------------|-------------------------------------------|
| **Sigmoid**             | (0, 1)                | Binary classification (output layer)       | Interpretable probabilities, smooth          | Vanishing gradients, not zero-centered     |
| **Tanh**                | (-1, 1)               | Hidden layers                             | Zero-centered, smooth                       | Vanishing gradients                       |
| **ReLU**                | [0, ∞)                | Hidden layers (default)                   | Fast convergence, sparse, efficient          | Dying ReLU, not zero-centered             |
| **Leaky ReLU**          | (-∞, ∞)               | Hidden layers (fix dying ReLU)            | Prevents dying ReLU, efficient              | Fixed \(\alpha\), not zero-centered       |
| **PReLU**               | (-∞, ∞)               | Deep networks (e.g., ResNet)              | Learnable \(\alpha\), efficient              | Increased complexity, not zero-centered    |
| **ELU**                 | (-α, ∞)               | Hidden layers (robust convergence)         | Avoids dying ReLU, smooth negative inputs   | Computationally expensive                  |
| **SELU**                | (-λ α, ∞)             | Self-normalizing networks                 | Self-normalizing, avoids dying ReLU         | Strict requirements, computationally expensive |

---

### Key Mathematical Concepts
- **Non-Linearity**: All activation functions introduce non-linearity, enabling neural networks to model complex patterns.
- **Gradients**: The derivative of the activation function determines how gradients flow during backpropagation. Functions with small derivatives (e.g., sigmoid, tanh) suffer from vanishing gradients.
- **Sparsity**: ReLU and its variants promote sparsity by outputting zero for negative inputs, reducing computation.
- **Zero-Centered Outputs**: Tanh and ELU (to some extent) are closer to zero-centered, improving gradient flow compared to sigmoid or ReLU.
- **Self-Normalization**: SELU’s scaling ensures stable activations, but requires specific conditions.

### Practical Considerations
- **ReLU** is the go-to choice for most modern networks due to its simplicity and effectiveness.
- **Leaky ReLU/PReLU** are used when dying ReLU is a concern.
- **ELU/SELU** are preferred for robust convergence or self-normalizing properties, but require more computation.
- **Sigmoid/Tanh** are less common in hidden layers but still used in specific cases (e.g., sigmoid for binary classification outputs).

---
