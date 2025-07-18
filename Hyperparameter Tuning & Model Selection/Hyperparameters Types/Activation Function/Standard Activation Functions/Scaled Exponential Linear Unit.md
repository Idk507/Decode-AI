# Scaled Exponential Linear Unit (SELU) Activation Function

## Mathematical Definition

SELU is a scaled version of ELU designed for self-normalizing neural networks:

```math
\text{SELU}(x) = \lambda \begin{cases} 
x & \text{if } x \geq 0 \\
\alpha (e^x - 1) & \text{if } x < 0 
\end{cases}
```

Where:
- λ ≈ 1.0507, α ≈ 1.6733 (fixed constants derived mathematically)
- The scaling factor λ ensures output variance remains stable

## Mathematical Properties

- **Range**: (-λα, ∞)
- **Derivative**:
  ```math
  \frac{d}{dx} \text{SELU}(x) = \lambda \begin{cases} 
  1 & \text{if } x \geq 0 \\
  \alpha e^x & \text{if } x < 0 
  \end{cases}
  ```
- **Self-Normalizing**: When used with proper initialization and network design (e.g., LeCun initialization), SELU maintains activations with mean 0 and variance 1

## Use Cases

- **Self-Normalizing Networks (SNNs)**: Used in networks designed to maintain stable activations without batch normalization
- **Example**: Feedforward networks where normalization is critical

## Advantages

- **Self-Normalizing**: Stabilizes training without additional normalization layers
- **Avoids Dying ReLU**: Like ELU, provides non-zero gradients for negative inputs
- **Smooth for Negative Inputs**: Inherits ELU's smoothness

## Problems

- **Strict Requirements**: Requires specific initialization (e.g., LeCun normal) and network architecture (e.g., no skip connections) to work effectively
- **Computationally Expensive**: Involves exponentials, like ELU
- **Less Flexible**: The fixed λ and α limit adaptability

## Implementation in Python

```python
import numpy as np
import matplotlib.pyplot as plt

# SELU function
def selu(x, alpha=1.6733, lambda_=1.0507):
    return lambda_ * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# Derivative of SELU
def selu_derivative(x, alpha=1.6733, lambda_=1.0507):
    return lambda_ * np.where(x >= 0, 1, alpha * np.exp(x))

# Generate input values
x = np.linspace(-10, 10, 100)
y = selu(x)
y_deriv = selu_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='SELU Function', color='blue')
plt.plot(x, y_deriv, label='SELU Derivative', color='red', linestyle='--')
plt.title('SELU Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

### Code Explanation

- **SELU Function**: Scales ELU by λ, with fixed α = 1.6733 and λ = 1.0507
- **Derivative**: Scales ELU's derivative by λ
- **Visualization**: Similar to ELU but scaled, with a larger output range due to λ

## Key Points

- SELU is specifically designed for self-normalizing neural networks
- The constants λ and α are mathematically derived to ensure proper normalization
- Works best with specific initialization schemes and network architectures
- Provides smooth gradients and avoids vanishing gradient problems
