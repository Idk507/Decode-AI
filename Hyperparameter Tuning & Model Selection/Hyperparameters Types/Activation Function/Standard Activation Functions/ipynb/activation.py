import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.
    Range: (0, 1)
    Use Case: Binary classification (output layer)
    """
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """
    Tanh (Hyperbolic Tangent) activation function.
    Range: (-1, 1)
    Use Case: Hidden layers
    """
    return np.tanh(x)

def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function.
    Range: [0, infinity)
    Use Case: Hidden layers (default)
    """
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    Range: (-infinity, infinity)
    Use Case: Hidden layers (fix dying ReLU)
    Args:
        x: Input array.
        alpha: Slope for negative inputs (default 0.01).
    """
    return np.where(x > 0, x, x * alpha)

def prelu(x, alpha):
    """
    PReLU (Parametric ReLU) activation function.
    Range: (-infinity, infinity)
    Use Case: Deep networks (e.g., ResNet)
    Args:
        x: Input array.
        alpha: Learnable parameter (can be a single value or an array matching x's shape).
    """
    return np.where(x > 0, x, x * alpha)

def elu(x, alpha=1.0):
    """
    ELU (Exponential Linear Unit) activation function.
    Range: (-alpha, infinity)
    Use Case: Hidden layers (robust convergence)
    Args:
        x: Input array.
        alpha: Scaling factor for negative inputs (default 1.0).
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x, alpha=1.6732632423543772, scale=1.0507009873554805):
    """
    SELU (Scaled Exponential Linear Unit) activation function.
    Range: (-lambda * alpha, infinity)
    Use Case: Self-normalizing networks
    Args:
        x: Input array.
        alpha: SELU alpha parameter.
        scale: SELU scaling parameter.
    """
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

if __name__ == "__main__":
    # Example usage:
    test_input = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

    print("Input:", test_input)
    print("--------------------")

    print("Sigmoid:", sigmoid(test_input))
    print("Tanh:", tanh(test_input))
    print("ReLU:", relu(test_input))
    print("Leaky ReLU (alpha=0.1):", leaky_relu(test_input, alpha=0.1))

    # For PReLU, alpha would typically be a learnable parameter in a neural network
    # Here, we'll just use a fixed value for demonstration.
    print("PReLU (alpha=0.2):", prelu(test_input, alpha=0.2))

    print("ELU (alpha=1.0):", elu(test_input))
    print("SELU:", selu(test_input))
