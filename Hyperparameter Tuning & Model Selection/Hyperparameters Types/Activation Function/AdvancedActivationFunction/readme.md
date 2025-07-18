1. [Swish Activation Function](#1-swish-activation-function)
2. [GELU (Gaussian Error Linear Unit) Activation Function](#2-gelu-gaussian-error-linear-unit-activation-function)
3. [Mish Activation Function](#3-mish-activation-function)
4. [Softplus Activation Function](#4-softplus-activation-function)
5. [SiLU (Sigmoid-Weighted Linear Unit) Activation Function](#5-silu-sigmoid-weighted-linear-unit-activation-function)
6. [Summary and Comparison](#6-summary-and-comparison)

---

 Summary and Comparison

| **Activation Function** | **Range**        | **Use Case**                              | **Advantages**                              | **Problems**                              |
|-------------------------|------------------|-------------------------------------------|---------------------------------------------|-------------------------------------------|
| **Swish**               | (-∞, ∞)         | Deep networks (e.g., Inception)           | Smooth, outperforms ReLU, learnable \(\beta\)| Computationally expensive, not zero-centered |
| **GELU**                | (-∞, ∞)         | Transformers (e.g., BERT, GPT)            | Smooth, probabilistic, high performance      | Computationally expensive, complex         |
| **Mish**                | (-∞, ∞)         | Computer vision (e.g., YOLOv4)            | Smooth, outperforms ReLU in some tasks       | Computationally expensive, less studied    |
| **Softplus**            | (0, ∞)          | Smooth ReLU alternative                   | Smooth, simple derivative                    | No negative outputs, computationally expensive |
| **SiLU**                | (-∞, ∞)         | EfficientNet, mobile networks             | Simple, outperforms ReLU                    | Computationally expensive, not zero-centered |

### Key Mathematical Concepts
- **Smoothness**: Swish, GELU, Mish, Softplus, and SiLU are all smooth and differentiable, avoiding ReLU’s sharp transition at \(x = 0\).
- **Non-Monotonicity**: Swish, GELU, Mish, and SiLU allow negative outputs, reducing the dying neuron problem.
- **Probabilistic Weighting**: GELU’s use of the Gaussian CDF provides a theoretically grounded weighting of inputs.
- **Computational Cost**: All functions involve exponentials (sigmoid, softplus, tanh), making them slower than ReLU but potentially more expressive.
- **Non-Zero-Centered**: None of these functions are zero-centered, which can affect gradient flow compared to tanh.

### Practical Considerations
- **Swish/SiLU**: Good general-purpose alternatives to ReLU, especially in CNNs and efficient architectures.
- **GELU**: The standard for transformers in NLP due to its empirical success.
- **Mish**: Promising for computer vision but less widely adopted.
- **Softplus**: Niche use when smoothness is critical, often as a component in other functions (e.g., Mish).

---

