
---

### **1. Standard Activation Functions**
#### **(a) Sigmoid (Logistic)**
- **Formula**:  
  \[
  \sigma(x) = \frac{1}{1 + e^{-x}}
  \]
- **Range**: (0, 1)  
- **Use Case**: Binary classification (output layer).  
- **Problems**: Vanishing gradients, not zero-centered.

#### **(b) Hyperbolic Tangent (Tanh)**
- **Formula**:  
  \[
  \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \]
- **Range**: (-1, 1)  
- **Use Case**: Hidden layers (zero-centered, better than sigmoid).  
- **Problems**: Vanishing gradients for extreme values.

#### **(c) Rectified Linear Unit (ReLU)**
- **Formula**:  
  \[
  \text{ReLU}(x) = \max(0, x)
  \]
- **Range**: [0, ∞)  
- **Use Case**: Default for hidden layers (fast convergence).  
- **Problems**: Dying ReLU (neurons stuck at 0).

#### **(d) Leaky ReLU**
- **Formula**:  
  \[
  \text{LeakyReLU}(x) = \begin{cases} 
  x & \text{if } x \geq 0 \\
  \alpha x & \text{if } x < 0 
  \end{cases}
  \]
  (α ≈ 0.01)  
- **Range**: (-∞, ∞)  
- **Use Case**: Fixes dying ReLU problem.  

#### **(e) Parametric ReLU (PReLU)**
- **Formula**: Like LeakyReLU, but α is **learnable**.  
- **Use Case**: Optimized performance in deep networks (e.g., ResNet).  

#### **(f) Exponential Linear Unit (ELU)**
- **Formula**:  
  \[
  \text{ELU}(x) = \begin{cases} 
  x & \text{if } x \geq 0 \\
  \alpha (e^x - 1) & \text{if } x < 0 
  \end{cases}
  \]
- **Range**: (-α, ∞)  
- **Use Case**: Avoids dying ReLU, smoother for negative inputs.  

#### **(g) Scaled Exponential Linear Unit (SELU)**
- **Formula**:  
  \[
  \text{SELU}(x) = \lambda \begin{cases} 
  x & \text{if } x \geq 0 \\
  \alpha (e^x - 1) & \text{if } x < 0 
  \end{cases}
  \]
  (λ ≈ 1.0507, α ≈ 1.6733)  
- **Use Case**: Self-normalizing networks (SNNs).  

---

### **2. Advanced/Specialized Activation Functions**
#### **(a) Swish**
- **Formula**:  
  \[
  \text{Swish}(x) = x \cdot \sigma(\beta x) \quad (\beta \text{ is learnable or fixed})
  \]
- **Range**: (-∞, ∞)  
- **Use Case**: Outperforms ReLU in some deep networks (Google Brain).  

#### **(b) GELU (Gaussian Error Linear Unit)**
- **Formula**:  
  \[
  \text{GELU}(x) = x \cdot \Phi(x) \quad (\Phi \text{ is Gaussian CDF})
  \]
- **Use Case**: Transformers (e.g., BERT, GPT).  

#### **(c) Mish**
- **Formula**:  
  \[
  \text{Mish}(x) = x \cdot \tanh(\text{softplus}(x))
  \]
- **Use Case**: Computer vision (e.g., YOLOv4).  

#### **(d) Softplus**
- **Formula**:  
  \[
  \text{Softplus}(x) = \log(1 + e^x)
  \]
- **Range**: (0, ∞)  
- **Use Case**: Smooth approximation of ReLU.  

#### **(e) SiLU (Sigmoid-Weighted Linear Unit)**
- **Formula**:  
  \[
  \text{SiLU}(x) = x \cdot \sigma(x)
  \]
- **Use Case**: Variant of Swish (used in EfficientNet).  

---

### **3. Probabilistic Activation Functions**
#### **(a) Softmax**
- **Formula**:  
  \[
  \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
  \]
- **Use Case**: Multi-class classification (output layer).  

#### **(b) Sparsemax**
- **Formula**: Projects input to a sparse probability distribution.  
- **Use Case**: Attention mechanisms (sparse alternatives to softmax).  

#### **(c) Gumbel-Softmax**
- **Formula**: Differentiable approximation of categorical sampling.  
- **Use Case**: Discrete latent variables in VAEs.  

---

### **4. Other Niche Activation Functions**
#### **(a) Bent Identity**
- **Formula**:  
  \[
  \text{BentId}(x) = \frac{\sqrt{x^2 + 1} - 1}{2} + x
  \]
- **Use Case**: Rare, used in regression tasks.  

#### **(b) Maxout**
- **Formula**:  
  \[
  \text{Maxout}(x) = \max(w_1^T x + b_1, w_2^T x + b_2)
  \]
- **Use Case**: Learns piecewise linear functions.  

#### **(c) GLU (Gated Linear Unit)**
- **Formula**:  
  \[
  \text{GLU}(x) = x \cdot \sigma(Wx + b)
  \]
- **Use Case**: Natural language processing (e.g., Transformer variants).  

#### **(d) Snake**
- **Formula**:  
  \[
  \text{Snake}(x) = x + \frac{1 - \cos(2x)}{2}
  \]
- **Use Case**: Periodic data modeling.  

---

### **5. Activation Functions for Specific Architectures**
- **Step Function**: Binary classifiers (perceptrons).  
- **Identity**: Regression tasks (output layer).  
- **Radial Basis Function (RBF)**: RBF networks.  
- **Sinc**: Signal processing.  

---

### **Key Considerations When Choosing Activation Functions**
1. **Vanishing Gradients**: Avoid sigmoid/tanh in deep networks.  
2. **Sparsity**: ReLU/LeakyReLU induce sparsity.  
3. **Differentiability**: Needed for gradient-based optimization.  
4. **Computational Cost**: Simpler functions (ReLU) are faster.  
5. **Dead Neurons**: Use LeakyReLU/Swish to mitigate.  

---

### **Recent Trends (2024)**
- **Learnable Activations**: E.g., **PAU (Padé Activation Unit)**.  
- **Dynamic Activations**: Adapt parameters during training.  
- **Attention-Based Activations**: E.g., **ACON (Adaptively CONtinuous)**.  
