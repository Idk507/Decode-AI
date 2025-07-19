
---

### **1. Standard Neural Networks (MLPs, CNNs, RNNs)**
| **Activation Function** | **Where Used**               | **Why Used**                                                                 |
|-------------------------|------------------------------|------------------------------------------------------------------------------|
| **ReLU**                | Hidden layers (default)      | Fast convergence, avoids saturation (but can cause "dead neurons").         |
| **Leaky ReLU**          | Hidden layers (if ReLU fails)| Prevents dead neurons by allowing small negative outputs (α ≈ 0.01).         |
| **PReLU**               | Deep CNNs (e.g., ResNet)     | Learns the negative slope (α) for better task-specific optimization.        |
| **SELU**                | Self-Normalizing Networks    | Enables automatic normalization of activations in deep networks (no BatchNorm needed). |
| **Swish**               | Deep networks (Google Brain) | Smoother than ReLU, often outperforms it in transformers/CNNs.              |
| **GELU**                | Transformers (BERT, GPT)     | Approximates stochastic regularization (used in SOTA NLP models).           |
| **Mish**                | Computer Vision (YOLOv4)     | Smoother gradient flow than ReLU, improves accuracy.                        |

---

### **2. Output Layers (Task-Specific)**
| **Activation**  | **Where Used**               | **Why Used**                                                                 |
|------------------|------------------------------|------------------------------------------------------------------------------|
| **Sigmoid**      | Binary classification        | Squashes output to [0,1] for probability (logistic regression).             |
| **Tanh**         | Bounded regression (e.g., LSTM gates) | Zero-centered output (-1,1), better gradients than sigmoid.         |
| **Softmax**      | Multi-class classification   | Normalizes outputs to probabilities (sum=1).                                |
| **Linear**       | Regression tasks             | Unbounded output (no squashing).                                            |
| **Sparsemax**    | Attention mechanisms         | Sparse probability distribution (alternative to softmax).                   |

---

### **3. Specialized Architectures**
| **Activation**   | **Where Used**               | **Why Used**                                                                 |
|------------------|------------------------------|------------------------------------------------------------------------------|
| **GLU**          | Transformers (e.g., PaLM)    | Gating mechanism (σ(Wx + b) improves gradient flow in deep networks.        |
| **Maxout**       | Adversarial training         | Learns piecewise linear functions (robust to attacks).                      |
| **Gumbel-Softmax**| Discrete latent variables (VAEs) | Differentiable approximation of categorical sampling.                   |
| **SiLU**         | EfficientNet, Diffusion Models | Smooth ReLU variant (x·σ(x)), balances positive/negative activations.   |
| **Snake**        | Periodic data (e.g., audio)  | Captures oscillatory patterns (x + (1-cos(2x))/2).                          |

---

### **4. Niche Use Cases**
| **Activation**   | **Where Used**               | **Why Used**                                                                 |
|------------------|------------------------------|------------------------------------------------------------------------------|
| **ELU**          | Robust deep learning         | Smooth negative outputs (avoids dying ReLU, slower than ReLU).              |
| **Softplus**     | Probabilistic models         | Smooth ReLU approximation (log(1 + eˣ)).                                    |
| **Bent Identity**| Regression (rare)            | Smoother alternative to ReLU for non-linear fits.                           |
| **RBF**          | Radial Basis Function Networks| Used in kernel methods for distance-based feature mapping.                  |

---

### **5. Emerging Trends (2024)**
| **Activation**   | **Where Used**               | **Why Used**                                                                 |
|------------------|------------------------------|------------------------------------------------------------------------------|
| **ACON**         | Dynamic networks             | Learns to switch between linear/non-linear behavior (e.g., EfficientNetV2). |
| **PAU**          | Scientific ML                | Padé approximant for high-precision modeling (e.g., physics simulations).   |
| **Learnable ReLU**| AutoML pipelines            | Optimizes ReLU parameters (slope/threshold) per layer.                      |

---

### **Key Rules of Thumb**
1. **Default Choice**: ReLU for hidden layers (fast, simple).  
2. **Dying ReLU?** → Leaky ReLU/Swish.  
3. **Output Layers**:  
   - Classification: **Sigmoid** (binary), **Softmax** (multi-class).  
   - Regression: **Linear** (unbounded), **Tanh** (bounded).  
4. **Transformers/NLP**: **GELU** (standard), **GLU** (efficient gating).  
5. **Vision**: **Mish/Swish** (YOLO, EfficientNet).  

---

### **When to Experiment?**
- Training instability → Try **SELU** (self-normalizing nets).  
- Need smooth gradients → **Swish/Mish/ELU**.  
- Discrete outputs → **Gumbel-Softmax**.  
- Periodic data → **Snake/Sinusoidal**.  

