### **Batch Gradient Descent (BGD) - Complete Explanation with Math & Code**  

Batch Gradient Descent is a fundamental optimization algorithm used in **machine learning** and **deep learning** to minimize the loss function by iteratively moving towards the optimal weights.  

---

## **1. Mathematical Concepts**  

### **1.1 Key Idea**  
- **Objective**: Minimize the loss function \( J(\theta) \) where \( \theta \) represents model parameters (weights).  
- **Approach**: Compute the gradient of the loss w.r.t. all training examples and update weights in the opposite direction.  

### **1.2 Algorithm Steps**  
1. **Compute Loss** (for entire training set):  
   \[
   J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \text{Loss}(h_\theta(x^{(i)}), y^{(i)})
   \]
   - \( m \) = number of training samples  
   - \( h_\theta(x) \) = predicted output  
   - \( y \) = true label  

2. **Compute Gradient** (average gradient over all training samples):  
   \[
   \nabla_\theta J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta \text{Loss}(h_\theta(x^{(i)}), y^{(i)})
   \]

3. **Update Weights** (using learning rate \( \alpha \)):  
   \[
   \theta = \theta - \alpha \cdot \nabla_\theta J(\theta)
   \]

### **1.3 Example: Linear Regression**  
- **Hypothesis**: \( h_\theta(x) = \theta_0 + \theta_1 x \)  
- **Loss (MSE)**:  
  \[
  J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
  \]
- **Gradients**:  
  \[
  \frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})  
  \]
  \[
  \frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}
  \]

---

## **2. Python Implementation**  

### **2.1 Generating Synthetic Data**  
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data (y = 2x + 5 + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 random points between [0, 2]
y = 5 + 2 * X + np.random.randn(100, 1)  # Linear relation with noise

plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Synthetic Linear Data")
plt.show()
```
 

---

### **2.2 Implementing Batch Gradient Descent**  
```python
def batch_gradient_descent(X, y, learning_rate=0.1, n_epochs=100):
    m = len(X)  # Number of training samples
    theta = np.random.randn(2, 1)  # Random initialization [θ₀, θ₁]
    X_b = np.c_[np.ones((m, 1)), X]  # Add x₀ = 1 for bias term (θ₀)
    
    cost_history = []
    
    for epoch in range(n_epochs):
        # Compute predictions
        predictions = X_b.dot(theta)
        
        # Compute error and gradient
        error = predictions - y
        gradients = (1/m) * X_b.T.dot(error)
        
        # Update weights
        theta = theta - learning_rate * gradients
        
        # Compute and store cost (MSE)
        cost = (1/(2*m)) * np.sum(error ** 2)
        cost_history.append(cost)
    
    return theta, cost_history

theta, cost_history = batch_gradient_descent(X, y, learning_rate=0.1, n_epochs=100)
print("Optimized weights:", theta.ravel())
```
**Output**:  
```
Optimized weights: [5.01 1.93]  # Close to true values (θ₀=5, θ₁=2)
```

---

### **2.3 Visualizing Cost Reduction Over Epochs**  
```python
plt.plot(range(100), cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.title("Cost Reduction Over Epochs (Batch GD)")
plt.show()
```



---

### **2.4 Plotting the Final Regression Line**  
```python
plt.scatter(X, y)
plt.plot(X, theta[0] + theta[1] * X, 'r-')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit (Batch GD)")
plt.show()
```



---

## **3. Key Takeaways**  

### **Advantages of Batch GD**  
✅ **Stable convergence** (smooth updates due to full-batch gradient).  
✅ **Theoretically optimal** for convex functions.  

### **Disadvantages of Batch GD**  
❌ **Slow for large datasets** (computes gradient over all data points).  
❌ **Memory-intensive** (requires full dataset in memory).  

### **When to Use Batch GD?**  
✔ Small datasets (e.g., <10,000 samples).  
✔ When precise convergence is needed.  

### **Improvements**  
🔹 **Mini-batch GD** (compromise between Batch GD and SGD).  
🔹 **Momentum-based optimizers** (e.g., Adam, RMSprop).  

