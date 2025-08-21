The **Perceptron Learning Rule** is an algorithm used to train a perceptron, a type of single-layer artificial neural network, to classify inputs into two classes (binary classification). Introduced by Frank Rosenblatt in 1958, it adjusts the weights of the perceptron based on errors in predictions to improve classification accuracy. Unlike the McCulloch-Pitts neuron, the perceptron can learn from data by updating its weights.

### Key Components:
- **Inputs**: $\( x_1, x_2, \ldots, x_n \)$ (real-valued or binary).
- **Weights**: $\( w_1, w_2, \ldots, w_n \)$ (associated with each input).
- **Bias**: A constant term (often treated as a weight $\( w_0 \)$ for a fixed input $\( x_0 = 1 \))$.
- **Output**: Binary (0 or 1) based on a threshold activation function:
  \[
  y = 
  \begin{cases} 
  1 & \text{if } \sum_{i=0}^n w_i x_i > 0 \\
  0 & \text{otherwise}
  \end{cases}
  \]
- **Target**: Desired output $\( t \)$ (0 or 1) for a given input.
- **Learning Rate**: $\( \eta \)$ (a small positive constant, e.g., 0.1), controlling the step size of weight updates.

### Perceptron Learning Rule:
The rule updates weights when the perceptron's prediction $\( y \)$ differs from the target $\( t \)$. For each training example:
1. Compute the weighted sum: $\( \sum_{i=0}^n w_i x_i \)$.
2. Apply the activation function to get the predicted output $\( y \)$.
3. Update weights if $\( y \neq t \)$:
   \[
   w_i \leftarrow w_i + \eta (t - y) x_i \quad \text{for all } i = 0, 1, \ldots, n
   \]
   - If $\( t = 1 \)$ and $\( y = 0 \)$, increase weights to make the sum more positive.
   - If $\( t = 0 \)$ and $\( y = 1 \)$, decrease weights to make the sum more negative.
   - If $\( t = y \)$, no update (weights remain unchanged).

### Algorithm Steps:
1. Initialize weights $\( w_i \)$ and bias $\( w_0 \)$ to small random values or zeros.
2. For each training example $\( (x_1, x_2, \ldots, x_n, t) \)$:
   - Compute output $\( y \)$.
   - Update weights using the learning rule.
3. Repeat for multiple epochs (iterations over the dataset) until convergence (no errors) or a fixed number of iterations.

### Convergence:
- The perceptron learning rule guarantees convergence to a solution (correct weights) if the data is **linearly separable** (a hyperplane can separate the two classes).
- If the data is not linearly separable, the algorithm will not converge and may oscillate.

### Example:
Suppose a perceptron with two inputs $\( x_1, x_2 \)$, weights $\( w_1 = 0.5, w_2 = 0.5 \)$, bias $\( w_0 = -0.5 \)$, and learning rate $\( \eta = 0.1 \)$. Training example: $\( x_1 = 1, x_2 = 1, t = 1 \)$.

1. Compute: $\( \sum w_i x_i = (-0.5) \cdot 1 + 0.5 \cdot 1 + 0.5 \cdot 1 = 0.5 > 0 \)$, so $\( y = 1 \)$.
2. Since $\( y = t \)$, no update.
Now, for $\( x_1 = 1, x_2 = 0, t = 0 \)$:
1. Compute: $\( \sum w_i x_i = (-0.5) \cdot 1 + 0.5 \cdot 1 + 0.5 \cdot 0 = 0 > 0 \)$, so $\( y = 1 \)$.
2. Since $\( y \neq t \)$, update:
   - $\( w_0 \leftarrow -0.5 + 0.1 (0 - 1) \cdot 1 = -0.6 \)$
   - $\( w_1 \leftarrow 0.5 + 0.1 (0 - 1) \cdot 1 = 0.4 \)$
   - $\( w_2 \leftarrow 0.5 + 0.1 (0 - 1) \cdot 0 = 0.5 \)$

### Limitations:
- Works only for **linearly separable** data (e.g., cannot solve XOR without additional layers).
- Binary output limits its use for complex problems.
- Sensitive to learning rate choice (too high: instability; too low: slow convergence).

### Significance:
The perceptron learning rule was a pioneering step in machine learning, introducing the concept of weight updates based on error. It inspired modern neural networks and gradient-based learning, though it’s limited compared to multilayer perceptrons or deep learning models.

If you want a detailed example (e.g., training a perceptron for a specific dataset) or a comparison with other learning rules, let me know!
