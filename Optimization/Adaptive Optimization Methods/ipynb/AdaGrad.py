import numpy as np
import matplotlib.pyplot as plt

# Function and gradient
def f(x):
    return x**2

def grad(x):
    return 2 * x

# AdaGrad parameters
x = 5.0
learning_rate = 0.1
epsilon = 1e-8
G = 0
n_steps = 50

x_history = [x]

for t in range(1, n_steps+1):
    g = grad(x)
    G += g**2
    adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)
    x -= adjusted_lr * g
    x_history.append(x)

# Plot convergence
plt.plot(x_history, label="x values over steps")
plt.xlabel("Iteration")
plt.ylabel("x")
plt.title("AdaGrad Convergence on f(x) = x²")
plt.grid(True)
plt.legend()
plt.show()

print(f"Final value of x: {x:.4f}")
