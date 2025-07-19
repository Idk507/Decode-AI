import numpy as np
import matplotlib.pyplot as plt

# Bent Identity function
def bent_identity(x):
    return (np.sqrt(x**2 + 1) - 1) / 2 + x

# Derivative of Bent Identity
def bent_identity_derivative(x):
    return x / (2 * np.sqrt(x**2 + 1)) + 1

# Generate input values
x = np.linspace(-10, 10, 100)
y = bent_identity(x)
y_deriv = bent_identity_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Bent Identity Function', color='blue')
plt.plot(x, y_deriv, label='Bent Identity Derivative', color='red', linestyle='--')
plt.title('Bent Identity Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Maxout function with k=2 linear transformations
def maxout(x, w1=1.0, b1=0.0, w2=0.5, b2=0.5):
    linear1 = w1 * x + b1
    linear2 = w2 * x + b2
    return np.maximum(linear1, linear2)

# Derivative of Maxout
def maxout_derivative(x, w1=1.0, b1=0.0, w2=0.5, b2=0.5):
    linear1 = w1 * x + b1
    linear2 = w2 * x + b2
    return np.where(linear1 >= linear2, w1, w2)

# Generate input values
x = np.linspace(-10, 10, 100)
y = maxout(x)
y_deriv = maxout_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Maxout Function', color='blue')
plt.plot(x, y_deriv, label='Maxout Derivative', color='red', linestyle='--')
plt.title('Maxout Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# GLU activation
def glu(x, w=1.0, b=0.0):
    sigmoid = 1 / (1 + np.exp(-(w * x + b)))
    return x * sigmoid

# Derivative of GLU
def glu_derivative(x, w=1.0, b=0.0):
    sigmoid = 1 / (1 + np.exp(-(w * x + b)))
    sigmoid_deriv = sigmoid * (1 - sigmoid)
    return sigmoid + x * sigmoid_deriv * w

# Generate input values
x = np.linspace(-10, 10, 100)
y = glu(x)
y_deriv = glu_derivative(x)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='GLU Function', color='blue')
plt.plot(x, y_deriv, label='GLU Derivative', color='red', linestyle='--')
plt.title('GLU Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Snake activation function
def snake(x):
    return x + np.sin(x)**2

# Derivative of Snake
def snake_derivative(x):
    return 1 + np.sin(2 * x)

# Generate input
x = np.linspace(-10, 10, 100)
y = snake(x)
y_deriv = snake_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Snake Function', color='blue')
plt.plot(x, y_deriv, label='Snake Derivative', color='red', linestyle='--')
plt.title('Snake Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

