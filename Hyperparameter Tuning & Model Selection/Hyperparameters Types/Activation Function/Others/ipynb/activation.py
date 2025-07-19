import numpy as np
import matplotlib.pyplot as plt

# Step Activation Function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Generate input
x = np.linspace(-10, 10, 100)
y = step_function(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Step Function', color='blue')
plt.title('Step Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Identity Activation Function
def identity(x):
    return x

# Derivative of Identity
def identity_derivative(x):
    return np.ones_like(x)

# Generate input values
x = np.linspace(-10, 10, 100)
y = identity(x)
y_deriv = identity_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Identity Function', color='blue')
plt.plot(x, y_deriv, label='Identity Derivative', color='red', linestyle='--')
plt.title('Identity Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# RBF Activation Function
def rbf(x, c=0.0, sigma=1.0):
    return np.exp(-((x - c)**2) / (2 * sigma**2))

# Derivative of RBF
def rbf_derivative(x, c=0.0, sigma=1.0):
    return -((x - c) / sigma**2) * rbf(x, c, sigma)

# Generate input values
x = np.linspace(-10, 10, 200)
y = rbf(x)
y_deriv = rbf_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='RBF Function', color='blue')
plt.plot(x, y_deriv, label='RBF Derivative', color='red', linestyle='--')
plt.title('RBF Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Sinc Function
def sinc(x):
    return np.where(x == 0, 1.0, np.sin(x) / x)

# Derivative of Sinc
def sinc_derivative(x):
    return np.where(x == 0, 0.0, (np.cos(x) / x) - (np.sin(x) / x**2))

# Generate input values
x = np.linspace(-10, 10, 200)
y = sinc(x)
y_deriv = sinc_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sinc Function', color='blue')
plt.plot(x, y_deriv, label='Sinc Derivative', color='red', linestyle='--')
plt.title('Sinc Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

