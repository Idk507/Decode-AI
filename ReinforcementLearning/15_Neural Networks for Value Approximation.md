# Neural Networks for Value Approximation in Reinforcement Learning

Neural Networks for Value Approximation means using a neural network to **estimate how good a situation (state) is** in reinforcement learning when the number of possible situations is extremely large or continuous. Instead of storing values in a table, the agent learns a **prediction model** that outputs the value of any state it sees. Below is a full explanation from intuition to mathematics, written in simple language but covering all technical parts end to end.

---

# 1. Intuition (simple explanation)

Imagine a robot playing a game. Each moment is a **state**. Some states are good because they lead to winning, and some are bad because they lead to losing.

If there were only a few states, the robot could memorize:
- State A → value 5
- State B → value −2

But real environments may have **millions or infinite states** (camera images, sensor readings, positions). Memorizing all values is impossible. So instead, the robot learns a **neural network** that takes the state as input and predicts the value.

This is called **value function approximation using neural networks**.

---

# 2. What is a value function?

The value of a state is written as:

V(s)

It means:

> "If I start in state s and continue acting, how much total reward will I receive in the future?"

Future rewards are discounted using a **discount factor γ (gamma)**:

V(s) = E[r₁ + γ r₂ + γ² r₃ + ...]

where:
- r₁ = reward now
- r₂ = reward next step
- γ (0–1) reduces the importance of far future rewards
- E means average (expected value)

Since we do not know the true value function, we approximate it using a neural network:

V̂(s; θ)

θ represents all the network weights.

---

# 3. Neural network approximation

The neural network takes the state as input and outputs a predicted value:

V̂(s; θ) = NeuralNetwork(s, θ)

Internally, a neural network performs repeated operations:

Layer 1:
h₁ = σ(W₁ s + b₁)

Layer 2:
h₂ = σ(W₂ h₁ + b₂)

Output:
V̂(s) = W₃ h₂ + b₃

where:
- W = weight matrices
- b = bias vectors
- σ = activation function (ReLU, sigmoid, etc.)

These weights θ = {W₁, b₁, W₂, b₂, W₃, b₃} are learned.

---

# 4. Learning target (Temporal Difference learning)

When the agent moves:

state s → next state s'
remember:
- reward received = r

A better estimate of the value should be:

Target = r + γ V̂(s'; θ)

This comes from the Bellman equation.

---

# 5. Error (TD error)

We measure how wrong the network was:

δ = r + γ V̂(s'; θ) − V̂(s; θ)

This is called the **Temporal Difference (TD) error**.

- δ > 0 → state better than expected
- δ < 0 → state worse than expected

---

# 6. Loss function

To train the neural network, we define a squared error:

L(θ) = ½ (Target − V̂(s; θ))²

= ½ (r + γ V̂(s'; θ) − V̂(s; θ))²

The goal is to minimize this loss.

---

# 7. Gradient descent update (core math)

We adjust parameters using gradient descent:

θ ← θ − α ∇θ L(θ)

Compute gradient:

∇θ L = −δ ∇θ V̂(s; θ)

So update becomes:

θ ← θ + α δ ∇θ V̂(s; θ)

This means:
- If prediction was too small → increase weights
- If prediction was too big → decrease weights

Backpropagation computes ∇θ V̂ automatically.

---

# 8. Step-by-step example

Suppose:
- predicted value V̂(s) = 4
- reward r = 2
- predicted next value V̂(s') = 6
- γ = 0.9

Target:
2 + 0.9 × 6 = 7.4

Error:
δ = 7.4 − 4 = 3.4

Gradient descent changes network weights slightly so the prediction moves toward 7.4.

After many updates, predictions become accurate.

---

# 9. Why neural networks are used

Neural networks allow:
- learning from images and sensors
- handling continuous states
- generalizing to unseen states
- representing complex nonlinear relationships

This is why modern reinforcement learning systems use neural networks instead of tables.

---

# 10. Full training pipeline (end-to-end)

The agent observes a state, feeds it into the neural network, predicts the value, takes an action, receives reward and next state, computes the TD target, calculates the prediction error, performs backpropagation to update weights, and repeats this process continuously until predictions stabilize.

---

# 11. Key terms explained simply

State: the current situation  
Reward: feedback received from the environment  
Value function: long-term goodness of a state  
Discount factor γ: importance of future rewards  
Parameters θ: neural network weights  
TD error: difference between expected and actual outcome  
Gradient descent: method to adjust weights to reduce error  

---

# 12. Simple intuitive summary

Neural networks for value approximation mean teaching a neural network to **guess how good a situation is**, checking whether that guess was correct after seeing what happens next, and then slightly adjusting the network so future guesses are better. Over time, the network becomes a powerful predictor of long-term rewards, allowing the agent to make intelligent decisions even in situations it has never seen before.
