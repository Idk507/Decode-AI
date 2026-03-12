**Vanilla Policy Gradient Loss**  
(also known as **REINFORCE** or Monte-Carlo Policy Gradient — the most basic form)

This is the **original, simplest** policy-gradient method (Williams 1992, popularized in Sutton & Barto Ch. 13).  
It directly optimizes the policy parameters θ by **gradient ascent** on expected return, without any critic, baseline, or clipping.

### 1. Core Goal
We want to **maximize** the expected (discounted) return under policy π_θ:

J(θ) = E_{τ ~ π_θ} [ R(τ) ]  
where R(τ) = ∑_{t=0}^T γ^t r_t   (or undiscounted γ=1 in episodic tasks)

### 2. The Famous Vanilla Policy Gradient Theorem (REINFORCE form)

The gradient of performance is:

∇_θ J(θ) = E_{s,a ~ π_θ} [ ∇_θ log π_θ(a|s) ⋅ G_t ]

More precisely (causality version — most used):

∇_θ J(θ) = E [ ∑_{t=0}^T ∇_θ log π_θ(a_t | s_t) ⋅ G_t ]

Where:
- G_t = return from timestep t onward = r_t + γ r_{t+1} + γ² r_{t+2} + …  
  (Monte-Carlo estimate — full future sum, no bootstrapping)
- The expectation is over trajectories sampled from the current policy π_θ

### 3. Why This Works (Intuition)

- ∇_θ log π_θ(a|s) = score function = tells us **which direction to tweak θ** to make action a more/less likely in state s
- Multiply by G_t:
  - If G_t is **high** (good trajectory from here) → push θ to make this action **more probable**
  - If G_t is **low** (bad trajectory) → push θ to make this action **less probable**
- Average over many samples → we reinforce good actions and discourage bad ones

### 4. The "Loss" People Actually Write in Code

In practice, we do **gradient ascent** on J(θ), which is equivalent to **gradient descent on the negative**:

**Vanilla REINFORCE loss** (to **minimize**):

L(θ) = - (1/N) ∑_{i=1}^N ∑_{t=0}^{T_i} [ log π_θ(a_t^i | s_t^i) ⋅ G_t^i ]

- The negative sign flips ascent → descent (standard in PyTorch/TensorFlow)
- Average over episodes (or trajectories) and timesteps
- No advantage, no baseline — pure Monte-Carlo return weighting

**Gradient of this loss** = - policy gradient → minimizing L is same as maximizing J

### 5. Update Rule (Stochastic Gradient Ascent Version)

For each sampled timestep (or whole episode):

θ ← θ + α ⋅ G_t ⋅ ∇_θ log π_θ(a_t | s_t)

In batch form (most common):

θ ← θ + α ⋅ (1/N) ∑_{trajectories} ∑_t G_t ⋅ ∇_θ log π_θ(a_t | s_t)

### 6. Pseudocode – Vanilla REINFORCE (Classic Style)

```text
Initialize policy parameters θ

for iteration = 1, 2, ...:
    for episode = 1 to N_episodes:
        Generate episode: s0, a0, r0, s1, a1, r1, ..., sT, aT, rT
        Compute returns backward:
            G_T = r_T
            for t = T-1 downto 0:
                G_t = r_t + γ G_{t+1}
    
    Compute policy gradient estimate:
        g = 0
        for each timestep t in all episodes:
            g += ∇_θ log π_θ(a_t | s_t) * G_t
    
    Update:
        θ ← θ + (α / total_timesteps) * g    # or normalize differently
```

Modern libraries often batch it and use Adam instead of raw SGD.

### 7. Typical PyTorch-style Loss Snippet (2025–2026)

```python
# Inside training loop, after collecting batch
log_probs = policy.get_log_prob(states, actions)   # shape [batch_size]
returns    = compute_returns(rewards, dones, gamma)  # G_t for each step

# Vanilla REINFORCE "loss" (negative because we minimize)
loss = - (log_probs * returns).mean()              # simplest version

# With per-episode normalization (helps a bit)
# loss = - (log_probs * (returns - returns.mean())).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 8. Key Properties & Problems

| Aspect                  | Vanilla REINFORCE (no baseline)          |
|-------------------------|------------------------------------------|
| Bias                    | Unbiased (correct on average)            |
| Variance                | **Extremely high** (full episode return) |
| Sample efficiency       | Poor (needs many episodes)               |
| On-policy / Off-policy  | On-policy only                           |
| Works with              | Discrete + continuous actions            |
| Still used in 2026?     | Rarely alone — usually with baseline/GAE |

### 9. Quick Evolution Path (from Vanilla)

- Vanilla REINFORCE → add baseline → **REINFORCE with baseline** (huge variance drop)
- Baseline = learned V(s) → **Actor-Critic**
- Multi-step returns + λ → **GAE** (used in A2C, PPO, etc.)
- Clip ratio + multiple epochs → **PPO** (current standard)

**One memorable sentence**:

**Vanilla policy gradient loss = - log π(a|s) × G_t** (averaged) → it literally weights the log-probability of taken actions by how good the full future turned out to be.
