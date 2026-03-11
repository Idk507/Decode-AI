**Trust Region Methods in Reinforcement Learning**  
(TRPO + PPO family – detailed theory, math, pseudocode, implementation notes, and code patterns)

Trust region methods ensure **policy updates remain stable** by restricting how much the new policy π_new can differ from the old policy π_old.  
The most famous members are **TRPO** (2015) and **PPO** (2017).  
Almost every modern on-policy RL codebase uses PPO (or a close variant).

### 1. Core Idea – Why Trust Regions?

Vanilla policy gradient can take destructive steps:

- Gradient looks good on current batch
- After update → policy changes too much
- Performance collapses (very common in deep + continuous control)

**Solution**: Maximize surrogate advantage **while constraining policy divergence**.

Most common constraint = **average KL divergence**:

```
E_{s ~ d_old} [ KL(π_old(·|s) || π_new(·|s)) ] ≤ δ    (δ ≈ 0.01–0.03)
```

### 2. Surrogate Objective (same for TRPO & PPO)

Let r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)   ← probability ratio

Surrogate loss (to **maximize**):

L(θ) = E_t [ r_t(θ) ⋅ Â_t ]

Where Â_t = advantage estimate (usually GAE)

Intuition:  
- Â_t > 0 → we want to increase probability of a_t  
- Â_t < 0 → we want to decrease it  
- r_t(θ) controls **how much** we change the probability

### 3. TRPO – Trust Region Policy Optimization (2015)

**Exact constrained problem** (theoretical goal):

max_θ   E_t [ r_t(θ) Â_t ]  
s.t.    E_s [ KL(π_old || π_θ) ] ≤ δ  
        (and usually also a very small entropy bonus or linear constraint on step size)

**Practical TRPO approximations**:

1. Linearize surrogate around θ_old:  
   L(θ) ≈ gᵀ Δθ    where g = ∇_θ L(θ)|_{old}  (= vanilla policy gradient)

2. Quadratic approximation of KL:  
   KL ≈ ½ Δθᵀ H Δθ  
   H = Fisher information matrix ≈ (1/N) Σ ∇logπ ∇logπᵀ

3. Solve approximate problem:

max   gᵀ x  
s.t.  ½ xᵀ H x ≤ δ

Closed-form (via Lagrange):

x* = (√(2δ / (gᵀ H⁻¹ g))) ⋅ (H⁻¹ g)

→ natural gradient step scaled to exactly meet KL ≈ δ

**How to compute H⁻¹ g without huge matrix?**

→ **Conjugate Gradient (CG)** + **Hessian-vector products**:

H v = E[ (∇logπ ⋅ v) ⋅ ∇logπ ] + damping term  
(very efficient – no full matrix inversion)

**TRPO pseudocode (core loop)**

```text
for iteration = 1, 2, ...:
    collect trajectories with π_old
    
    compute advantages Â_t (GAE or simple)
    compute returns for value loss
    
    g = mean( ∇_θ log π_old(a|s) * Â )               # vanilla PG
    
    compute x = conjugate_gradient( H@f → g )        # approx H⁻¹ g
    
    step_size = √(2δ / (gᵀ x))
    
    proposed_step = step_size * x
    
    # backtracking line search
    for β in {1, 0.8, 0.64, ...}:
        θ_new = θ_old + β * proposed_step
        if  surrogate_improvement > 0  and  mean_KL ≤ δ:
            accept & break
    
    fit value function V_φ (regression on returns)
    
    θ_old ← θ_new
```

**TRPO – main pain points**

- Conjugate gradient + Hessian-vector products → complicated
- Backtracking line search → many forward passes
- Single epoch per data batch → sample inefficient

### 4. PPO – Proximal Policy Optimization (2017)

**Goal**: same monotonic improvement spirit, **much simpler implementation**

Two popular variants (PPO-clip is dominant):

**PPO-Clip** (most used)

Replace hard KL constraint with **clipped surrogate**:

L_clip(θ) = min( r_t(θ) Â_t  ,   clip(r_t(θ), 1-ε, 1+ε) ⋅ Â_t )

ε ≈ 0.1 – 0.2 (most common = 0.2)

Behavior:

- Â_t > 0 and r_t > 1+ε   → clip to (1+ε) Â_t   (prevents too greedy increase)
- Â_t < 0 and r_t < 1-ε   → clip to (1-ε) Â_t   (prevents too aggressive decrease)

→ automatically limits destructive large policy changes

**PPO objective (final – clipped + value + entropy)**

L(θ) = E_t [ L_clip(θ) - c₁ Value_loss + c₂ Entropy ]

(c₁ ≈ 0.5–1.0, c₂ ≈ 0.0–0.01)

**PPO pseudocode – realistic modern version**

```text
for iteration = 1, 2, ...:
    collect rollout buffer with π_old (N steps or M envs)
    
    compute GAE advantages Â_t
    compute returns R_t
    
    for epoch in 1..K_epochs (3–10 typical):
        shuffle buffer
        for mini-batch in buffer:
            r = π_θ(a|s) / π_old(a|s)
            
            surr1 = r * Â
            surr2 = clip(r, 1-ε, 1+ε) * Â
            
            actor_loss  = -min(surr1, surr2).mean()
            critic_loss = (R - V_φ(s))^2 .mean()   # or Huber
            entropy     = π_θ.entropy().mean()
            
            total_loss = actor_loss + c1*critic_loss - c2*entropy
            
            optimizer.zero_grad()
            total_loss.backward()
            grad_clip if wanted
            optimizer.step()
    
    # Early stopping (optional but common)
    if mean_KL > target_kl * 1.5: break epochs early
    
    θ_old ← θ_current   (copy or soft update)
```

### 5. Key Implementation Differences – TRPO vs PPO

| Aspect                     | TRPO                                      | PPO-clip                                  |
|----------------------------|-------------------------------------------|--------------------------------------------|
| Constraint                 | Hard KL ≤ δ                               | Soft – clipping in objective               |
| Optimization               | Second-order (CG + Hvp)                   | First-order (Adam / AdamW)                 |
| Data usage                 | Usually 1 epoch                           | Multiple epochs (3–15)                     |
| Lines of code (core)       | ~200–400 (CG is painful)                  | ~50–120                                    |
| Sample efficiency          | Good                                      | Better (multi-epoch)                       |
| Stability                  | Very good (monotonic theory)              | Excellent in practice                      |
| Modern libraries default   | Almost never                              | Almost always (SB3, CleanRL, Tianshou, etc.) |

### 6. Recommended Modern Patterns (2025–2026 style)

```python
# Very common PPO building blocks (PyTorch)

# 1. GAE advantage + return computation
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    ...
    # standard discounted + lambda trick

# 2. Clipped surrogate (heart of PPO)
ratio = pi_new.log_prob(actions) - pi_old.log_prob(actions)
ratio = ratio.exp()
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages
policy_loss = -torch.min(surr1, surr2).mean()

# 3. Value function loss (often clipped too in modern impl)
value_loss = 0.5 * (returns - values).pow(2).mean()
# or clipped value: torch.clamp(values_old, values - 0.2, values + 0.2)

# 4. Typical hyperparameters 2025
clip_eps     = 0.2
ent_coef     = 0.0 → 0.01 (adaptive sometimes)
vf_coef      = 0.5
lr           = 3e-4
n_epochs     = 10
batch_size   = 64–256
norm_adv     = True   # normalize advantages (helps a lot)
norm_ret     = False  # sometimes
ortho_init   = True   # for policy net
```

### 7. Quick Recommendation – What to Use / Study in 2026

| Goal                               | What to implement / read first          |
|------------------------------------|------------------------------------------|
| Understand theory deeply           | TRPO paper + appendix                   |
| Want cleanest learning code        | Spinning Up PPO (PyTorch)               |
| Want realistic / strong performance| CleanRL PPO or vwxyzjn/ppo-details      |
| Production / fast experiments      | Stable-Baselines3 PPO or rl-zoo3        |
| Minimal code to teach students     | Single-file PPO (~200 lines)            |

**Bottom line (one sentence)**

TRPO = theoretically beautiful trust-region method using natural gradient + hard KL constraint → complicated.  
PPO = practical approximation using **probability ratio clipping** → dramatically simpler, usually equally good or better → became the standard on-policy algorithm.

