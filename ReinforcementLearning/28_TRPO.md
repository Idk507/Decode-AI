**Trust Region Policy Optimization (TRPO)**  
(2015, Schulman et al. — foundational on-policy policy-gradient method)

TRPO is the algorithm that introduced the **trust region** idea to deep reinforcement learning, guaranteeing (almost always) **monotonic policy improvement** while allowing large, stable updates — even for high-dimensional nonlinear policies (deep neural nets).

It directly inspired **PPO**, which is simpler and became far more popular, but TRPO remains important for understanding **why** clipping / constraints work and for theoretical stability.

### 1. Core Motivation — Why TRPO Was Needed

Vanilla policy gradient (REINFORCE) suffers from:
- Very high variance (long returns)
- Unpredictable step sizes → sometimes huge destructive updates
- Performance collapse after a promising-looking step

Natural Policy Gradient (Amari 1998, Kakade 2002) improves direction (uses Fisher-information preconditioning), but still lacks guarantees on step size.

TRPO combines:
- Surrogate objective (importance-sampled advantage)
- Hard **trust region constraint** on policy divergence (KL divergence)
- Practical approximations (linear/quadratic + conjugate gradient)

Result → **largest safe step** possible without breaking the local approximation.

### 2. Key Theoretical Guarantee

From Kakade & Langford (2002) and TRPO paper:

Let π_old = current policy, π = new candidate policy.

Expected improvement bound:

η(π) ≥ η(π_old) + L_π_old(π) - C · max_s D_KL(π_old(·|s) || π(·|s))

Where:
- η(·) = true expected return
- L_π_old(π) = surrogate advantage = E_{s~ρ_old, a~π_old} [ (π(a|s)/π_old(a|s)) Â^{π_old}(s,a) ]
- C is a (loose) problem-dependent constant
- D_KL = Kullback-Leibler divergence

**Conclusion**: If you **maximize L** while keeping **KL small enough**, you get **guaranteed monotonic improvement** in true performance η.

TRPO enforces this via **average KL constraint** (easier to estimate than max KL).

### 3. The TRPO Optimization Problem

**Exact (theoretical) form**:

max_θ   L_θ_old(θ) = E_{s,a ~ π_old} [ r(θ) Â_old(s,a) ]

subject to   E_s [ D_KL(π_old(·|s) || π_θ(·|s)) ] ≤ δ     (δ small, e.g. 0.01)

Where r(θ) = π_θ(a|s) / π_old(a|s)   (probability ratio)

### 4. Practical TRPO — Approximations Used

TRPO solves an **approximate version**:

1. Linearize surrogate around θ_old:

L(θ) ≈ gᵀ (θ − θ_old)    where g = ∇_θ L(θ)|_{θ_old}   (= vanilla policy gradient!)

2. Quadratic approximation of KL:

D_KL ≈ ½ (θ − θ_old)ᵀ H (θ − θ_old)

H ≈ Fisher information matrix = E[ ∇logπ ∇logπᵀ ]   (average over samples)

→ Problem becomes:

max   gᵀ x

s.t.   ½ xᵀ H x ≤ δ     (x = θ − θ_old)

**Closed-form solution** (Lagrange multiplier):

x* = √(2δ / (gᵀ H⁻¹ g))   ×   H⁻¹ g

→ This is a **scaled natural gradient** step: direction = H⁻¹ g, size tuned to meet KL ≈ δ exactly.

### 5. How to Compute H⁻¹ g Without Inverting Huge Matrix?

→ **Fisher-vector products** (Hvp) + **Conjugate Gradient (CG)** solver

Hvp(v) = E[ (∇logπ · v) ∇logπ ] + damping term (small identity added for stability)

CG approximates solution to H x = g without full matrix.

Very efficient — scales to deep nets with millions of parameters.

### 6. Safety Net: Backtracking Line Search

Even with CG + theory, approximations can fail → actual KL > δ or no improvement.

TRPO adds **backtracking line search**:

- Compute full step x*
- Try α = 1,  β=0.8, β², β³, … (β ≈ 0.8 typical)
- θ_new = θ_old + α x*
- Accept first step where:
  - surrogate L(θ_new) > L(θ_old)   (improvement)
  - mean KL ≤ δ   (constraint satisfied)

This makes TRPO **extremely conservative** and reliable.

### 7. TRPO Pseudocode (Modern / Spinning Up Style)

```text
for iteration = 1, 2, ...:
    # 1. Collect on-policy data with π_old
    Run policy → collect states, actions, rewards, dones
    
    # 2. Compute advantages & returns
    Â_t = GAE(returns - V(s_t))   # or simple MC return - baseline
    G_t = discounted returns
    
    # 3. Compute vanilla policy gradient g
    g = mean over samples [ ∇_θ log π_old(a|s) * Â ]
    
    # 4. Approximate natural gradient step via CG
    x = conjugate_gradient( Hvp → g )   # solves approx H x = g
    
    # 5. Compute max step size scalar
    approx_kl = 0.5 * g.dot(x)
    step_frac = sqrt(2 * δ / approx_kl)
    full_step = step_frac * x
    
    # 6. Backtracking line search
    for j in 0..max_backtrack:
        α = β^j
        proposed_θ = θ_old + α * full_step
        
        # evaluate surrogate & actual KL
        L_new = compute_surrogate(proposed_θ, data)
        kl_new = compute_mean_kl(proposed_θ, data)
        
        if L_new > L_old and kl_new ≤ δ:
            accept θ ← proposed_θ
            break
    
    # 7. Update value function (critic) via regression on returns
    fit V_φ to minimize (V(s) - G)^2
```

### 8. Comparison: TRPO vs PPO (Quick 2026 View)

| Property                  | TRPO (2015)                              | PPO (2017)                               |
|---------------------------|------------------------------------------|------------------------------------------|
| Constraint                | Hard KL ≤ δ                              | Soft — clipped ratio                     |
| Optimization              | Second-order (CG + Hvp)                  | First-order (Adam)                       |
| Data efficiency           | Single pass over batch                   | Multiple epochs                          |
| Implementation difficulty | Hard (CG, line search)                   | Easy                                     |
| Monotonicity guarantee    | Theoretical + backtracking               | Empirical (very reliable)                |
| Modern usage (2026)       | Rarely implemented directly              | Default on-policy algorithm everywhere   |

### 9. Summary – One-Liner You Can Remember

**TRPO = maximize surrogate advantage L while strictly constraining average KL(old || new) ≤ δ → take largest safe natural-gradient-like step → monotonic improvement almost guaranteed.**

Even though PPO largely replaced it in practice (simpler, faster, similar performance), TRPO is still the cleanest place to learn:
- surrogate objectives
- trust regions
- natural gradients
- why constraints/clipping prevent collapse

