**Trust Region Policy Optimization (TRPO) & Related Concepts**  
(Explained simply, step-by-step, with math breakdowns — assuming you know basic policy gradients)

TRPO is one of the most important **on-policy** policy-gradient methods ever invented (Schulman et al. 2015).  
It solved a huge pain point: vanilla policy gradients (REINFORCE / vanilla PG) often take steps that are **too big** → performance **crashes** dramatically.

TRPO guarantees **monotonic improvement** (almost always) by staying inside a **trust region**.

### 1. Core Problem TRPO Solves

Vanilla policy gradient update:

θ_new = θ_old + α ∇_θ J(θ_old)

But:
- Advantage estimates are **noisy**
- Policy can change **radically** in one step (especially with deep nets)
- A "good-looking" gradient direction can actually hurt performance a lot if step is too large

TRPO's answer: **Don't take arbitrary steps — take the largest possible step that still keeps the new policy "close enough" to the old one.**

"Close enough" is measured by **KL divergence** (a kind of distance between probability distributions).

### 2. Theoretical Foundation (Key Insight from Kakade & Langford 2002)

Let π_old be the current policy, π_new the next one.

Expected return of new policy:

J(π_new) = E_{s ~ d_old, a ~ π_new} [A_old(s,a)] + J(π_old)

More precisely (importance sampling identity):

J(π_new) ≥ J(π_old) + E_{s ~ d_old, a ~ π_old} [ r(θ) A^π_old(s,a) ]   -   something × KL(π_old || π_new)

Where r(θ) = π_new(a|s) / π_old(a|s)   (probability ratio)

The exact bound (proven in TRPO paper appendix):

J(π_new) ≥ J(π_old) + E_t [ r(θ) Â_t ]   -   C × max_s KL(π_old(·|s) || π_new(·|s))

Where C is some constant depending on γ, rewards, etc.

**Takeaway**: If you maximize the **surrogate objective**

L(θ) = E_t [ (π_θ(a_t|s_t) / π_old(a_t|s_t)) Â_t ]

**while keeping KL small**, you get **monotonic improvement guarantee** (the real J increases).

### 3. TRPO Optimization Problem (The Real One)

Maximize surrogate advantage:

max_θ   E_t [ r(θ) Â_t ]     (≈ surrogate L(θ))

subject to   E_s [ KL(π_old(·|s) || π_θ(·|s)) ] ≤ δ     (average KL constraint)

δ is small hyperparameter (e.g. 0.01)

This is a **constrained optimization** problem.

### 4. Practical TRPO — How It's Actually Solved (Approximations)

TRPO approximates this with:

1. **Linear approximation** of the surrogate objective around θ_old:

L(θ) ≈ g^T (θ - θ_old)     where g = ∇_θ L(θ) |_{θ_old}   (vanilla policy gradient!)

2. **Quadratic approximation** of the KL divergence (Fisher information matrix):

KL(θ_old, θ) ≈ (1/2) (θ - θ_old)^T H (θ - θ_old)

Where **H** = average Fisher-information matrix:

H = E_s [ ∇_θ log π_old (·|s) ∇_θ log π_old (·|s)^T ]   ≈ empirical average over samples

So the approximate problem becomes:

max_θ   g^T (θ - θ_old)

subject to   (1/2) (θ - θ_old)^T H (θ - θ_old) ≤ δ

This is a **quadratic program** with quadratic constraint.

### 5. Closed-Form Solution (Math Breakdown)

Let x = θ - θ_old

Maximize   g^T x

s.t.       (1/2) x^T H x ≤ δ

Using Lagrange multipliers → solution:

x* = √(2δ / (g^T H^{-1} g) )   ×   H^{-1} g

→ θ_new = θ_old + √(2δ / (g^T H^{-1} g) )   ×   H^{-1} g

This is basically **natural gradient** step:   H^{-1} g   (natural gradient direction)

scaled so that the KL step is **exactly δ**.

### 6. Extra Safety: Backtracking Line Search

Even with the math above, approximation error can happen → sometimes the step still violates monotonicity.

TRPO adds:

- Compute full step α = √(2δ / (g^T H^{-1} g) )
- Try θ = θ_old + α × (H^{-1} g)
- Compute actual surrogate improvement & KL
- If improvement < expected OR KL > δ → backtrack: reduce α by factor β (e.g. 0.8), repeat

This almost always ensures **monotonic improvement**.

### 7. TRPO Summary — What You Actually Implement

1. Collect on-policy rollout with π_old
2. Compute advantages Â_t (using GAE or simple return - V(s))
3. Compute vanilla PG g = (1/N) Σ ∇_θ log π(a|s) Â
4. Compute Fisher-vector product (to approximate H^{-1} g without inverting huge matrix)
   - Use conjugate gradient (CG) to solve Hx = g approximately
5. Compute step size scalar √(2δ / g^T x)
6. Apply natural gradient step + backtracking line search
7. Update value function (critic) separately

Very stable, monotonic, but:
- Computationally heavy (CG + multiple forward/backward passes)
- Hard to implement (especially Fisher products)

### 8. PPO — Proximal Policy Optimization (The Popular Successor)

PPO (Schulman 2017) keeps almost the same idea but makes it **much simpler** and **sample-efficient**.

Two main versions:

**PPO-Clip** (most common):

Instead of KL constraint, **clip the probability ratio** inside the objective:

L^{clip}(θ) = E_t [ min( r_t(θ) Â_t ,   clip(r_t(θ), 1-ε, 1+ε) Â_t ) ]

Where r_t(θ) = π_θ(a|s) / π_old(a|s)

ε ≈ 0.2

Intuition:
- If advantage > 0 → don't let ratio go > 1+ε (don't exploit too aggressively)
- If advantage < 0 → don't let ratio go < 1-ε (don't make bad actions much more likely)

**PPO-Penalty** (less used): add β KL term to objective (adaptive β)

**Why PPO usually wins over TRPO**:
- No conjugate gradient, no Fisher matrix
- Just ordinary Adam/gradient descent
- Multiple epochs over same batch → more data efficient
- Much easier to implement & tune
- Usually matches or beats TRPO performance

### Quick Comparison Table

| Aspect                  | TRPO                              | PPO (Clip)                           |
|-------------------------|-----------------------------------|--------------------------------------|
| Constraint type         | Strict KL ≤ δ                     | Clipped surrogate objective          |
| Monotonicity guarantee  | Theoretical + backtracking        | Empirical (very reliable)            |
| Computation             | Heavy (CG, Fisher products)       | Light (just SGD/Adam)                |
| Sample efficiency       | Good                              | Better (multiple epochs)             |
| Implementation difficulty | Hard                             | Easy                                 |
| Modern usage (2025–2026)| Rarely used directly             | **Default** in most libraries & papers (Stable-Baselines, CleanRL, etc.) |
| Works with recurrent    | Yes                               | Yes                                  |

### Final One-Liner Summary

**TRPO** = "Take biggest safe step inside trust region defined by KL divergence" → stable monotonic improvement, but complicated.  
**PPO** = "TRPO but way simpler: just clip the probability ratio instead of KL constraint" → almost same stability, much easier, became the de-facto on-policy standard.

TRPO is mostly studied for understanding the theory behind stable policy updates.

Want pseudocode, GAE equations, or comparison plots next?
