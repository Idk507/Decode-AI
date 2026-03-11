**DDPG, TD3, SAC**  
(Off-policy actor-critic algorithms for **continuous action spaces**)

These three algorithms are the most influential off-policy methods for continuous control tasks (robotics, MuJoCo, DM Control, autonomous driving, etc.).  
All are **actor-critic**, **off-policy**, and use **deterministic or near-deterministic policies** (except SAC is explicitly stochastic).

### Quick Summary Table

| Property                  | DDPG (2015)                          | TD3 (2018)                                   | SAC (2018)                                      |
|---------------------------|--------------------------------------|----------------------------------------------|-------------------------------------------------|
| Policy type               | Deterministic (μ(s))                | Deterministic (μ(s))                        | Stochastic (Gaussian, reparameterized)         |
| Exploration               | OU noise / Gaussian noise added     | Gaussian noise on target actions            | Entropy bonus → policy learns to explore       |
| Critic(s)                 | 1 Q-network                         | **2 Q-networks** (min used)                 | **2 Q-networks** (min used)                    |
| Overestimation fix        | None → suffers heavily              | **Clipped Double Q-Learning**               | Clipped Double Q + entropy helps               |
| Policy update frequency   | Every critic update                 | **Delayed** (every 2 critic updates)        | Every critic update                            |
| Target policy smoothing   | No                                  | **Yes** (adds noise to target action)       | Built-in (stochastic policy)                   |
| Objective                 | Maximize Q                          | Maximize Q                                  | Maximize **Q + α·entropy** (max entropy RL)    |
| Typical sample efficiency | Moderate–poor                       | Good                                        | **Very good** (often best)                     |
| Stability / Hyperparam sensitivity | Very sensitive                     | Much better than DDPG                       | **Most stable** of the three                   |
| Performance on MuJoCo (2020–2025 benchmarks) | Usually weakest                  | Very strong                                 | Often strongest or very close to TD3           |
| Modern usage (2025)       | Rarely used as baseline             | Still very competitive                      | **Default choice** in many labs + libraries    |

### 1. DDPG – Deep Deterministic Policy Gradient

**Core idea**  
Combine DQN-style Q-learning with deterministic policy gradient → actor directly outputs best action (no sampling during action selection).

**Main components**
- Actor: π_θ(s) → deterministic action
- Critic: Q_φ(s, a) (one network)
- Target networks (soft update τ ≪ 1)
- Exploration: add Ornstein-Uhlenbeck / Gaussian noise to actions during rollouts
- Bellman target:  
  y = r + γ Q_φ'(s', μ_θ'(s') )   ← uses target actor + target critic

**Biggest problem**  
**Severe Q overestimation bias** (like in early DQN before Double DQN)  
→ actor exploits overestimated actions → policy collapse or very poor performance

### 2. TD3 – Twin Delayed Deep Deterministic Policy Gradient  
(Direct successor / strong fix of DDPG – Fujimoto 2018)

**Three key fixes** (all three matter a lot)

1. **Clipped Double Q-Learning**  
   Train **two independent critics** Q₁, Q₂  
   Target:  
   y = r + γ min(Q₁'(s', a'), Q₂'(s', a'))  
   where a' = μ_θ'(s')  
   → reduces overestimation (very effective)

2. **Delayed Policy Updates**  
   Update actor & target networks **less frequently**  
   Typical ratio: update policy every **2** critic updates  
   → critics become more accurate before policy exploits them

3. **Target Policy Smoothing**  
   When computing target action:  
   a' ← μ_θ'(s') + clip(ε),   ε ~ 𝒩(0, σ)  
   → makes Q target smoother → reduces variance & overestimation even more

**Result**  
→ TD3 is dramatically more stable than DDPG  
→ Often 2–5× better final score on MuJoCo

### 3. SAC – Soft Actor-Critic  
(Haarnoja et al. 2018 – maximum entropy RL)

**Core philosophical difference**  
Instead of only maximizing expected reward, also maximize **entropy** of the policy  
→ encourages exploration intrinsically (no need for injected noise)

**Key changes compared to TD3**

- Policy is **stochastic**  
  π_θ(a|s) = 𝒩(μ_θ(s), σ_θ(s))   (usually state-dependent std dev)

- Reparameterization trick for gradients:  
  a = μ_θ(s) + σ_θ(s) ⊙ ε,   ε ~ 𝒩(0,1)

- Actor objective:  
  J(π) = 𝔼 [ Q(s,a) + α log π(a|s) ]   (maximize reward **+** entropy)

- Automatic temperature tuning  
  α is learned so that policy entropy stays near a target value H̄  
  → no manual tuning of exploration scale

- Still uses **two clipped Q-networks** (same trick as TD3)

- No explicit target policy smoothing needed (stochasticity already smooths)

**Advantages**
- Very strong **intrinsic exploration**
- Extremely stable training curves
- Excellent sample efficiency (often best or tied with TD3+)
- Works well in high-dimensional / hard exploration tasks

**Disadvantages**
- Slightly more computationally expensive (stochastic policy + entropy term)
- More hyperparameters (but auto α tuning helps a lot)

### Quick Rule of Thumb (2025–2026)

| Situation                                 | Recommended choice       | Why? |
|-------------------------------------------|---------------------------|------|
| You want **simplest possible** code       | DDPG                     | (but expect worse results) |
| Strong baseline, few lines of code        | **TD3**                  | Very good performance / stability ratio |
| Want **best performance + stability**     | **SAC**                  | Current de-facto standard in most papers |
| Hard exploration, sparse rewards          | SAC                      | Entropy bonus helps a lot |
| Very strict real-time inference (no sampling) | TD3 or DDPG           | Deterministic policy |
| You already have TD3 working well         | Try SAC next             | Usually gains another 10–30% |

### Modern Reality Check (2024–2026 papers & libraries)

- **SAC** is implemented by default in almost every new RL codebase (Stable-Baselines3-Zoo, CleanRL, Tianshou, ACME, etc.)
- **TD3** is still very competitive and sometimes wins on specific tasks
- **DDPG** is mostly used as a “bad baseline” to show how much better the others are


Let me know if you want pseudocode / equations for any of them!
