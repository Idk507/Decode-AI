

### 1. Three paradigms of machine learning

| Paradigm              | Feedback / Supervision                  | Goal                              | Examples                          |
|-----------------------|------------------------------------------|-----------------------------------|-----------------------------------|
| Supervised learning   | Correct label / target for each example  | Minimize prediction error         | Classification, regression        |
| Unsupervised learning | No labels — find structure               | Discover patterns / compression   | Clustering, autoencoders, PCA     |
| Reinforcement learning| Scalar reward / punishment (delayed, sparse) | Maximize long-term cumulative reward | Games, robotics, recommendation systems, LLMs with RLHF |

RL is **trial-and-error learning from interaction** — there is no supervisor telling the agent what the correct action is; only a **numerical signal (reward)** telling how good or bad the outcome was.

### 2. Core RL loop (Agent ↔ Environment)

At each discrete time step t = 0, 1, 2, … :

1. Agent observes **state** Sₜ (or partial observation Oₜ)
2. Agent chooses **action** Aₜ
3. Environment responds with:
   - scalar **reward** Rₜ₊₁ ∈ ℝ
   - next state Sₜ₊₁
4. t ← t+1, repeat

Goal: choose actions so that the **expected long-run cumulative (discounted) reward** is maximized.

### 3. Return (objective to maximize)

**Gₜ** = total return starting from time t

**Gₜ = Rₜ₊₁ + γ Rₜ₊₂ + γ² Rₜ₊₃ + … = Σ_{k=0}^∞ γᵏ Rₜ₊₁₊ₖ**

- **γ ∈ [0, 1)** → **discount factor**
  - γ = 0 → myopic (only care about immediate reward)
  - γ → 1 → very far-sighted (infinite-horizon problems)

Most interesting problems use **0.95 ≤ γ ≤ 0.999**

Two task types:

- **Episodic** → natural termination (absorbing state with reward 0 forever) → finite-horizon return
- **Continuing** → never ends → needs γ < 1 for convergence

### 4. Markov Decision Process (MDP) — formal setting

Almost all classical & modern RL theory assumes the environment is (or can be approximated as) an **MDP**:

⟨ S, A, P, R, γ ⟩

- **S** — state space (can be discrete, continuous, image, text embedding, …)
- **A** — action space
- **P(s'|s,a) = Pr{Sₜ₊₁ = s' | Sₜ = s, Aₜ = a}** — transition probability (dynamics)
- **R(s,a) or R(s,a,s')** — expected reward (can be stochastic)
- **γ** — discount

**Markov property** (crucial):  
The future is conditionally independent of the past given the current state  
Pr{Sₜ₊₁, Rₜ₊₁, … | history} = Pr{Sₜ₊₁, Rₜ₊₁, … | Sₜ, Aₜ}

If the true environment is not Markov → we try to make the agent's observation history Markovian (recurrent policies, RNNs, transformers, etc.).

### 5. Policy — what the agent does

**π(a|s)** = probability of selecting action a in state s (stochastic policy)  
or deterministic: π(s) = a

Goal → find **optimal policy π*** such that  
v_π*(s) ≥ v_π(s)   for all s, all π

### 6. Value functions — the core quantities

**State-value function** v_π(s)  
= expected return when starting in s and following π forever

**v_π(s) ≐ E_π [ Gₜ | Sₜ = s ]**

**Action-value function** q_π(s,a)  
= expected return when starting in s, taking a, then following π

**q_π(s,a) ≐ E_π [ Gₜ | Sₜ = s, Aₜ = a ]**

Optimal versions (no policy dependence on right):

**v*(s) = max_π v_π(s)**  
**q*(s,a) = max_π q_π(s,a)**

Fundamental relation (policy improvement theorem):

**π*(s) = argmax_a q*(s,a)**  
→ optimal policy is greedy w.r.t. optimal action-value function

### 7. Bellman equations — the single most important idea in RL

**Bellman expectation equations** (for any fixed policy π)

**v_π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ v_π(s') ]**

**q_π(s,a) = Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ Σ_{a'} π(a'|s') q_π(s',a') ]**

**Bellman optimality equations**

**v*(s) = max_a Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ v*(s') ]**

**q*(s,a) = Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ max_{a'} q*(s',a') ]**

These are **self-consistency / fixed-point equations**.  
Solving them → finding the true value of being in each state/action under optimal/given behavior.

### 8. Four fundamental families of solution methods

| Family                  | Learns what?               | Needs model P,R ? | On-policy / Off-policy | Key classical algos               | Modern deep versions             |
|-------------------------|----------------------------|-------------------|------------------------|-----------------------------------|----------------------------------|
| Dynamic Programming     | Value function             | Yes               | —                      | Value Iteration, Policy Iteration | rarely used directly             |
| Monte-Carlo             | Returns → value            | No                | On-policy              | First-visit / every-visit MC      | less common in deep RL           |
| Temporal-Difference     | Bootstrapped TD targets    | No                | Both                   | SARSA, Q-learning, Expected SARSA | DQN, Rainbow, SAC (partly)       |
| Policy Gradient         | Policy parameters directly | No                | Mostly on-policy       | REINFORCE                         | PPO, A2C/A3C, SAC, TD3           |

**Generalized Policy Iteration (GPI)** — almost all methods alternate between:

- **Policy evaluation** (improve estimate of v_π or q_π)
- **Policy improvement** (make policy better given current value estimates)

Even when done very asynchronously → still converges to optimal under mild conditions.

### 9. Temporal Difference learning — most important practical insight

TD methods combine Monte-Carlo (experience) + DP (bootstrapping).

Basic TD(0) update for value function:

**V(Sₜ) ← V(Sₜ) + α [ Rₜ₊₁ + γ V(Sₜ₊₁) − V(Sₜ) ]**

**TD error** δₜ = Rₜ₊₁ + γ V(Sₜ₊₁) − V(Sₜ)

→ learn prediction error of your own value estimates

**Q-learning** (off-policy, learns optimal Q even with exploratory policy):

**Q(S,A) ← Q(S,A) + α [ R + γ max_{A'} Q(S',A') − Q(S,A) ]**

**SARSA** (on-policy):

**Q(S,A) ← Q(S,A) + α [ R + γ Q(S',A') − Q(S,A) ]**  
(A' is action actually sampled from current policy)

### 10. Policy Gradient Theorem (foundation of modern deep RL)

**∇_θ J(θ) = E [ ∇_θ log π_θ(A|S) ⋅ Q^π(S,A) ]**

REINFORCE uses Monte-Carlo estimate → very high variance.

**Actor-Critic** → learn critic (V or Q) to estimate **advantage** instead of raw return:

**A(s,a) = Q(s,a) − V(s)** (or more stable variants like GAE)

Modern workhorse algorithms (2025):

- **PPO** (Proximal Policy Optimization) — clipped surrogate objective
- **SAC** (Soft Actor-Critic) — maximum entropy RL (adds entropy bonus → better exploration)
- **TD3** / **TD-MPC** — state-of-the-art model-free continuous control

### Quick conceptual map (2025 view)

1. MDP + return + discount
2. Policy π(a|s)
3. Value functions v_π, q_π, v*, q*
4. Bellman equations (expectation & optimality)
5. Generalized Policy Iteration
6. Temporal Difference + bootstrapping
7. Q-learning / SARSA → DQN family
8. Policy Gradient Theorem
9. Actor-Critic + Advantage function
10. Entropy regularization + distributional RL + PPO/SAC → current SOTA

