**Group Relative Policy Optimization (GRPO)**  
(Explained in detail — simple terms, math breakdown, comparison to PPO, why it's popular in 2025–2026 for LLM reasoning)

GRPO is a **PPO-inspired, critic-free** reinforcement learning algorithm that became very popular after **DeepSeek-Math** (2024) and especially **DeepSeek-R1 / R1-Zero** (late 2024 – early 2025).  
It is now the dominant method for **Reinforcement Learning with Verifiable Rewards (RLVR)** and post-training of open reasoning models (e.g. DeepSeek series, many Qwen/Math/Phi reasoning variants, etc.).

### 1. Quick Summary – GRPO vs PPO at a Glance

| Aspect                        | PPO (classic, 2017)                              | GRPO (DeepSeek-Math / R1 style, 2024–2025)                  |
|-------------------------------|--------------------------------------------------|-------------------------------------------------------------|
| Needs separate **critic/value** model? | Yes (usually same size as policy)                | **No** — huge memory & compute saving (~40–50%)             |
| Advantage estimation          | Learned value function V(s) → Â = G - V(s) or GAE | **Group-relative**: Â_i = r_i - mean(r_group) or normalized |
| Sampling per prompt           | Usually 1 rollout per prompt                     | **Multiple** (group size G = 4–16 typical) completions      |
| Baseline                      | Learned V(s)                                     | Empirical mean (or normalized mean/std) of the group        |
| KL penalty                    | Optional (sometimes via adaptive β)              | **Explicit** -β D_KL(π_θ || π_ref) in the loss              |
| Clipping                      | Yes (ratio clip 1±ε)                             | Yes — usually same PPO-style clip                           |
| Best for                      | RLHF (preference / human reward)                 | **RLVR** — verifiable math/code/reasoning rewards           |
| Memory footprint              | High (policy + critic + ref)                     | Lower (only policy + ref model)                             |
| Used in (2025–2026)           | Still common, but less for pure reasoning        | Dominant in open reasoning models (DeepSeek, Qwen-Math, etc.) |

### 2. Core Idea – Why GRPO Exists

In classic PPO for LLMs:
- You need a **reward model** (RM) to give scalar r for each completion.
- You also need a **value head / critic** to estimate baseline → reduce variance.
- Critic is almost as big as policy → doubles VRAM usage.

For **reasoning / math / code** tasks:
- Rewards are often **verifiable** (rule-based checker: correct/incorrect format, exact match, process reward, etc.).
- Absolute reward scale varies wildly per prompt (some problems easy → high rewards, hard → low).
- Relative ranking within the same prompt is more reliable than absolute values.

GRPO insight:
> Instead of learning a value function, **just sample several completions for the same prompt**, score them all with the same reward function, and treat the **group mean as the baseline**.  
> This gives a **cheap, on-the-fly, prompt-adaptive baseline** — no extra model needed.

### 3. How GRPO Works – Step by Step

1. **Collect data** (on-policy style):
   - Take prompt x
   - Sample **G completions** {o₁, o₂, ..., o_G} from current policy π_θ_old (or mixture with ref)
   - Compute reward r_i = reward_model(x, o_i) for each (or rule-based verifier)

2. **Compute group-relative advantage** (heart of GRPO):

   Most common simple version (DeepSeek-Math style):

   ```
   mean_r = (1/G) Σ r_i
   Â_i    = r_i - mean_r
   ```

   Sometimes normalized (more stable, used in many impl):

   ```
   mean_r = avg(r_group)
   std_r  = std(r_group)   # or small epsilon if std≈0
   Â_i    = (r_i - mean_r) / (std_r + ε)
   ```

   → Positive if better than average in group, negative if worse.

3. **Surrogate objective** (very PPO-like):

   ```
   L_GRPO(θ) = E[ min( r(θ) Â ,  clip(r(θ), 1-ε, 1+ε) Â )  - β D_KL(π_θ || π_ref) ]
   ```

   Where:
   - r(θ) = π_θ(o|x) / π_old(o|x)   (token-level or sequence-level ratio)
   - clip usually ε = 0.2
   - β ≈ 0.01–0.1 (KL strength, sometimes adaptive)
   - π_ref = reference model (often SFT checkpoint) → prevents collapse

4. **Update**:
   - Multiple epochs over the batch (like PPO)
   - Adam(W) optimizer, gradient clipping, etc.

### 4. Math – Why It Still (Approximately) Works

Recall PPO clipped surrogate maximizes a lower bound on policy improvement.

GRPO uses **empirical group mean as baseline** instead of V(s).

Theoretical justification (loose):
- If group is large enough and policy doesn't change too fast → mean(r_group) ≈ E[r | x] under current policy
- → Â_i ≈ r_i - E[r|x]   (like advantage with perfect baseline)
- Variance is reduced because baseline is conditioned on the exact prompt
- KL penalty + clipping keeps updates conservative

In practice: works surprisingly well even with small G (4–8) for reasoning tasks.

### 5. Typical Implementation Choices (2025–2026 style)

- Group size G = 8–16 (trade-off: more stable advantage vs. more reward model calls)
- Reward = process + outcome (DeepSeek style) or pure outcome
- KL computed w.r.t. SFT reference model
- Sometimes add length penalty or format reward inside verifier
- Normalize advantages per group (helps when reward scales differ wildly)
- Often use **LoRA / QLoRA** on policy (very memory efficient)

### 6. Advantages & Why Everyone Switched for Reasoning

- **~50% less VRAM** — no critic
- **Cheaper** — reward model calls are the bottleneck, but verifiable rewards are fast
- **More stable on hard reasoning** — relative comparison beats noisy absolute scores
- **Easier to get right** — fewer hyperparameters than full PPO (no value loss coeff, no GAE λ tuning)

Disadvantages:
- Needs **multiple samples per prompt** → more inference cost during training
- Less general than PPO (works best when rewards are cheap & reliable)
- Group variance can be high if G small or rewards very binary

### 7. One-Liner Summary You Can Remember

**GRPO = PPO but replace the learned critic with group-mean normalization of rewards from multiple sampled completions of the same prompt → much cheaper, still stable, became the default for math/reasoning RL post-DeepSeek-R1.**



### 1. Full Loss Equations (Exact from Papers + Variants)

**Original GRPO (DeepSeekMath 2024 – token-level, most stable)**

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q,\{o_i\}} \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Bigg[ \min \Big( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}\big(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon\big) \hat{A}_{i,t} \Big) - \beta \, D_{\text{KL}}[\pi_\theta(\cdot | q, o_i^{<t}) || \pi_{\text{ref}}] \Bigg]$$

Variable Definitions

VariableDescription$\theta$The parameters of the policy being optimized.$G$The group size (number of outputs generated per prompt).$r_{i,t}(\theta)$The probability ratio between the current policy and the old policy.$\hat{A}_{i,t}$The Group Relative Advantage, calculated by normalizing rewards within the group.$\epsilon$The clipping hyperparameter (standard in PPO-style objectives).$\beta$The coefficient controlling the KL divergence penalty.$\pi_{\text{ref}}$The reference model (usually the SFT model) used to prevent catastrophic forgetting.Why use GRPO?

Unlike traditional Reinforcement Learning from Human Feedback (RLHF) which requires a separate Critic (Value) model to estimate a baseline, GRPO uses the mean reward of the group $\{o_1, o_2, ..., o_G\}$ as the baseline. This significantly reduces VRAM usage and computational overhead during training.


**DeepSeek-R1 variant (2025 – often sequence-level ratio shown)**

<img width="603" height="80" alt="image" src="https://github.com/user-attachments/assets/a0f3284b-b0a7-41dd-bc8e-31b253c33678" />


<img width="747" height="222" alt="image" src="https://github.com/user-attachments/assets/e8004690-aebc-44f6-b34f-de37bbac69f3" />

- **Process + Outcome** (DeepSeekMath style):
- 
<img width="359" height="120" alt="image" src="https://github.com/user-attachments/assets/5b805eab-e93f-4dc1-9a33-60a99f5ef108" />

  (cumulative future normalized step rewards)

**KL term** (unbiased estimator used in both papers):
<img width="387" height="109" alt="image" src="https://github.com/user-attachments/assets/64c504e1-00f9-4038-8145-f11578270d86" />


**Hyperparameters (paper defaults)**
- \(G\) = group size = 64 (DeepSeekMath) / 8–16 (modern efficient)
- \(\epsilon\) = 0.2
- \(\beta\) = 0.04 (Math) / 0.001–0.01 (R1 + libraries)

### 2. Loss Aggregation Variants (Critical for Stability)

| Mode (used in verl/TRL)       | Equation                                      | When to use                              | Stability |
|-------------------------------|-----------------------------------------------|------------------------------------------|-----------|
| **token-mean** (default 2025–2026) | Average over **all tokens** in batch         | Long CoT, reasoning                      | Best      |
| seq-mean-token-sum            | Mean over sequences, sum tokens               | Short responses                          | OK        |
| seq-mean-token-mean (original paper) | Mean over sequences, mean tokens          | Avoid in long reasoning (unstable)       | Poor      |
| Dr.GRPO style                 | Divide by fixed MAX_TOKENS (e.g. 2048)       | Remove length bias                       | Excellent |

### 3. GRPO Pseudocode (Modern Practical Version)

```text
Algorithm: GRPO Training Loop (2026 style)

for iteration in 1..num_iters:
    # 1. Sample batch of prompts
    prompts = sample_batch(D)
    
    # 2. Rollout group of completions (on-policy)
    for each prompt q:
        {o1, o2, ..., oG} = sample_from_π_old(q, n=G, temp=0.7–1.0)
        r_i = verifier(q, o_i)   # accuracy + format + length penalty
    
    # 3. Compute group-relative advantages (per group!)
    for each group:
        mean_r = average(r_group)
        std_r  = std(r_group) + 1e-8
        for i in group:
            A_i = (r_i - mean_r) / std_r          # or without /std (Dr.GRPO)
            # assign A_i to every token of o_i (or cumulative for process reward)
    
    # 4. Multiple epochs over the data (like PPO)
    for epoch in 1..num_epochs (usually 1–4):
        for mini-batch:
            # forward pass with current π_θ and old π_old + ref
            ratio = exp( logπ_θ - logπ_old )     # per token or per seq
            kl    = unbiased_kl(π_θ, π_ref)
            
            surr1 = ratio * advantages
            surr2 = clip(ratio, 1-ε, 1+ε) * advantages
            
            actor_loss = -mean( min(surr1, surr2) ) + β * mean(kl)
            
            total_loss = actor_loss   # (no critic!)
            optimizer.step()
    
    # 5. Update old policy
    π_old ← π_θ (or soft copy)
```

### 4. Common Tricks – GRPO++ / DAPO Style (2025–2026 Best Practices)

These are the **real-world upgrades** that made GRPO actually work at scale (from DAPO paper, Dr.GRPO, DeepSeek-V3.2, etc.):

1. **Asymmetric / Higher Clipping** (DAPO core trick)
   - Use `[1 - 0.2, 1 + 0.28]` or even `[1-0.2, 1+0.5]`
   - Prevents entropy collapse while allowing exploration

2. **Dynamic Sampling / Zero-Gradient Filtering**
   - Over-sample (e.g. 16 instead of 8)
   - Drop groups where **all** completions have reward=1 (perfect accuracy)
   - Keeps advantages meaningful

3. **Token-Level Loss + Global Normalization** (DAPO + Dr.GRPO)
   - Always prefer `token-mean` or fixed MAX_TOKENS normalization
   - Eliminates length hacking

4. **No Std Normalization** (Dr.GRPO)
   - `A_i = r_i - mean_r` only (remove /std)
   - Much more stable on hard math problems

5. **Adaptive / Domain-Specific KL**
   - Math domain → β ≈ 0 (or very small)
   - Code/general → β = 0.001–0.01
   - Reweighted KL or k3+ estimator for unbiased gradients

6. **Overlong / Truncation Shaping**
   - Soft penalty for responses near max length instead of hard -1
   - Or mask truncated samples completely

7. **Other Production Tricks**
   - Keep top-p/top-k sampling mask (DeepSeek-V3.2)
   - Per-reward-type group normalization (if you have multiple rewards)
   - Large effective batch (512+ prompts)
   - Monitor entropy + response length + validation accuracy every 100 steps

### 5. Ready-to-Run Code Implementation (2026 Style)

**Easiest: Hugging Face TRL GRPOTrainer** (official, one-line)

```python
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from trl.rewards import accuracy_reward  # or your own verifier

dataset = load_dataset("your_math_dataset")

training_args = GRPOConfig(
    num_generations=8,      # G = group size
    max_prompt_length=512,
    max_completion_length=2048,
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=4,   # effective batch = 4 * 8
    gradient_accumulation_steps=8,
    clip_eps=0.2,
    beta=0.01,              # KL coef
    # DAPO-style tricks via extra args if using experimental
)

trainer = GRPOTrainer(
    model=model,
    reward_model=accuracy_reward,   # or custom function
    args=training_args,
    train_dataset=dataset,
    # peft_config=peft_config for QLoRA
)

trainer.train()
```

**Minimal PyTorch Loss Core** (copy-paste into any custom loop – same as TRL/verl)

```python
def grpo_loss(logits, labels, old_logits, ref_logits, rewards, mask):
    # logits shape: [batch, seq_len, vocab]
    logprob = torch.log_softmax(logits, dim=-1)
    old_logprob = torch.log_softmax(old_logits, dim=-1)
    ref_logprob = torch.log_softmax(ref_logits, dim=-1)
    
    # per-token ratio
    ratio = torch.exp(logprob - old_logprob)
    
    # group-relative advantage (already computed outside)
    # advantages shape: [batch] or [batch, seq_len]
    
    # clipped surrogate
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # KL (unbiased)
    kl = ref_logprob - logprob
    kl = torch.exp(kl) - kl - 1
    kl_loss = beta * kl.mean()
    
    return policy_loss + kl_loss
```

**Production-Grade (verl style config snippet)**

```yaml
algorithm:
  adv_estimator: "grpo"
  norm_adv_by_std_in_grpo: False   # Dr.GRPO style

actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean"     # recommended
    use_kl_loss: True
    kl_loss_coef: 0.001
    clip_ratio: 0.2
  ref:
    rollout:
      n: 8                          # group size
```

**One-line recommendation (March 2026)**

- Quick experiment → **TRL GRPOTrainer** + `num_generations=8` + `clip_eps=0.25`
- Maximum performance → **verl + DAPO-style tricks** (token-mean + no-std + dynamic sampling + asymmetric clip)
- Research → implement custom loss above + overlong shaping

That’s **everything** — original equations, all variants, pseudocode, every major trick (DAPO/Dr.GRPO/DeepSeek-V3.2), and production-ready code.

