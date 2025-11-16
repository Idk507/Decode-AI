

## 1) DDPM forward process (definitions)

<img width="717" height="385" alt="image" src="https://github.com/user-attachments/assets/eddbaacf-a022-4a54-b619-9f85d26a25a4" />


---

## 2) Exact posterior $(q(x_{t-1}\mid x_t,x_0))$

Using Gaussian identities, the true posterior of one reverse step is Gaussian:
<img width="370" height="36" alt="image" src="https://github.com/user-attachments/assets/c89964cb-1cbb-4c15-aba2-c5294ed70adc" />

with exact mean and variance:

* Posterior mean:
 <img width="408" height="66" alt="image" src="https://github.com/user-attachments/assets/ff59755c-bd2a-41f8-9cac-9c8680be63f8" />

  (One can obtain this by treating (x_t) as noisy linear observations of (x_{t-1}) and (x_0).)

* Posterior variance:
 <img width="180" height="71" alt="image" src="https://github.com/user-attachments/assets/fdf62d71-82d1-457d-9c6f-f2deccbbe51f" />

(These are standard identities used in DDPM papers.)

We can rewrite the mean in an algebraically convenient form. Substitute $(x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t},\epsilon)$ and simplify — this will be useful next.

---

## 3) Parameterizing reverse mean via $(\epsilon_\theta)$ (practical DDPM parameterization)

In practice many models predict $(\epsilon_\theta(x_t,t,c))$, an estimate of the noise $(\epsilon)$ used to form (x_t). The model may be conditional (on (c)) or unconditional. We derive an expression for the posterior mean $(\tilde\mu_t)$ in terms of the predicted $(\epsilon)$.

From the forward identity:
<img width="304" height="49" alt="image" src="https://github.com/user-attachments/assets/4193238c-6994-4e8a-b5a5-bb89685f7393" />

so solving for (x_0) gives:
<img width="263" height="72" alt="image" src="https://github.com/user-attachments/assets/25aafa18-2bfb-4aba-acbb-fca23b4d9a19" />

Plug this (x_0) into the posterior mean $(\tilde\mu_t(x_t,x_0))$. Doing algebraic substitution (straightforward but worth writing) yields:

<img width="608" height="103" alt="image" src="https://github.com/user-attachments/assets/dc498689-b112-47d3-a5e6-5b7ea69d6daf" />


Collect (x_t) terms and (\epsilon) terms; after simplification one obtains the widely used compact DDPM formula:

<img width="600" height="59" alt="image" src="https://github.com/user-attachments/assets/ed351d74-83e1-4ec6-b7eb-121b02c1f1a3" />


You can verify algebraically that the coefficients reduce to those above; this is the standard identity used in DDPM derivations. (Proof: expand both A_t and B_t and combine — the result is the displayed expression.)

Thus if we had the **true** noise (\epsilon) used to make (x_t), the exact posterior mean equals that expression. Therefore, if we have a model $(\epsilon_\theta(x_t,t,c))$ that predicts $(\epsilon)$, we **plug the prediction** into (★) to produce the learned reverse mean:

<img width="472" height="86" alt="image" src="https://github.com/user-attachments/assets/7d6fb6ea-9503-4830-b0b5-1645959a03d9" />

This $(\mu_\theta)$ is a proxy for $(\tilde\mu_t)$ and is used in sampling.

---

## 4) Reverse sampling step

Given $(\mu_\theta)$ and either a chosen variance schedule $( \Sigma_\theta)$ (often set to a fixed $(\tilde\beta_t I) $or similar), the reverse-step draw is:
<img width="309" height="45" alt="image" src="https://github.com/user-attachments/assets/16a1bd16-9699-43bb-8c95-8b71f87b1d04" />

or in sampling form
<img width="399" height="61" alt="image" src="https://github.com/user-attachments/assets/710141b4-8786-4fba-9c86-e579f1472122" />

Common choices:

* <img width="76" height="31" alt="image" src="https://github.com/user-attachments/assets/d08434a4-2cec-4e78-9891-53842ab7c87e" />
 (use exact posterior variance),
* or other heuristics (e.g., fixed, or learned). Many implementations use <img width="247" height="39" alt="image" src="https://github.com/user-attachments/assets/6101d9b7-f1ec-43d3-a492-75a9f89c5753" />


So the whole procedure: predict $(\epsilon_\theta)$, compute $(\mu_\theta)$ via (1), sample $(x_{t-1})$ as Gaussian around $(\mu_\theta)$.

---

## 5) Classifier-free guidance (CFG) algebra — how it alters the reverse mean

Classifier-free guidance is applied by combining an **unconditional** and a **conditional** model prediction of (\epsilon). Let

* $(\epsilon_c := \epsilon_\theta(x_t,t,c))$ — the conditional noise prediction (conditioned on the desired conditioning (c)),
* $(\epsilon_u := \epsilon_\theta(x_t,t,\varnothing))$ — the unconditional noise prediction (conditioning dropped).

CFG forms a guided noise estimate:
<img width="422" height="49" alt="image" src="https://github.com/user-attachments/assets/07a02448-a098-4031-adad-93f7a1965f95" />

where (w) is the guidance weight (commonly (w>1)).

**Plug $(\epsilon_{\text{guided}})$ into (1)** to get the guided reverse mean:

<img width="607" height="74" alt="image" src="https://github.com/user-attachments/assets/74623312-280e-46d3-b7ce-2588f5655b51" />


Expand $(\epsilon_{\text{guided}})$ :
<img width="488" height="67" alt="image" src="https://github.com/user-attachments/assets/313f87fa-5270-4b74-8865-f6e9ddd3b33f" />


Separate into unconditional mean plus a guidance term. First write the unconditional mean (using (\epsilon_u)):

<img width="269" height="62" alt="image" src="https://github.com/user-attachments/assets/98c5062d-17a3-4ec9-8081-a6fcb8860c42" />


Similarly, the conditional mean (using (\epsilon_c)):
<img width="283" height="74" alt="image" src="https://github.com/user-attachments/assets/8b549d65-45bd-4e5d-8209-e1d2155fb8ff" />


Now express $(\mu_{\text{guided}})$ as an affine combination of $(\mu_u)$ and $(\mu_c)$. Plugging and rearranging:

<img width="888" height="234" alt="image" src="https://github.com/user-attachments/assets/9124ce80-cc57-4b9c-8580-ff70b66edfb4" />


This is the key algebraic fact: **classifier-free guidance linearly interpolates (extrapolates when (w>1)) the unconditional and conditional reverse means**. In words:

* When (w=1): $(\mu_{\text{guided}}=\mu_c)$ (no guidance — pure conditional model).
* When (w=0): $(\mu_{\text{guided}}=\mu_u)$ (drop condition).
* When (w>1): you are extrapolating beyond the conditional mean toward stronger conditioning (this typically increases fidelity to the condition but reduces diversity).

Another instructive rearrangement shows the guided mean as conditional mean plus a correction term proportional to the difference $((\epsilon_c-\epsilon_u))$:

<img width="519" height="152" alt="image" src="https://github.com/user-attachments/assets/eb822662-6cb6-4a34-812f-0f9a44de6115" />


If you prefer it expressed by adding a correction to $(\mu_u)$:

<img width="546" height="66" alt="image" src="https://github.com/user-attachments/assets/b606d084-7130-4c49-af75-94390789849d" />


Both forms are algebraically equivalent; they make clear that CFG applies an additive **bias** to the mean proportional to ((\epsilon_c-\epsilon_u)). Since (\epsilon_c-\epsilon_u) encodes how the conditional model's predicted noise differs from the unconditional's, the guidance pushes the mean toward samples that the conditional model deems more consistent with (c).

---

## 6) Effect on variance and sampling

* **Variance**: classifier-free guidance is applied to the *mean* by blending $(\epsilon)$ predictions. Common implementations keep the reverse variance $(\sigma_t^2)$ unchanged (e.g., $(\sigma_t^2=\tilde\beta_t)$). That is, only the mean is altered — the added Gaussian noise scale is the same. So sampling step remains:
<img width="314" height="38" alt="image" src="https://github.com/user-attachments/assets/c922d049-b9f7-43e6-846e-ad3d6a489f5b" />


* **Interpretation**: guidance amplifies (or reduces) the mean shift toward the conditional manifold while keeping random jitter the same. Since the mean moves, sampling distribution shifts; guidance typically reduces diversity (extrapolation) but increases adherence to condition.

* **Stability**: since (w>1) extrapolates, extreme (w) can push means far away and cause artifacts; practical (w) often in 1.0–2.5 depending on model and task.

---

## 7) Alternative parameterization (predicting (x_0) directly)

Some implementations predict $(\hat x_0^\theta(x_t,t))$ directly. There is a one-to-one mapping between $(\epsilon_\theta)$ and $(\hat x_0^\theta)$:
<img width="370" height="61" alt="image" src="https://github.com/user-attachments/assets/3a1f3148-0320-4c76-a50f-ef58ac51918d" />

Plugging $(\hat x_0)$ into the exact posterior mean formula also yields an expression for $(\mu_\theta)$ in terms of $(\hat x_0)$; classifier-free guidance can be implemented by combining conditional/unconditional $(\hat x_0)$ analogously. Algebraically everything is equivalent (because both parameterizations are linear functions of $(\epsilon))$.

---

## 8) Quick sanity check / compact summary

<img width="782" height="272" alt="image" src="https://github.com/user-attachments/assets/fd6690a3-21ef-48dc-b05b-add6ec191822" />


---

## 9) Intuition & implications

* CFG works because the difference $(\epsilon_c-\epsilon_u)$ encodes how conditioning (c) changes the predicted denoising direction. Multiplying that difference by (w>1) biases the reverse mean more strongly toward the conditional denoising vector.
* Algebraically CFG is *exactly* an affine interpolation/extrapolation of reverse means — not an ad hoc hack — which explains why it behaves predictably and why (w) acts like a simple fidelity/diversity knob.
* Since variance is usually unchanged, CFG alters the posterior mean but not the sampling noise scale; one could experiment with changing $(\sigma_t)$ too, but standard practice leaves it unchanged.

---


