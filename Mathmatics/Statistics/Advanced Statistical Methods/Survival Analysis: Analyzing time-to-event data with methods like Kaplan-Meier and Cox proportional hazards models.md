Survival analysis is a statistical framework for analyzing time-to-event data, where the outcome of interest is the time until an event occurs, such as death, disease onset, or equipment failure. It is widely used in medical research, engineering, and social sciences. Unlike standard regression, survival analysis accounts for **censoring** (when the event is not observed for some subjects) and non-normal time distributions. Below, I’ll explain survival analysis in detail, focusing on the **Kaplan-Meier estimator** and **Cox proportional hazards model**, including their mathematical foundations, implementation, and applications.

---

### **1. Overview of Survival Analysis**

Survival analysis models the time until an event, often called the **survival time** or **time-to-event**. Key features include:
- **Event**: The occurrence of interest (e.g., death, failure, recovery).
- **Censoring**: When the event is not observed during the study period (e.g., a patient is lost to follow-up or the study ends before the event occurs).
- **Right Censoring**: The most common type, where the event has not occurred by the end of observation.
- **Survival Function**: The probability that the event has not occurred by time $\( t \)$, denoted $\( S(t) = P(T > t) \)$, where $\( T \)$ is the survival time.

#### **Key Concepts**
- **Survival Function**: $\( S(t) = P(T > t) \)$, the probability of surviving past time $\( t \)$.
- **Hazard Function**: The instantaneous rate of the event occurring at time $\( t \)$ , given survival up to $\( t \)$:
<img width="333" height="59" alt="image" src="https://github.com/user-attachments/assets/3a35c077-6a62-473f-adcf-081f0bd740ed" />

- **Cumulative Hazard**: $\( H(t) = \int_0^t h(u) \, du \)$ , related to the survival function by:
<img width="175" height="44" alt="image" src="https://github.com/user-attachments/assets/46aef419-207f-44eb-8ff9-c1eb9bc2d421" />

- **Censoring Types**:
  - **Right censoring**: Event time is unknown but known to be after a certain point.
  - **Left censoring**: Event occurred before observation began.
  - **Interval censoring**: Event occurred within a time interval.

---

### **2. Kaplan-Meier Estimator**

The **Kaplan-Meier (KM) estimator** is a non-parametric method to estimate the survival function $\( S(t) \)$ from observed survival times, accounting for censoring.

#### **Mathematical Formulation**
Suppose we have $\( n \)$ subjects with observed times $\( t_1 < t_2 < \dots < t_k \)$, where events (e.g., deaths) occur at distinct times. For each time $\( t_i \)$ :
- $\( d_i \)$ : Number of events (e.g., deaths) at time $\( t_i \)$.
- $\( n_i \)$ : Number of subjects at risk just before $\( t_i \)$ (those who have not experienced the event or been censored).

The KM estimator of the survival function is:

<img width="257" height="53" alt="image" src="https://github.com/user-attachments/assets/12d1d5ea-5f49-49d6-ac08-95d981457056" />

- Each term $\( 1 - \frac{d_i}{n_i} \)$ is the probability of surviving past time $\( t_i \)$, given survival up to $\( t_i \)$.
- The product is taken over all event times $\( t_i \leq t \)$.

#### **Properties**
- **Step Function**: $\( \hat{S}(t) \)$ is a step function that decreases at each event time.
- **Censoring**: Censored observations contribute to $\( n_i \)$ until their censoring time but do not affect $\( d_i \)$.
<img width="988" height="59" alt="image" src="https://github.com/user-attachments/assets/5f3a034e-eb62-4a39-9dd3-6e1ed12cc4fc" />


#### **Applications**
- Estimating survival probabilities in clinical trials (e.g., time to death in cancer patients).
- Comparing survival curves between groups using the **log-rank test**.

#### **Example**
Suppose we have data for 6 patients with survival times (in months) and event indicators (1 = event, 0 = censored):
- Times: 3, 5, 5, 8, 10, 12
- Status: 1, 1, 0, 1, 0, 0

Ordered event times are 3, 5, 8. Compute $\( \hat{S}(t) \)$:
<img width="901" height="179" alt="image" src="https://github.com/user-attachments/assets/6b93977e-d25a-44c0-a258-d9dc2aa26655" />


The survival function decreases at each event time and remains constant between events.

---

### **3. Cox Proportional Hazards Model**

The **Cox proportional hazards (PH) model** is a semi-parametric model that relates covariates to the hazard function without specifying the baseline hazard function.

#### **Mathematical Formulation**
The Cox model assumes the hazard function for subject \( i \) with covariates \( \mathbf{x}_i \) is:

<img width="296" height="50" alt="image" src="https://github.com/user-attachments/assets/b545a4bb-a326-4ee4-8033-24a37c0fcdd7" />

- $\( h_0(t) \)$ : Baseline hazard function (unspecified, non-parametric).
- $\( \mathbf{x}_i \)$ : Vector of covariates (e.g., age, treatment status).
- $\( \boldsymbol{\beta} \)$ : Vector of regression coefficients.
- $\( \exp(\beta_j) \)$ : Hazard ratio for a one-unit increase in covariate $\( x_j \)$, assuming other covariates are held constant.

The **proportional hazards assumption** implies that the hazard ratio $\( \frac{h(t | \mathbf{x}_i)}{h(t | \mathbf{x}_j)} = \exp((\mathbf{x}_i - \mathbf{x}_j)^T \boldsymbol{\beta}) \)$  is constant over time.

#### **Partial Likelihood**
Since $\( h_0(t) \)$ is unspecified, the Cox model uses a **partial likelihood** to estimate $\( \boldsymbol{\beta} \)$ . For $\( k \)$ distinct event times $\( t_1 < t_2 < \dots < t_k \)$, the partial likelihood is:

<img width="372" height="61" alt="image" src="https://github.com/user-attachments/assets/8905a0a3-ed0b-4383-9a20-56b8b8bf82b6" />

- $\( R(t_i) \)$ : Risk set at time $\( t_i \)$ (subjects still at risk, i.e., not censored or failed before $\( t_i \))$.
- The numerator is the hazard for the subject who experiences the event.
- The denominator sums the hazards for all subjects at risk.

The log-partial likelihood is:

<img width="588" height="51" alt="image" src="https://github.com/user-attachments/assets/afdb6d00-0542-4481-a795-d00a1c3e578b" />

Parameters $\( \boldsymbol{\beta} \)$ are estimated by maximizing $\( \ell(\boldsymbol{\beta}) \)$, typically using numerical methods like Newton-Raphson.

#### **Baseline Hazard**
After estimating $\( \boldsymbol{\beta} \)$, the baseline hazard $\( h_0(t) \)$ can be estimated using methods like the **Breslow estimator**:

<img width="266" height="66" alt="image" src="https://github.com/user-attachments/assets/0ca22917-ff5a-44dc-b1a2-0780ce783745" />

The cumulative baseline hazard is:

<img width="223" height="56" alt="image" src="https://github.com/user-attachments/assets/d72b45a4-ebdc-4565-8253-85fbe62a6050" />

The survival function for a subject with covariates \( \mathbf{x} \) is:

<img width="352" height="57" alt="image" src="https://github.com/user-attachments/assets/fb5e556d-ecb7-4ad4-aace-72bea58fdd6d" />

#### **Assumption Checking**
- **Proportional Hazards**: Test using Schoenfeld residuals or time-dependent covariates.
- **Linearity**: Check if continuous covariates have a linear effect on the log-hazard (e.g., using splines).

#### **Applications**
- **Medical Research**: Assessing the effect of treatments or risk factors on survival (e.g., effect of chemotherapy on cancer survival).
- **Reliability Engineering**: Modeling time to failure of mechanical systems.
- **Social Sciences**: Analyzing time to events like job loss or marriage.

#### **Example**
Suppose we model survival time for cancer patients based on age $(\( x_1 \))$ and treatment $(0 = control, 1 = treated, \( x_2 \))$. The Cox model is:

<img width="386" height="60" alt="image" src="https://github.com/user-attachments/assets/b1157104-5f1e-41fc-9822-2058fbd8a12a" />

- Fit the model to estimate $\( \beta_1, \beta_2 \)$.
- Interpret $\( \exp(\beta_2) \)$: The hazard ratio comparing treated to control patients.
- If $\( \exp(\beta_2) = 0.5 \)$ , the treatment reduces the hazard of death by 50%.

---

### **4. Implementation in Practice**

Both Kaplan-Meier and Cox models are implemented in software like R, Python, and SAS. Below is an example in Python using the `lifelines` library:

```python
from lifelines import KaplanMeierFitter, CoxPHFitter
import pandas as pd

# Sample data
data = pd.DataFrame({
    'time': [3, 5, 5, 8, 10, 12],
    'event': [1, 1, 0, 1, 0, 0],
    'age': [25, 30, 45, 22, 50, 60],
    'treatment': [0, 1, 1, 0, 1, 0]
})

# Kaplan-Meier
kmf = KaplanMeierFitter()
kmf.fit(data['time'], event_observed=data['event'])
kmf.plot_survival_function()

# Cox PH Model
cph = CoxPHFitter()
cph.fit(data[['time', 'event', 'age', 'treatment']], duration_col='time', event_col='event')
cph.print_summary()
```

This code estimates the survival function using Kaplan-Meier and fits a Cox model to assess the effect of age and treatment.

---

### **5. Advantages and Limitations**

#### **Kaplan-Meier**
- **Advantages**:
  - Non-parametric, no distributional assumptions.
  - Simple to compute and interpret.
- **Limitations**:
  - Cannot adjust for covariates.
  - Less informative for small sample sizes.

#### **Cox PH Model**
- **Advantages**:
  - Incorporates covariates to explain heterogeneity.
  - Semi-parametric, no need to specify $\( h_0(t) \)$.
- **Limitations**:
  - Assumes proportional hazards, which may not hold.
  - Limited to linear effects unless extended (e.g., with splines).

---

### **6. Extensions and Alternatives**
- **Parametric Models**: Assume a specific distribution for survival times (e.g., Weibull, exponential).
- **Accelerated Failure Time (AFT) Models**: Model the effect of covariates on survival time directly.
- **Time-Dependent Covariates**: Allow covariates or hazards to vary over time.
- **Competing Risks**: Model multiple possible events (e.g., death from different causes).

---

### **7. Conclusion**

Survival analysis is a powerful framework for modeling time-to-event data, handling censoring, and incorporating covariates. The Kaplan-Meier estimator provides a simple, non-parametric estimate of the survival function, while the Cox proportional hazards model allows for covariate adjustment under the proportional hazards assumption. These methods are widely applied in fields like medicine, engineering, and social sciences. For further exploration, I can provide a detailed example with real data, code for alternative models, or guidance on testing assumptions—let me know!
