In hypothesis testing, **p-values**, **test statistics**, and **significance levels** $(\(\alpha\))$ are core components that guide decision-making about whether to reject or fail to reject the null hypothesis. Below is a detailed explanation of these concepts, their roles, mathematical foundations, practical examples, and their interplay in hypothesis testing.

---

### **1. Definitions**

#### **a. P-value**
- The **p-value** is the probability of observing a test statistic (or a result at least as extreme) under the assumption that the null hypothesis $(\(H_0\))$ is true.
- It quantifies the strength of evidence against $\(H_0\)$:
  - **Small p-value** (e.g., $\(p \leq 0.05\))$: Strong evidence against $\(H_0\)$, suggesting the alternative hypothesis $(\(H_1\))$ may be true.
  - **Large p-value** (e.g., $\(p > 0.05\))$: Weak evidence against $\(H_0\)$, suggesting insufficient reason to reject it.
- Range: $\(0 \leq p \leq 1\)$.

#### **b. Test Statistic**
- A **test statistic** is a standardized value calculated from sample data that measures how far the observed sample statistic (e.g., sample mean, proportion) is from the value specified in $\(H_0\).$
- It is compared to a theoretical distribution (e.g., z, t, chi-square) to determine the p-value or whether it falls in the critical region.
- Examples:
  - **Z-statistic**: Used for large samples or known population variance.
  - **T-statistic**: Used for small samples with unknown variance.
  - **Chi-square statistic**: Used for categorical data.
  - **F-statistic**: Used for comparing variances (e.g., ANOVA).

#### **c. Significance Level $(\(\alpha\))$**
- The **significance level** $(\(\alpha\))$ is the threshold probability for rejecting $\(H_0\)$, representing the acceptable risk of a **Type I error** (rejecting $\(H_0\)$ when it is true).
- Common values:$\(\alpha = 0.05\) (5%) or \(\alpha = 0.01\) (1%)$.
- It determines the **critical region** (values of the test statistic leading to rejection of $\(H_0\))$ and is used to interpret the p-value:
  - If $\(p \leq \alpha\)$, reject $\(H_0\)$.
  - If $\(p > \alpha\)$, fail to reject $\(H_0\).$

---

### **2. Roles in Hypothesis Testing**

- **Test Statistic**: Quantifies how far the sample data deviates from $\(H_0\)$. It is the starting point for calculating the p-value or comparing to critical values.
- **P-value**: Provides a probabilistic measure of evidence against $\(H_0\)$, allowing researchers to assess whether the observed data is consistent with $\(H_0\)$.
- **Significance Level ($\(\alpha\)$)**: Sets the decision rule for rejecting $\(H_0\)$, balancing the trade-off between Type I and Type II errors.

---

### **3. Mathematical Foundations**

#### **a. Test Statistic**
The test statistic is calculated based on the type of test and parameter being tested. Common formulas include:
- **One-sample z-test (mean, known variance)**:
 <img width="174" height="82" alt="image" src="https://github.com/user-attachments/assets/05c4c7bf-7653-4ab6-9a14-49a32f98d74c" />

  - $\(\bar{x}\):$ Sample mean
  - $\(\mu_0\):$ Hypothesized population mean
  - $\(\sigma\):$ Population standard deviation
  - $\(n\):$ Sample size
- **One-sample t-test (mean, unknown variance)**:
 <img width="170" height="89" alt="image" src="https://github.com/user-attachments/assets/e4da70eb-ad46-4434-90e7-6fa365911c4c" />

  - $\(s\)$: Sample standard deviation
  - $\(df = n - 1\)$
- **Z-test for proportion**:
  <img width="223" height="118" alt="image" src="https://github.com/user-attachments/assets/8a3589cb-8d2c-4523-a665-5650ad6c6a2b" />

  - $\(\hat{p}\):$ Sample proportion
  - $\(p_0\)$: Hypothesized population proportion
- **Chi-square test (goodness-of-fit)**:
  <img width="264" height="94" alt="image" src="https://github.com/user-attachments/assets/49ae73a7-6a21-4c3f-be6b-aebceba67b0f" />

  - $\(O_i\)$: Observed frequencies
  - $\(E_i\)$: Expected frequencies

#### **b. P-value**
- The p-value is derived from the test statistic and the sampling distribution:
<img width="946" height="176" alt="image" src="https://github.com/user-attachments/assets/fcb6c689-21c8-4c2d-baa4-b6cf1b064c3e" />

- The p-value is calculated using statistical tables (e.g., z-table, t-table) or software.

#### **c. Significance Level and Critical Values**
- The significance level $(\(\alpha\))$ defines the critical region:
  - For a two-tailed z-test at $\(\alpha = 0.05\)$, critical values are $\(z = \pm 1.96\)$.
  - For a two-tailed t-test, critical values depend on degrees of freedom $(\(df\))$ and $\(\alpha\) (e.g., \(t_{0.025, 24} \approx 2.064\) for \(df = 24\)).$
- If the test statistic falls in the critical region (e.g., \(|z| > 1.96\)), reject \(H_0\).

---

### **4. Practical Examples**

#### **Example 1: One-Sample T-Test (Mean)**
**Scenario**: A company claims its employees complete a task in 30 minutes on average. A sample of 25 employees has a mean of 32 minutes and a standard deviation of 5 minutes. Test if the population mean differs from 30 minutes at $\(\alpha = 0.05\).$

- **Hypotheses**:
  - $\(H_0: \mu = 30\)$
  - $\(H_1: \mu \neq 30\)$ (two-tailed)

- **Significance Level**: \(\alpha = 0.05\)

- **Test Statistic**:
  <img width="395" height="100" alt="image" src="https://github.com/user-attachments/assets/0ddd630e-9384-4018-9eec-87fd386fb913" />

  - Degrees of freedom: $\(df = 25 - 1 = 24\)$

- **P-value**:
  - Using a t-table or software for $\(t = 2\), \(df = 24\)$, two-tailed: $\(p \approx 0.051\)$.

- **Decision**:
  - Since $\(p = 0.051 > \alpha = 0.05\)$, fail to reject $\(H_0\).$

- **Critical Value Approach**:
  - Critical values for $\(\alpha = 0.05\), \(df = 24\), two-tailed: \(t_{0.025, 24} \approx \pm 2.064\)$.
  - Since $\(|t| = 2 < 2.064\),$ the test statistic is not in the critical region, so fail to reject $\(H_0\).$

- **Interpretation**:
  - There is insufficient evidence at the 5% significance level to conclude that the mean task time differs from 30 minutes.

#### **Example 2: Z-Test for Proportion**
**Scenario**: A poll claims 60% of voters support a candidate. A sample of 200 voters finds 110 (55%) support the candidate. Test if the population proportion differs from 60% at \(\alpha = 0.05\).

- **Hypotheses**:
  - $\(H_0: p = 0.6\)$
  - $\(H_1: p \neq 0.6\)$ (two-tailed)

- **Significance Level**:$\(\alpha = 0.05\)$

- **Test Statistic**:
 <img width="640" height="174" alt="image" src="https://github.com/user-attachments/assets/9331be2a-d3e2-4820-9271-71b3abb27f56" />

- **P-value**:
  - For a two-tailed test: $\(p = 2 \cdot P(Z < -1.445) \approx 2 \cdot 0.074 = 0.148\)$ (using z-table).

- **Decision**:
  - Since $\(p = 0.148 > \alpha = 0.05\),$ fail to reject $\(H_0\).$

- **Critical Value Approach**:
  - Critical values for $\(\alpha = 0.05\)$, two-tailed: $\(z = \pm 1.96\)$.
  - Since $\(|z| = 1.445 < 1.96\)$, the test statistic is not in the critical region, so fail to reject $\(H_0\).$

- **Interpretation**:
  - There is insufficient evidence to conclude that the proportion of voters supporting the candidate differs from 60%.

---

### **5. Interplay Between P-values, Test Statistics, and Significance Levels**
- **Test Statistic**: Calculated first to measure the deviation of the sample statistic from $\(H_0\).$ It follows a specific distribution (e.g., normal, t) under $\(H_0\).$
- **P-value**: Derived from the test statistic, it translates the test statistic into a probability, making it easier to interpret the strength of evidence.
- **Significance Level $(\(\alpha\))$**: Acts as the decision threshold:
  - If the test statistic falls in the critical region (determined by $\(\alpha\))$, or equivalently, if $\(p \leq \alpha\), reject \(H_0\).$
  - The choice of $\(\alpha\)$ balances Type I error (false positives) and Type II error (false negatives).

---

### **6. Assumptions**
- **Random Sampling**: Ensures the sample is representative.
- **Normality** (for z-tests and t-tests):
  - Large sample size $(\(n \geq 30\))$ or normal population for z-tests.
  - Normal population or large sample for t-tests (less critical for large $\(n\)).$
  - For proportions: $\(n p_0 \geq 5\) and \(n (1 - p_0) \geq 5\).$
- **Independence**: Observations are independent.
- **Variance**: Known for z-tests; estimated from sample for t-tests.

---

### **7. Common Misinterpretations**
- **P-value**:
  - **Misconception**: The p-value is the probability that $\(H_0\)$ is true.
  - **Correct**: The p-value is the probability of observing the data (or more extreme) assuming $\(H_0\)$ is true.
  - A small p-value does not prove $\(H_1\)$; it only suggests evidence against $\(H_0\).$
- **Significance Level**:
  - **Misconception**: $\(\alpha = 0.05\)$ means there’s a 5% chance $\(H_0\)$ is false.
  - **Correct**: $\(\alpha\)$ is the probability of rejecting $\(H_0\)$ when it is true (Type I error rate).
- **Test Statistic**:
  - **Misconception**: A large test statistic always means significance.
  - **Correct**: Significance depends on the test statistic’s value relative to the critical region or p-value compared to $\(\alpha\)$.

---

### **8. Practical Considerations**
- **Choosing $\(\alpha\)$**:
  - $\(\alpha = 0.05\)$ is standard but may be adjusted (e.g., $\(\alpha = 0.01\)$ for stricter control of Type I errors in medical trials).
- **P-value Interpretation**:
  - A p-value slightly above $\(\alpha\) (e.g., \(p = 0.051\))$ does not necessarily mean “no effect”; it indicates insufficient evidence to reject $\(H_0\).$
  - Consider practical significance (effect size) alongside statistical significance.
- **Multiple Testing**:
  - Conducting multiple tests increases the overall Type I error rate. Correct using methods like Bonferroni (divide $\(\alpha\)$ by the number of tests).

---

### **9. Applications**
- **Medical Research**: Testing if a new treatment improves outcomes (e.g., lower p-value indicates evidence of effectiveness).
- **Business**: Testing if a marketing campaign increases sales.
- **Social Sciences**: Testing differences in group means or associations between variables.
- **Quality Control**: Testing if a product’s defect rate meets standards.

---

### **10. Tools and Implementation**
- **Software**: R (e.g., `t.test()`, `prop.test()`), Python (SciPy, Statsmodels), SPSS, Excel.
- **Python Example (One-Sample T-Test)**:
```python
from scipy import stats
import numpy as np

# Sample data
sample = np.random.normal(loc=32, scale=5, size=25)  # Simulated data
mu_0 = 30
alpha = 0.05

# Perform t-test
t_stat, p_value = stats.ttest_1samp(sample, mu_0)

print(f"Test Statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
if p_value <= alpha:
    print("Reject H0")
else:
    print("Fail to reject H0")
```

---

### **11. Visualizing Results**
To visualize the test statistic and critical region (e.g., for the t-test example):
- Plot the t-distribution $(\(df = 24\)).$
- Mark the test statistic $(\(t = 2\)).$
- Shade the critical regions $(\(t < -2.064\) and \(t > 2.064\) for \(\alpha = 0.05\), two-tailed).$
If you’d like a chart, I can provide a Chart.js configuration to visualize this.

---

In summary, **p-values**, **test statistics**, and **significance levels** work together to evaluate hypotheses. The test statistic measures deviation from $\(H_0\)$, the p-value quantifies evidence against $\(H_0\)$, and the significance level sets the threshold for decision-making.
