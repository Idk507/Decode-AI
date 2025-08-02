The **t-test** is a statistical hypothesis test used to determine whether there is a significant difference between the means of one or two populations, or whether a population mean differs from a hypothesized value, when the population variance is unknown and typically when the sample size is small. It relies on the t-distribution, which accounts for the additional uncertainty introduced by estimating the population variance from sample data. Below is a detailed explanation of the t-test, including its definition, types, assumptions, mathematical implementation, examples, and practical considerations.

---

### **1. Definition**
A t-test is used to test hypotheses about population means when the population variance $(\(\sigma^2\))$ is unknown and must be estimated from the sample. It compares a sample statistic (e.g., mean) to a hypothesized value or another sample mean, using the t-distribution to calculate test statistics and p-values.

- **Purpose**: To determine whether there is sufficient evidence to reject the null hypothesis (\(H_0\)) in favor of the alternative hypothesis $(\(H_1\))$ .
- **Key Feature**: Used when the population variance is unknown or when sample sizes are small (\(n < 30\)), as the t-distribution adjusts for the uncertainty in the variance estimate.

---

### **2. Types of T-Tests**

#### **a. One-Sample T-Test**
<img width="864" height="238" alt="image" src="https://github.com/user-attachments/assets/436341d9-6c94-4e83-a948-6023ed322e7c" />


#### **b. Independent Two-Sample T-Test**
<img width="923" height="368" alt="image" src="https://github.com/user-attachments/assets/4ab369f6-1cba-4c1e-81cc-df77474b741c" />


#### **c. Paired T-Test**
<img width="921" height="220" alt="image" src="https://github.com/user-attachments/assets/b4199061-e55f-433b-b0b2-c3d810648581" />

- Example: Testing if a training program improves employee performance by comparing pre- and post-training scores.

---

### **3. Assumptions**
- **Random Sampling**: Samples must be randomly selected to ensure representativeness.
- **Normality**: The data (or differences for paired tests) should be approximately normally distributed. This is critical for small samples $(\(n < 30\))$; for larger samples, the Central Limit Theorem (CLT) ensures the sampling distribution is approximately normal.
- **Independence**: Observations are independent (except in paired t-tests, where pairs are dependent).
- **Equal Variances (for independent t-test)**: If assuming equal variances, the population variances should be approximately equal (tested using Levene’s test or F-test).
- **Unknown Variance**: The population variance is unknown and estimated from the sample.

---

### **4. Mathematical Implementation**

#### **a. One-Sample T-Test**
<img width="616" height="400" alt="image" src="https://github.com/user-attachments/assets/6ae982ad-bda6-4da8-9058-2ee656f8cc74" />


#### **b. Independent Two-Sample T-Test (Equal Variances)**
<img width="642" height="402" alt="image" src="https://github.com/user-attachments/assets/c6e9e5c8-a774-4a00-a0ec-c1d2829b4e7e" />


#### **c. Independent Two-Sample T-Test (Unequal Variances, Welch’s T-Test)**
<img width="807" height="333" alt="image" src="https://github.com/user-attachments/assets/c9742e15-bdb6-42b1-a6ec-04da68dbbf81" />


#### **d. Paired T-Test**
<img width="648" height="308" alt="image" src="https://github.com/user-attachments/assets/23f45c89-a927-4eb9-86c2-44870dd0a539" />


#### **e. P-value**
<img width="759" height="185" alt="image" src="https://github.com/user-attachments/assets/c287daae-2d77-42a1-9a7a-dd03dad294ab" />

- Use t-tables or software to find the p-value.

---

### **5. Steps in Conducting a T-Test**
1. **State Hypotheses**: Define $\(H_0\) and \(H_1\)$.
2. **Choose Significance Level ($\alpha$)**: Typically $\(\alpha = 0.05\).$
3. **Verify Assumptions**: Check normality (e.g., Shapiro-Wilk test) and, for independent t-tests, equal variances (e.g., Levene’s test).
4. **Calculate Test Statistic**: Use the appropriate formula.
5. **Determine P-value or Critical Value**:
   - P-value: Use software or t-tables.
   - Critical Value: Based on \$\alpha$, $\(df\)$, and test type (e.g., for $\(\alpha = 0.05\), two-tailed, \(df = 20\), critical value \(t \approx \pm 2.086\)).$
6. **Make a Decision**:
   - If $\(p \leq \alpha\)$ or the test statistic falls in the critical region, reject $\(H_0\)$.
   - Otherwise, fail to reject $\(H_0\).$
7. **Interpret Results**: State the conclusion in context.

---

### **6. Practical Example: One-Sample T-Test**

**Scenario**: A school claims its students’ average math score is 75. A sample of 20 students has a mean score of 78 and a standard deviation of 6. Test whether the population mean differs from 75 at $\(\alpha = 0.05\)$.

- **Step 1: Hypotheses**
  - $\(H_0: \mu = 75\)$ (mean score is 75).
  - $\(H_1: \mu \neq 75\)$ (mean score differs, two-tailed).

- **Step 2: Significance Level**
  - $\(\alpha = 0.05\).$

- **Step 3: Test Statistic**
  - Given: $\(\bar{x} = 78\), \(\mu_0 = 75\), \(s = 6\), \(n = 20\).$
  - Degrees of freedom: $\(df = 20 - 1 = 19\).$
<img width="578" height="112" alt="image" src="https://github.com/user-attachments/assets/b225c6d2-6f7f-4503-a365-8a912c39d7c9" />

- **Step 4: P-value**
  - For a two-tailed test, $\(df = 19\), \(t = 2.237\)$: Using a t-table or software, $\(p \approx 0.037\)$.

- **Step 5: Decision**
  - Since $\(p = 0.037 \leq \alpha = 0.05\)$, reject $\(H_0\)$.
  - Alternatively, critical values for $\(\alpha = 0.05\), \(df = 19\), two-tailed: \(t \approx \pm 2.093\). Since \(|t| = 2.237 > 2.093\), reject \(H_0\).$

- **Step 6: Interpretation**
  - There is sufficient evidence at the 5% significance level to conclude that the mean math score differs from 75.

---

### **7. Practical Example: Paired T-Test**

**Scenario**: A fitness program claims to reduce body weight. Ten participants are weighed before and after the program, with differences (before - after): [2, 3, -1, 4, 2, 0, 3, 5, 1, 2]. Test whether the program reduces weight at $\(\alpha = 0.05\)$.

- **Step 1: Hypotheses**
  - $\(H_0: \mu_d = 0\)$ (mean weight difference is zero).
  - $\(H_1: \mu_d > 0\)$ (mean weight difference is positive, right-tailed).

- **Step 2: Significance Level**
  - $\(\alpha = 0.05\)$.

- **Step 3: Test Statistic**
  - Calculate mean difference: $\(\bar{d} = \frac{2 + 3 - 1 + 4 + 2 + 0 + 3 + 5 + 1 + 2}{10} = \frac{21}{10} = 2.1\)$.
  - Calculate standard deviation of differences $(\(s_d\))$:
   <img width="928" height="444" alt="image" src="https://github.com/user-attachments/assets/9a342012-49ca-4946-9876-a888f9e7cbc7" />


- **Step 6: Interpretation**
  - There is sufficient evidence to conclude that the fitness program reduces body weight.

---

### **8. Python Implementation (One-Sample T-Test)**

```python
from scipy import stats
import numpy as np

# Sample data
sample = np.random.normal(loc=78, scale=6, size=20)  # Simulated data
mu_0 = 75
alpha = 0.05

# Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(sample, mu_0)

print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.4f}")
if p_value <= alpha:
    print("Reject H0")
else:
    print("Fail to reject H0")
```

**Output** (approximate, due to random data):
```
T-statistic: 2.237, P-value: 0.037
Reject H0
```

---

### **9. Limitations and Considerations**
- **Normality Assumption**: Critical for small samples $(\(n < 30\))$. Use normality tests (e.g., Shapiro-Wilk) or non-parametric tests (e.g., Mann-Whitney U) if violated.
- **Sample Size**: Small samples increase the variability of the t-statistic, reducing power. For large samples $(\(n \geq 30\))$, the t-test approximates a z-test.
- **Equal Variances (Independent T-Test)**: If variances are unequal, use Welch’s t-test to avoid biased results.
- **Type I and II Errors**:
  - Type I error $(\(\alpha\))$: Risk of rejecting a true $\(H_0\)$, controlled by $\(\alpha\)$.
  - Type II error $(\(\beta\))$: Risk of failing to reject a false $\(H_0\)$, reduced by larger samples or effect sizes.

---

### **10. Applications**
- **Education**: Testing if a teaching method improves test scores.
- **Medical Research**: Comparing treatment effects (e.g., before vs. after).
- **Business**: Comparing performance metrics across groups.
- **Psychology**: Testing differences in behavioral outcomes.

---

In summary, the t-test is a versatile tool for testing hypotheses about means when the population variance is unknown. It includes one-sample, independent two-sample, and paired variants, each suited to specific scenarios.
