Hypothesis testing is a fundamental method in inferential statistics used to make decisions about population parameters based on sample data. It involves formulating hypotheses, collecting data, and using statistical methods to determine whether the evidence supports rejecting or failing to reject a specific claim about the population. Below is a detailed explanation of hypothesis testing, including its key concepts, steps, mathematical implementation, examples, assumptions, types of tests, and applications.

---

### **1. Definition**
Hypothesis testing is a statistical process for evaluating claims about a population parameter (e.g., mean, proportion, variance) using sample data. It tests a **null hypothesis** (a statement of no effect or no difference) against an **alternative hypothesis** (a statement of an effect or difference) and quantifies the evidence using probability.

- **Purpose**: To determine whether there is sufficient evidence in the sample to reject the null hypothesis in favor of the alternative, while controlling the risk of errors.
- **Outcome**: Either reject the null hypothesis or fail to reject it (note: failing to reject does not prove the null hypothesis is true).

---

### **2. Key Concepts**

#### **a. Hypotheses**
- **Null Hypothesis $(\(H_0\))$**: The default assumption, typically stating no effect, no difference, or no relationship.
  - Example: $\(H_0: \mu = 50\)$ (the population mean is 50).
- **Alternative Hypothesis $(\(H_1\) or \(H_a\))$**: The claim to be tested, stating an effect, difference, or relationship.
  - Example: $\(H_1: \mu \neq 50\)$ (the population mean is not 50).
- Types of alternative hypotheses:
  - **Two-tailed**: Tests for any difference (e.g., $\(H_1: \mu \neq 50\)$).
  - **One-tailed (left-tailed)**: Tests for a decrease (e.g., $\(H_1: \mu < 50\)$).
  - **One-tailed (right-tailed)**: Tests for an increase (e.g., $\(H_1: \mu > 50\)$).

#### **b. Significance Level $(\(\alpha\))$**
- The probability of rejecting $\(H_0\)$ when it is true (Type I error rate).
- Common values: $\(\alpha = 0.05\)$ (5%) or $\(\alpha = 0.01\)$ (1%).
- Represents the threshold for "statistically significant" results.

#### **c. Test Statistic**
- A value calculated from sample data that measures how far the sample statistic is from the value specified in $\(H_0\).$
- Common test statistics:
  - **Z-statistic**: Used when the population variance is known or sample size is large $(\(n \geq 30\)).$
  - **T-statistic**: Used when the population variance is unknown and sample size is small.
  - **F-statistic**: Used for comparing variances (e.g., ANOVA).
  - **Chi-square statistic**: Used for categorical data or variance tests.

#### **d. P-value**
- The probability of observing a test statistic at least as extreme as the one calculated, assuming \(H_0\) is true.
- Decision rule:
  - If $\(p \leq \alpha\)$, reject $\(H_0\).$
  - If $\(p > \alpha\)$, fail to reject $\(H_0\).$
- Smaller p-values indicate stronger evidence against $\(H_0\).$

#### **e. Types of Errors**
- **Type I Error**: Rejecting $\(H_0\)$ when it is true $(false positive, probability = \(\alpha\)).$
- **Type II Error**: Failing to reject $\(H_0\)$ when it is false (false negative, probability = $\(\beta\)).$
- **Power**: The probability of correctly rejecting $\(H_0\)$ when it is false $(\(1 - \beta\))$. Power increases with larger sample sizes, larger effect sizes, and smaller $\(\alpha\).$

#### **f. Critical Region and Critical Value**
- The **critical region** is the set of test statistic values leading to rejection of \(H_0\).
- The **critical value** is the boundary of the critical region, determined by $\(\alpha\)$ and the test type (e.g., z = 1.96 for a two-tailed test at $\(\alpha = 0.05\)).$
- If the test statistic falls in the critical region, reject $\(H_0\).$

---

### **3. Steps in Hypothesis Testing**

1. **State the Hypotheses**:
   - Define $\(H_0\)$ and $\(H_1\)$ clearly, specifying the parameter and test type (one-tailed or two-tailed).
2. **Choose the Significance Level (\(\alpha\))**:
   - Typically $\(\alpha = 0.05\)$ or 0.01.
3. **Select the Appropriate Test**:
   - Based on the parameter, sample size, and data type (e.g., z-test, t-test, chi-square test).
4. **Calculate the Test Statistic**:
   - Use sample data and the appropriate formula.
5. **Determine the P-value or Critical Value**:
   - Compare the test statistic to the critical value or compute the p-value.
6. **Make a Decision**:
   - Reject $\(H_0\)$ if $\(p \leq \alpha\)$ or if the test statistic falls in the critical region; otherwise, fail to reject $\(H_0\).$
7. **Interpret the Results**:
   - State the conclusion in the context of the problem, considering practical significance.

---

### **4. Common Hypothesis Tests**

#### **a. Z-Test (Mean, Known Variance)**
- Used when the population variance $(\(\sigma^2\))$ is known and the sample size is large $(\(n \geq 30\))$.
- Test statistic:
  <img width="169" height="106" alt="image" src="https://github.com/user-attachments/assets/42f2e6ef-b4b0-4584-99e8-7da9e9ae055d" />

  - $\(\bar{x}\)$: Sample mean
  - $\(\mu_0\)$: Hypothesized population mean under \(H_0\)
  - $\(\sigma\)$: Population standard deviation
  - $\(n\)$: Sample size

#### **b. T-Test (Mean, Unknown Variance)**
- **One-sample t-test**:
  - Tests if the sample mean differs from a hypothesized value.
  - Test statistic:
    <img width="138" height="98" alt="image" src="https://github.com/user-attachments/assets/4ebdc106-7c31-405b-8c35-10a217eaaa95" />

    - $\(s\)$: Sample standard deviation
    - $\(df = n - 1\)$
- **Independent t-test**:
  - Compares means of two independent groups.
  - Test statistic:
   <img width="244" height="119" alt="image" src="https://github.com/user-attachments/assets/173129de-9380-40b2-8402-5810b1eb6560" />

    - Assumes equal variances or uses Welch’s t-test if variances differ.
- **Paired t-test**:
  - Compares means of paired data (e.g., before and after).
  - Test statistic:
    <img width="192" height="115" alt="image" src="https://github.com/user-attachments/assets/1fc47a98-dce5-4723-9560-44301cc2ba26" />

    - $\(\bar{d}\)$: Mean of differences
    - $\(s_d\)$: Standard deviation of differences

#### **c. ANOVA (Analysis of Variance)**
- Tests differences in means across three or more groups.
- **One-way ANOVA**: Tests one factor.
  - Test statistic: $\(F = \frac{\text{Between-group variance}}{\text{Within-group variance}}\)$
- **Two-way ANOVA**: Tests two factors and their interaction.

#### **d. Chi-Square Test**
- **Goodness-of-fit test**: Tests if observed frequencies match expected frequencies.
  - Test statistic:
    <img width="244" height="89" alt="image" src="https://github.com/user-attachments/assets/7c93c8c3-e6b6-4d87-a923-4456bccf6df9" />

    - $\(O_i\)$: Observed frequencies
    - $\(E_i\)$: Expected frequencies
- **Test of independence**: Tests if two categorical variables are independent.

#### **e. Non-parametric Tests**
- Used when data violates normality or other assumptions.
  - **Mann-Whitney U test**: Compares two independent groups.
  - **Wilcoxon signed-rank test**: Compares paired data.
  - **Kruskal-Wallis test**: Compares three or more groups.

---

### **5. Mathematical Implementation**

#### **Example 1: One-Sample T-Test**
**Scenario**: A company claims its employees complete a task in 30 minutes on average. A sample of 25 employees has a mean of 32 minutes and a standard deviation of 5 minutes. Test if the population mean differs from 30 minutes at $\(\alpha = 0.05\)$.

- **Step 1: State hypotheses**
  - $\(H_0: \mu = 30\)$ (mean task time is 30 minutes).
  - $\(H_1: \mu \neq 30\)$ (mean task time is not 30 minutes, two-tailed).

- **Step 2: Choose significance level**
  - $\(\alpha = 0.05\).$

- **Step 3: Select test**
  - Use a one-sample t-test $(unknown variance, \(n = 25\)).$

- **Step 4: Calculate test statistic**
 <img width="676" height="132" alt="image" src="https://github.com/user-attachments/assets/15880563-df1c-4733-bcb5-6e4e985d6af1" />

- **Step 5: Determine p-value or critical value**
  - For a two-tailed test, critical value $\(t_{0.025, 24} \approx 2.064\).$
  - P-value: Using a t-table or software, $\(p \approx 0.051\) (two-tailed, \(t = 2, df = 24\)).$

- **Step 6: Make a decision**
  - Since $\(p = 0.051 > 0.05\)$, fail to reject $\(H_0\).$
  - Alternatively, $\(t = 2 < 2.064\)$, so the test statistic is not in the critical region.

- **Step 7: Interpret**
  - There is insufficient evidence at $\(\alpha = 0.05\)$ to conclude that the mean task time differs from 30 minutes.

#### **Example 2: Z-Test for Proportion**
**Scenario**: A poll claims 60% of voters support a candidate. A sample of 200 voters finds 110 (55%) support the candidate. Test if the population proportion differs from 60% at \(\alpha = 0.05\).

- **Step 1: State hypotheses**
  - $\(H_0: p = 0.6\)$
  - $\(H_1: p \neq 0.6\) (two-tailed).$

- **Step 2: Choose significance level**
  - $\(\alpha = 0.05\).$

- **Step 3: Select test**
  - Use a z-test for proportion $(\(n = 200\), large sample).$

- **Step 4: Calculate test statistic**
 <img width="754" height="176" alt="image" src="https://github.com/user-attachments/assets/53a236b6-92cb-4ad2-86e1-0ab05bddc08f" />


- **Step 5: Determine p-value**
  - For a two-tailed test, $\(p = 2 \cdot P(Z < -1.445) \approx 2 \cdot 0.074 = 0.148\)$ (using z-tables).

- **Step 6: Make a decision**
  - Since $\(p = 0.148 > 0.05\)$, fail to reject $\(H_0\).$

- **Step 7: Interpret**
  - There is insufficient evidence to conclude that the proportion of voters supporting the candidate differs from 60%.

---

### **6. Assumptions**
- **Random Sampling**: Ensures the sample is representative.
- **Normality**:
  - For z-tests: Large sample size $(\(n \geq 30\))$ or normal population.
  - For t-tests: Normal population or large sample size.
  - For proportions: $\(n p_0 \geq 5\)$ and $\(n (1 - p_0) \geq 5\).$
- **Independence**: Observations are independent.
- **Variance**: Equal variances for independent t-tests (unless using Welch’s t-test).

---

### **7. Limitations**
- **Misinterpretation of P-value**: A small p-value indicates evidence against $\(H_0\)$, not the probability that $\(H_0\)$ is true.
- **Type I and II Errors**: No test eliminates both risks entirely.
- **Multiple Testing**: Conducting multiple tests increases the Type I error rate (corrected using Bonferroni or other methods).
- **Practical vs. Statistical Significance**: A statistically significant result may not be practically meaningful.

---

### **8. Applications**
- **Medical Research**: Testing if a drug reduces symptoms compared to a placebo.
- **Business**: Comparing sales performance across regions.
- **Social Sciences**: Testing relationships between variables (e.g., education and income).
- **Quality Control**: Testing if a process meets specifications.

---

### **9. Tools**
- **Software**: R (e.g., `t.test()`, `prop.test()`), Python (SciPy, Statsmodels), SPSS, Excel.
- **Example Python Code (One-Sample T-Test)**:
```python
from scipy import stats
import numpy as np

# Sample data
sample = np.random.normal(loc=32, scale=5, size=25)  # Simulated data
mu_0 = 30
alpha = 0.05

# Perform t-test
t_stat, p_value = stats.ttest_1samp(sample, mu_0)

print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
if p_value <= alpha:
    print("Reject H0")
else:
    print("Fail to reject H0")
```

---

In summary, hypothesis testing is a structured approach to making data-driven decisions about population parameters. It involves formulating hypotheses, calculating test statistics, and interpreting results while controlling errors.
