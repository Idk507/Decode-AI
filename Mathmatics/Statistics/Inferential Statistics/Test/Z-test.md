The **Z-test** is a statistical hypothesis test used to determine whether a population parameter (typically the mean or proportion) differs from a hypothesized value or whether two populations differ in terms of their means or proportions, assuming the population variance is known or the sample size is large. It relies on the standard normal distribution (Z-distribution) to calculate test statistics and p-values. Below is a detailed explanation of the Z-test, including its definition, assumptions, mathematical implementation, types, examples, and practical considerations.

---

### **1. Definition**
A Z-test is used to test hypotheses about population parameters (e.g., mean or proportion) by comparing a sample statistic to a hypothesized value or another sample statistic, assuming the data follows a normal distribution or the sample size is sufficiently large for the Central Limit Theorem (CLT) to apply. The test statistic is standardized to follow the standard normal distribution (mean = 0, standard deviation = 1).

- **Purpose**: To determine whether there is sufficient evidence to reject the null hypothesis $(\(H_0\))$ in favor of the alternative hypothesis $(\(H_1\)).$
- **Key Feature**: Assumes known population variance $(\(\sigma^2\))$ or large sample size $(\(n \geq 30\))$ for normality of the sampling distribution.

---

### **2. Types of Z-Tests**

#### **a. One-Sample Z-Test (Mean)**
- Tests whether the population mean $(\(\mu\))$ differs from a hypothesized value $(\(\mu_0\))$.
- Hypotheses:
  <img width="705" height="97" alt="image" src="https://github.com/user-attachments/assets/ed0ecb1e-588f-438d-a3f4-36400ab4a1f3" />


#### **b. One-Sample Z-Test (Proportion)**
- Tests whether the population proportion $(\(p\))$ differs from a hypothesized value $(\(p_0\)).$
- Hypotheses:
<img width="692" height="91" alt="image" src="https://github.com/user-attachments/assets/5bff1b05-14b9-45f7-ba30-b51b0155f393" />


#### **c. Two-Sample Z-Test (Means)**
- Tests whether the means of two populations $(\(\mu_1, \mu_2\))$ differ.
- Hypotheses:
 <img width="513" height="98" alt="image" src="https://github.com/user-attachments/assets/c14654b1-bd27-4913-999c-75cebbad933f" />


#### **d. Two-Sample Z-Test (Proportions)**
- Tests whether the proportions of two populations $(\(p_1, p_2\))$ differ.
- Hypotheses:
<img width="520" height="105" alt="image" src="https://github.com/user-attachments/assets/93f71768-f636-4ef5-95e7-e33474d8a1ef" />

---

### **3. Assumptions**
- **Random Sampling**: The sample(s) must be randomly selected to ensure representativeness.
- **Normality**:
  - For mean tests: The population is normally distributed, or the sample size is large $(\(n \geq 30\))$ for the CLT to ensure the sampling distribution of the mean is approximately normal.
  - For proportion tests: The sample size is large enough for the normal approximation $(\(n p_0 \geq 5\)$ and $\(n (1 - p_0) \geq 5\)$ for one-sample; similar for two-sample).
- **Known Variance**: The population variance $(\(\sigma^2\))$ is known for mean tests. If unknown, a t-test is typically used unless $\(n\)$ is large, in which case the sample variance $(\(s^2\))$ approximates $\(\sigma^2\)$.
- **Independence**: Observations within and between samples (for two-sample tests) are independent.

---

### **4. Mathematical Implementation**

#### **a. One-Sample Z-Test (Mean)**
- **Test Statistic**:
<img width="687" height="313" alt="image" src="https://github.com/user-attachments/assets/a7652dcb-569d-456e-afee-63565bb6fd05" />

- **P-value**:
  <img width="369" height="161" alt="image" src="https://github.com/user-attachments/assets/4ef8213b-4e78-49f5-ae21-a201261916e9" />

#### **b. One-Sample Z-Test (Proportion)**
- **Test Statistic**:
  <img width="607" height="298" alt="image" src="https://github.com/user-attachments/assets/50ac9cd1-a0a3-414a-9322-dde54726ac31" />


#### **c. Two-Sample Z-Test (Means)**
<img width="715" height="381" alt="image" src="https://github.com/user-attachments/assets/c4f3873c-40d6-4ab5-a132-de706e9918b2" />


#### **d. Two-Sample Z-Test (Proportions)**
<img width="838" height="332" alt="image" src="https://github.com/user-attachments/assets/4c6d8c0e-4f79-461e-80a6-606f8cb2e4c8" />

---

### **5. Steps in Conducting a Z-Test**
1. **State Hypotheses**: Define $\(H_0\) and \(H_1\) (e.g., \(H_0: \mu = \mu_0\), \(H_1: \mu \neq \mu_0\)).$
2. **Choose Significance Level $(\(\alpha\))$**: Typically $\(\alpha = 0.05\)$.
3. **Calculate Test Statistic**: Use the appropriate formula.
4. **Determine P-value or Critical Value**:
   - P-value: Use z-tables or software to find the probability.
   - Critical Value: For $\(\alpha = 0.05\)$, two-tailed: $\(z = \pm 1.96\)$; right-tailed: $\(z = 1.645\)$; left-tailed: $\(z = -1.645\)$.
5. **Make a Decision**:
   - If $\(p \leq \alpha\)$ or the test statistic falls in the critical region, reject $\(H_0\)$.
   - Otherwise, fail to reject $\(H_0\)$.
6. **Interpret Results**: State the conclusion in the context of the problem.

---

### **6. Practical Example: One-Sample Z-Test (Mean)**

**Scenario**: A factory claims its machines produce parts with an average length of 50 mm, with a known population standard deviation of 2 mm. A sample of 36 parts has a mean length of 50.5 mm. Test whether the population mean differs from 50 mm at \(\alpha = 0.05\).

<img width="787" height="402" alt="image" src="https://github.com/user-attachments/assets/1997a293-3d1f-4151-b183-879671d803a5" />
<img width="988" height="357" alt="image" src="https://github.com/user-attachments/assets/0e2111d7-4a85-412c-a5f8-a5a86866bd79" />


- **Step 6: Interpretation**
  - There is insufficient evidence at the 5% significance level to conclude that the mean length of parts differs from 50 mm.

---

### **7. Practical Example: Two-Sample Z-Test (Proportions)**

**Scenario**: Two marketing campaigns are tested. Campaign A results in 120 conversions out of 500 clicks (\(\hat{p}_1 = 0.24\)), and Campaign B results in 150 conversions out of 600 clicks (\(\hat{p}_2 = 0.25\)). Test whether the conversion rates differ at \(\alpha = 0.05\).

- **Step 1: Hypotheses**
  - $\(H_0: p_1 = p_2\)$ (conversion rates are equal).
  - $\(H_1: p_1 \neq p_2\)$ (conversion rates differ, two-tailed).

- **Step 2: Significance Level**
  - $\(\alpha = 0.05\)$.

- **Step 3: Test Statistic**
<img width="1026" height="408" alt="image" src="https://github.com/user-attachments/assets/03a04ac2-aae6-4f5e-8c54-536404d5c5cb" />


- **Step 4: P-value**
  <img width="772" height="133" alt="image" src="https://github.com/user-attachments/assets/ef760169-4ed5-41ad-964d-7083466fbc9c" />


- **Step 5: Decision**
  - Since $\(p = 0.704 > \alpha = 0.05\)$, fail to reject $\(H_0\)$.

- **Step 6: Interpretation**
  - There is insufficient evidence to conclude that the conversion rates of the two campaigns differ.

---

### **8. Python Implementation (One-Sample Z-Test)**

```python
from scipy import stats
import numpy as np

# One-sample Z-test (mean)
x_bar = 50.5  # Sample mean
mu_0 = 50     # Hypothesized mean
sigma = 2     # Population standard deviation
n = 36        # Sample size
alpha = 0.05

# Calculate z-statistic
z = (x_bar - mu_0) / (sigma / np.sqrt(n))

# Calculate p-value (two-tailed)
p_value = 2 * stats.norm.cdf(-abs(z))

print(f"Z-statistic: {z:.3f}, P-value: {p_value:.4f}")
if p_value <= alpha:
    print("Reject H0")
else:
    print("Fail to reject H0")
```

**Output**:
```
Z-statistic: 1.500, P-value: 0.1336
Fail to reject H0
```

---

### **9. Limitations and Considerations**
<img width="955" height="400" alt="image" src="https://github.com/user-attachments/assets/a402693c-e452-4dc5-8a7d-0c8e3d3941d8" />


---

### **10. Applications**
- **Quality Control**: Testing if a product’s mean dimension meets specifications.
- **Medical Research**: Testing if a treatment changes a population mean or proportion.
- **Marketing**: Comparing conversion rates between campaigns.
- **Social Sciences**: Testing differences in population proportions or means.

---

In summary, the Z-test is a powerful tool for hypothesis testing when the population variance is known or sample sizes are large. It’s widely used for testing means and proportions in one- or two-sample scenarios.
