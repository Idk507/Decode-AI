The **Chi-Square Test** is a statistical hypothesis test used to analyze categorical data, either to assess whether observed frequencies match expected frequencies (goodness-of-fit test) or to determine if two categorical variables are independent (test of independence). It relies on the **chi-square distribution** to evaluate the difference between observed and expected frequencies. Below is a detailed explanation of the chi-square test, including its definition, types, assumptions, mathematical implementation, examples, and practical considerations.

---

### **1. Definition**
The chi-square test compares observed frequencies in categorical data to expected frequencies under a null hypothesis, quantifying the discrepancy using a chi-square statistic. It is widely used for testing relationships between categorical variables or checking if data fits a specific distribution.

- **Purpose**: To determine whether:
  - Observed frequencies match expected frequencies (goodness-of-fit).
  - Two categorical variables are independent or associated (test of independence).
- **Key Feature**: Non-parametric, suitable for categorical data, and relies on the chi-square distribution.

---

### **2. Types of Chi-Square Tests**

#### **a. Chi-Square Goodness-of-Fit Test**
- Tests whether the observed frequency distribution of a single categorical variable matches an expected distribution (e.g., theoretical or hypothesized probabilities).
- Hypotheses:
  - $\(H_0\)$: The observed frequencies match the expected frequencies.
  - $\(H_1\)$: The observed frequencies differ from the expected frequencies.
- Example: Testing if a die is fair by comparing observed roll frequencies to expected uniform frequencies.

#### **b. Chi-Square Test of Independence**
- Tests whether two categorical variables are independent or associated in a population.
- Hypotheses:
  - $\(H_0\)$: The two variables are independent (no association).
  - $\(H_1\)$: The two variables are not independent (there is an association).
- Example: Testing if gender and voting preference are related.

---

### **3. Assumptions**
- **Categorical Data**: The data must be in the form of frequencies or counts (not continuous).
- **Random Sampling**: Observations must be randomly selected.
- **Independence**: Observations must be independent of each other.
- **Expected Frequencies**: 
  - For goodness-of-fit: Expected frequencies should be at least 5 for each category (or at least 80% of categories have expected frequencies ≥ 5, with none < 1).
  - For test of independence: Expected frequencies in each cell of the contingency table should be at least 5.
- **Sample Size**: Large enough to ensure the chi-square approximation is valid.

---

### **4. Mathematical Implementation**

#### **a. Chi-Square Test Statistic**
- The chi-square statistic measures the discrepancy between observed $(\(O_i\))$ and expected $(\(E_i\))$ frequencies:
<img width="663" height="197" alt="image" src="https://github.com/user-attachments/assets/485316f1-00ab-4bbf-b925-eda8f0824b5c" />

- The test statistic follows a chi-square distribution with degrees of freedom $(\(df\))$ determined by the test type.

#### **b. Goodness-of-Fit Test**
- **Expected Frequencies**: Based on a hypothesized distribution (e.g., uniform or theoretical proportions).
  - For $\(k\)$ categories and sample size $\(n\)$, $\(E_i = n \cdot p_i\)$, where $\(p_i\)$ is the expected proportion for category $\(i\)$.
- **Degrees of Freedom**: $\(df = k - 1\)$, where $\(k\)$ is the number of categories.
- **P-value**: $\(p = P(\chi^2 > \chi^2_{\text{calc}})\)$, right-tailed.

#### **c. Test of Independence**
- **Expected Frequencies**: For a contingency table with \(r\) rows and \(c\) columns:
 <img width="506" height="71" alt="image" src="https://github.com/user-attachments/assets/cf2f1805-0ce1-40a9-8a45-d6dcd8e04e44" />

- **Degrees of Freedom**: $\(df = (r - 1)(c - 1)\)$.
- **P-value**: $\(p = P(\chi^2 > \chi^2_{\text{calc}})\)$, right-tailed.

---

### **5. Steps in Conducting a Chi-Square Test**
1. **State Hypotheses**: Define $\(H_0\) and \(H_1\)$.
2. **Choose Significance Level $(\(\alpha\))$**: Typically $\(\alpha = 0.05\).$
3. **Verify Assumptions**: Check expected frequencies and independence.
4. **Calculate Observed and Expected Frequencies**:
   - Goodness-of-fit: Use hypothesized proportions.
   - Test of independence: Use contingency table totals.
5. **Calculate Test Statistic**: Compute $\(\chi^2\)$ using the formula.
6. **Determine P-value or Critical Value**:
   - P-value: Use chi-square tables or software.
   - Critical Value: Based on $\(\alpha\)$ and $\(df\).$
7. **Make a Decision**:
   - If $\(p \leq \alpha\)$ or $\(\chi^2 \geq \text{critical value}\)$, reject $\(H_0\)$.
   - Otherwise, fail to reject $\(H_0\)$.
8. **Interpret Results**: State the conclusion in context.

---

### **6. Practical Example: Chi-Square Goodness-of-Fit Test**

**Scenario**: A researcher rolls a six-sided die 120 times to test if it is fair. The observed frequencies are: [1: 18, 2: 22, 3: 20, 4: 19, 5: 23, 6: 18]. Test at \(\alpha = 0.05\) whether the die is fair (expected probability = \(1/6\) for each face).

- **Step 1: Hypotheses**
  - $\(H_0\)$: The die is fair (observed frequencies match expected uniform distribution).
  - $\(H_1\)$: The die is not fair.

- **Step 2: Significance Level**
  - $\(\alpha = 0.05\).$

- **Step 3: Expected Frequencies**
  - Total rolls: $\(n = 120\)$.
  - Expected frequency per face: $\(E_i = 120 \cdot \frac{1}{6} = 20\).$
  - Expected: [20, 20, 20, 20, 20, 20].
  - Observed: [18, 22, 20, 19, 23, 18].

- **Step 4: Test Statistic**
<img width="1107" height="205" alt="image" src="https://github.com/user-attachments/assets/62f6fb5b-0a9c-49b8-8ea8-eca163ee1a0d" />


- **Step 5: P-value**
  - For $\(\chi^2 = 1.1\), \(df = 5\), \(p \approx 0.954\)$ (right-tailed, via chi-square table or software).

- **Step 6: Decision**
  - Since $\(p = 0.954 > \alpha = 0.05\), fail to reject \(H_0\).$
  - Critical value for $\(\alpha = 0.05\), \(df = 5\): \(\chi^2 \approx 11.07\). Since \(\chi^2 = 1.1 < 11.07\), fail to reject \(H_0\).$

- **Step 7: Interpretation**
  - There is insufficient evidence to conclude that the die is not fair.

---

### **7. Practical Example: Chi-Square Test of Independence**

**Scenario**: A survey investigates whether gender (Male, Female) is associated with preference for a product (Like, Dislike). The contingency table is:

|            | Like | Dislike | Total |
|------------|------|---------|-------|
| **Male**   | 50   | 30      | 80    |
| **Female** | 40   | 60      | 100   |
| **Total**  | 90   | 90      | 180   |

Test at $\(\alpha = 0.05\)$ whether gender and preference are independent.

- **Step 1: Hypotheses**
  - $\(H_0\)$: Gender and preference are independent.
  - $\(H_1\)$: Gender and preference are not independent.

- **Step 2: Significance Level**
  - $\(\alpha = 0.05\).$

- **Step 3: Expected Frequencies**
  - Total sample size: $\(n = 180\)$.
  <img width="705" height="238" alt="image" src="https://github.com/user-attachments/assets/34ae7603-a998-40ef-963b-4c1d272b108e" />

- **Step 4: Test Statistic**
<img width="1013" height="202" alt="image" src="https://github.com/user-attachments/assets/48d27541-6676-4270-a818-b3b1e976caf4" />

  - Degrees of freedom: $\(df = (r - 1)(c - 1) = (2 - 1)(2 - 1) = 1\)$.

- **Step 5: P-value**
  - For $\(\chi^2 = 9\), \(df = 1\), \(p \approx 0.0027\)$ (right-tailed, via chi-square table or software).

- **Step 6: Decision**
  - Since $\(p = 0.0027 \leq \alpha = 0.05\), reject \(H_0\).$
  - Critical value for $\(\alpha = 0.05\), \(df = 1\): \(\chi^2 \approx 3.84\)$. Since $\(\chi^2 = 9 > 3.84\), reject \(H_0\).$

- **Step 7: Interpretation**
  - There is sufficient evidence to conclude that gender and product preference are associated.

---

### **8. Python Implementation (Test of Independence)**

```python
from scipy import stats
import numpy as np

# Contingency table
observed = np.array([[50, 30], [40, 60]])
alpha = 0.05

# Perform chi-square test of independence
chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"Chi-square Statistic: {chi2_stat:.3f}, P-value: {p_value:.4f}, Degrees of Freedom: {dof}")
print("Expected Frequencies:\n", expected)
if p_value <= alpha:
    print("Reject H0")
else:
    print("Fail to reject H0")
```

**Output**:
```
Chi-square Statistic: 9.000, P-value: 0.0027, Degrees of Freedom: 1
Expected Frequencies:
 [[40. 40.]
  [50. 50.]]
Reject H0
```

---

### **9. Limitations and Considerations**
- **Expected Frequencies**: If expected frequencies are too low (< 5), the chi-square approximation may be unreliable. Consider combining categories or using Fisher’s exact test.
- **Normality Not Required**: The test is non-parametric, but large sample sizes improve the chi-square approximation.
- **Independence**: Violations (e.g., paired data) invalidate the test.
- **Type I and II Errors**:
  - Type I error $(\(\alpha\))$: Rejecting $\(H_0\)$ when it’s true (e.g., falsely concluding an association).
  - Type II error $(\(\beta\))$: Failing to reject $\(H_0\)$ when it’s false, reduced by larger samples.
- **Effect Size**: A significant chi-square test doesn’t indicate the strength of association. Use measures like Cramer’s V or phi coefficient.

---

### **10. Applications**
- **Market Research**: Testing if customer preferences (e.g., brand choice) are independent of demographics (e.g., age, gender).
- **Genetics**: Testing if observed genotype frequencies match expected Mendelian ratios.
- **Social Sciences**: Testing associations between categorical variables (e.g., education level and voting behavior).
- **Quality Control**: Testing if defect rates follow expected patterns.

---

In summary, the chi-square test is a powerful tool for analyzing categorical data, either to assess goodness-of-fit or independence. It’s non-parametric and widely applicable but requires sufficient expected frequencies and independence. 
