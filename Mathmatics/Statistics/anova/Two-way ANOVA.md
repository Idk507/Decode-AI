Two-Way Analysis of Variance (ANOVA) is a statistical method used to examine the effects of two independent variables (factors) on a dependent variable, including whether the factors interact with each other. It extends One-Way ANOVA by allowing analysis of two categorical independent variables simultaneously, making it ideal for studying main effects and interaction effects. Below is a detailed explanation of Two-Way ANOVA, including its mechanics, assumptions, calculations, interpretation, and a practical example.

---

### **1. Purpose and Context**
Two-Way ANOVA tests whether:
- The **main effects** of each independent variable significantly affect the dependent variable.
- There is an **interaction effect** between the two independent variables, meaning the effect of one factor depends on the level of the other.

**Example**: You’re studying the effect of teaching method (Lecture, Online, Hybrid) and class size (Small, Large) on student exam scores. The dependent variable is the exam score, and the independent variables are teaching method (3 levels) and class size (2 levels).

- **Hypotheses**:
  - **Main Effect of Factor A (e.g., Teaching Method)**:
    - H₀: All teaching method means are equal.
    - H₁: At least one teaching method mean differs.
  - **Main Effect of Factor B (e.g., Class Size)**:
    - H₀: All class size means are equal.
    - H₁: At least one class size mean differs.
  - **Interaction Effect**:
    - H₀: There is no interaction between teaching method and class size.
    - H₁: There is an interaction (the effect of teaching method depends on class size, or vice versa).

---

### **2. Key Concepts**
Two-Way ANOVA partitions the total variance into four components:
- **Variance due to Factor A**: Effect of the first independent variable (e.g., teaching method).
- **Variance due to Factor B**: Effect of the second independent variable (e.g., class size).
- **Variance due to Interaction (A × B)**: Effect of the combination of both factors.
- **Within-group variance (Error)**: Random variability within groups.

The **F-statistic** is calculated for each effect (Factor A, Factor B, Interaction) by comparing the variance explained by each to the error variance:
<img width="381" height="111" alt="image" src="https://github.com/user-attachments/assets/4b46a467-0889-4aeb-a5e4-67f38348bc1a" />

A large F-value suggests a significant effect.

---

### **3. Types of Two-Way ANOVA**
1. **Without Replication**: Each combination of Factor A and Factor B has one observation. Tests only main effects (no interaction).
2. **With Replication**: Multiple observations per combination, allowing testing of main effects and interaction. This is more common.
3. **Balanced vs. Unbalanced Design**:
   - Balanced: Equal number of observations per cell (combination of factor levels).
   - Unbalanced: Unequal observations, requiring adjustments in calculations.

---

### **4. Assumptions**
Two-Way ANOVA assumes:
1. **Normality**: The dependent variable is approximately normally distributed within each group (combination of factor levels). Check with histograms, Q-Q plots, or Shapiro-Wilk test.
2. **Homogeneity of Variances**: Variances are equal across groups (test with Levene’s or Bartlett’s test).
3. **Independence**: Observations are independent within and across groups (ensured by study design, e.g., random assignment).
4. **Interval or Ratio Data**: The dependent variable is continuous.

If assumptions are violated:
- Use transformations (e.g., log) for non-normality or unequal variances.
- Consider non-parametric alternatives (e.g., aligned rank transform) or robust ANOVA methods.

---

### **5. Mathematical Framework**
Two-Way ANOVA involves calculating **sum of squares (SS)**, **degrees of freedom (df)**, **mean squares (MS)**, and **F-statistics** for each effect.

#### **Notation**
<img width="801" height="426" alt="image" src="https://github.com/user-attachments/assets/51e751d0-cc1a-417a-ac25-e4ffecba0866" />


#### **Sum of Squares**
1. **Total Sum of Squares (SST)**:
 <img width="684" height="158" alt="image" src="https://github.com/user-attachments/assets/808357f0-3371-48c0-aa74-51f78ec873ae" />

2. **Sum of Squares for Factor A (SSA)**:
  <img width="667" height="151" alt="image" src="https://github.com/user-attachments/assets/6417accc-47e0-4676-9a88-03de10dc2aac" />


3. **Sum of Squares for Factor B (SSB)**:
 <img width="685" height="155" alt="image" src="https://github.com/user-attachments/assets/04fd96b6-23ec-4c6b-8a6f-2fc37fd31f65" />


4. **Sum of Squares for Interaction (SSAB)**:
<img width="724" height="140" alt="image" src="https://github.com/user-attachments/assets/d2b196bd-3022-4073-8967-d46ffa3bc92b" />


5. **Sum of Squares Error (SSE)**:
  <img width="722" height="154" alt="image" src="https://github.com/user-attachments/assets/b525bf38-e4ce-429b-848f-000badd11a07" />

6. **Degrees of Freedom**:
   <img width="418" height="235" alt="image" src="https://github.com/user-attachments/assets/5e05567f-f61f-49c0-9c99-88cc5a00b5aa" />

7. **Mean Squares**:
 <img width="235" height="199" alt="image" src="https://github.com/user-attachments/assets/756edf7d-fc84-4caa-98a3-5dd024720a91" />

8. **F-Statistics**:
  <img width="351" height="155" alt="image" src="https://github.com/user-attachments/assets/0abbda1f-2203-4cad-b31d-86f4c494d5a3" />


9. **P-Values**: Compare each F-statistic to the F-distribution with appropriate degrees of freedom to determine significance (typically at \( \alpha = 0.05 \)).

---

### **6. ANOVA Table**
Results are summarized in an ANOVA table:
<img width="949" height="332" alt="image" src="https://github.com/user-attachments/assets/39535dde-cefb-4cc5-89d9-b0dd2c84138e" />


---

### **7. Interaction Effect**
The interaction effect is critical in Two-Way ANOVA. If significant, it means the effect of one factor depends on the level of the other. For example:
- If the effect of teaching method on scores varies by class size (e.g., Online works better in small classes but not large ones), there’s an interaction.
- Visualize interactions with an **interaction plot**, where lines representing one factor’s levels across the other factor’s levels are non-parallel (indicating interaction).

---

### **8. Post-Hoc Tests**
If main effects or interactions are significant:
- **For Main Effects**: Use Tukey’s HSD, Bonferroni, or Scheffé to compare levels of a factor (e.g., pairwise comparisons of teaching methods).
- **For Interactions**: Analyze simple effects (e.g., effect of teaching method at each class size) or use post-hoc tests for cell means.

---

### **9. Example: Step-by-Step**
#### **Scenario**
You’re testing the effects of teaching method (A: Lecture, B: Online, C: Hybrid) and class size (Small, Large) on exam scores (out of 100). You have 3 observations per cell (balanced design). Data:

| Teaching Method | Class Size | Scores                     |
|-----------------|------------|----------------------------|
| Lecture         | Small      | 70, 72, 68                |
| Lecture         | Large      | 65, 63, 67 Gemini | 60, 62, 64                |
| Online          | Small      | 85, 88, 83                |
| Online          | Large      | 78, 80, 76                |
| Hybrid          | Small      | 75, 77, 73                |
| Hybrid          | Large      | 70, 72, 68                |

#### **Step 1: Hypotheses**
- Factor A (Teaching Method): H₀: μ_Lecture = μ_Online = μ_Hybrid; H₁: At least one differs.
- Factor B (Class Size): H₀: μ_Small = μ_Large; H₁: Means differ.
- Interaction: H₀: No interaction; H₁: Interaction exists.

#### **Step 2: Calculate Means**
- **Teaching Method Means**:
  <img width="403" height="140" alt="image" src="https://github.com/user-attachments/assets/59f8f3b9-6389-48ff-b74c-a58a73e0399d" />

- **Class Size Means**:
 <img width="478" height="91" alt="image" src="https://github.com/user-attachments/assets/3df12304-be80-4b0f-98dd-1249237eacc0" />

- **Cell Means** $(e.g., Lecture-Small = \( \frac{70+72+68}{3} = 70 \), etc.)$.
- **Grand Mean**: $\( \frac{70+72+\dots+68}{18} = 73.39 \)$

#### **Step 3: Sum of Squares**
<img width="989" height="232" alt="image" src="https://github.com/user-attachments/assets/46ffdacf-c71f-4852-a706-66e8e614173f" />


#### **Step 4: Degrees of Freedom**
<img width="445" height="239" alt="image" src="https://github.com/user-attachments/assets/76997336-927d-4f8b-9382-e00491bd1645" />

#### **Step 5: Mean Squares**
<img width="312" height="187" alt="image" src="https://github.com/user-attachments/assets/145e53fa-f2fb-448f-b401-a4c017df95ad" />


#### **Step 6: F-Statistics**
<img width="464" height="140" alt="image" src="https://github.com/user-attachments/assets/4cc7adc5-0b07-4553-94d3-ccc870b48a9e" />

#### **Step 7: P-Values**
Using F-distribution tables or software:
<img width="696" height="148" alt="image" src="https://github.com/user-attachments/assets/b94b4267-497d-4936-88f5-f653479f9d1e" />

#### **Step 8: ANOVA Table**
| Source         | SS       | df  | MS      | F      | P-Value  |
|----------------|----------|-----|---------|--------|----------|
| Teaching Method| 824.78   | 2   | 412.39  | 41.28  | <0.0001  |
| Class Size     | 183.39   | 1   | 183.39  | 18.36  | 0.0011   |
| Interaction    | 24.78    | 2   | 12.39   | 1.24   | 0.323    |
| Error          | 119.83   | 12  | 9.99    |        |          |
| Total          | 1152.78  | 17  |         |        |          |

#### **Step 9: Conclusion**
- **Teaching Method**: Significant (p < 0.0001). At least one method differs.
- **Class Size**: Significant (p = 0.0011). Small classes differ from large.
- **Interaction**: Not significant (p = 0.323). No evidence that the effect of teaching method depends on class size.

#### **Step 10: Post-Hoc Tests**
Since teaching method is significant, use Tukey’s HSD to compare:
- Online (81.67) vs. Lecture (66): Significant.
- Online vs. Hybrid (72.5): Significant.
- Hybrid vs. Lecture: Not significant.

#### **Chart: Interaction Plot**
To visualize main effects and check for interaction, here’s an interaction plot showing mean scores by teaching method across class sizes:

<img width="1023" height="546" alt="image" src="https://github.com/user-attachments/assets/7f126844-cd28-4629-b3b0-4e772987f48f" />


The near-parallel lines suggest no strong interaction, consistent with the non-significant interaction p-value.

---

### **10. Interpretation**
- **Teaching Method**: Online (mean = 81.67) yields higher scores than Hybrid (72.5) and Lecture (66).
- **Class Size**: Small classes (mean = 76.78) yield higher scores than large classes (70).
- **Interaction**: No significant interaction, so the effect of teaching method is consistent across class sizes.
- **Effect Size**: Eta-squared for teaching method:$ \( \eta^2 = \frac{824.78}{1152.78} \approx 0.715 \)$ (71.5% of variance explained). For class size: $\( \eta^2 = \frac{183.39}{1152.78} \approx 0.159 \)$ (15.9%).

---

### **11. Practical Considerations**
- **Software**: Use Python (`statsmodels`), R (`aov()`), or SPSS for calculations.
- **Assumption Checks**:
  - Normality: Shapiro-Wilk or Q-Q plots.
  - Homogeneity: Levene’s test.
- **If Assumptions Fail**: Use Welch’s ANOVA or non-parametric methods.
- **Balanced Design**: This example is balanced; unbalanced designs require adjustments (e.g., Type II or III SS in R).

---

### **12. Limitations**
- Requires post-hoc tests to identify specific group differences.
- Sensitive to assumption violations, especially with unbalanced designs.
- Interaction effects can be complex to interpret without visualization.

---

### **13. Additional Requests**
If you’d like:
- Python/R code to run this analysis.
- A deeper dive into post-hoc tests or simple effects.
- A different dataset or visualization (e.g., bar chart).
- Help with your own data.

