**Canonical Correlation Analysis (CCA)** is a multivariate statistical technique used to explore and quantify the relationships between two sets of variables. It identifies linear combinations of variables from each set that are maximally correlated with each other. Below is a concise explanation of CCA, focusing on understanding the relationship between two sets of variables:

### Key Concepts
- **Objective**: CCA finds pairs of linear combinations (canonical variates) from two sets of variables, **X** (p variables) and **Y** (q variables), such that the correlation between the pairs is maximized.
- **Model**: For two variable sets **X** = (X₁, X₂, ..., Xₚ)ᵀ and **Y** = (Y₁, Y₂, ..., Yₖ)ᵀ, CCA seeks canonical variates:
  - **Uᵢ** = **aᵢᵀX** (linear combination of **X** variables).
  - **Vᵢ** = **bᵢᵀY** (linear combination of **Y** variables).
  - Such that the correlation between **Uᵢ** and **Vᵢ**, called the *canonical correlation* (ρᵢ), is maximized, subject to constraints (e.g., orthogonality of subsequent pairs).
- **Canonical Correlations**: The correlations ρ₁, ρ₂, ..., ρₘ (where m = min(p, q)) are ordered such that ρ₁ ≥ ρ₂ ≥ ... ≥ ρₘ ≥ 0.
- **Canonical Variates**: The pairs (U₁, V₁), (U₂, V₂), ..., (Uₘ, Vₘ) are the canonical variate pairs, with **aᵢ** and **bᵢ** being the canonical coefficients.

### Mathematical Framework
- **Covariance Matrix**: Consider the covariance matrix of the combined variables:
 <img width="265" height="105" alt="image" src="https://github.com/user-attachments/assets/3b2f1feb-2b0a-41a2-a7b5-1f83b8647697" />

  where:
  - **Σₓₓ**: Covariance matrix of **X**.
  - **Σᵧᵧ**: Covariance matrix of **Y**.
  - **Σₓᵧ**: Cross-covariance matrix between **X** and **Y**.
- **Optimization**: CCA solves for **aᵢ** and **bᵢ** to maximize:
  <img width="338" height="96" alt="image" src="https://github.com/user-attachments/assets/7c0ed9a4-b567-49c4-9803-c72c4be1062e" />

  subject to constraints that subsequent canonical variates are uncorrelated with previous ones.
- **Solution**: The canonical correlations and coefficients are found via eigenvalue decomposition of a matrix derived from **Σₓₓ⁻¹ΣₓᵧΣᵧᵧ⁻¹Σᵧₓ** or singular value decomposition (SVD) of the correlation matrix.

### Steps in CCA
1. **Standardize the Data**: Center and scale **X** and **Y** variables to ensure comparability (optional but common).
2. **Compute Covariance Matrices**: Estimate **Σₓₓ**, **Σᵧᵧ**, and **Σₓᵧ** from the data.
3. **Solve for Canonical Variates**:
   - Compute the canonical correlations (ρᵢ) and coefficients (**aᵢ**, **bᵢ**) using eigenvalue or SVD methods.
   - The number of canonical pairs is min(p, q).
4. **Interpret Results**:
   - **Canonical Correlations**: Assess the strength of relationships (ρ₁, ρ₂, ...).
   - **Canonical Loadings**: Correlations between original variables and their canonical variates to understand contributions.
   - **Canonical Scores**: Compute **Uᵢ** and **Vᵢ** for each observation for further analysis or visualization.
5. **Test Significance**: Use tests (e.g., Wilks’ Lambda) to evaluate whether canonical correlations are statistically significant.

### Properties
- **Maximizing Correlation**: The first canonical pair (U₁, V₁) has the highest possible correlation, the second pair (U₂, V₂) the next highest, and so on, subject to orthogonality constraints.
- **Orthogonality**: Canonical variates within each set are uncorrelated (e.g., Cov(Uᵢ, Uⱼ) = 0 for i ≠ j).
- **Interpretation**: High canonical correlations indicate strong linear relationships between the two sets of variables.

### Applications
- **Social Sciences**: Relating psychological test scores (e.g., cognitive skills) to behavioral measures (e.g., performance metrics).
- **Finance**: Analyzing relationships between economic indicators (e.g., interest rates, inflation) and asset returns.
- **Bioinformatics**: Connecting genetic markers to phenotypic traits.
- **Machine Learning**: Used in multi-view learning or as a precursor to other analyses.

### Example
Suppose **X** contains 3 variables (e.g., math, reading, writing scores) and **Y** contains 2 variables (e.g., job performance, satisfaction). CCA might find:
- First canonical pair: U₁ (a combination of academic scores) highly correlated with V₁ (a combination of job metrics), with ρ₁ = 0.85.
- Canonical loadings show that math and reading strongly contribute to U₁, while job performance drives V₁.

### Limitations
- **Linearity**: CCA assumes linear relationships between variable sets.
- **Sample Size**: Requires sufficient observations to estimate covariances reliably (typically n > p + q).
- **Interpretation**: Canonical variates may be hard to interpret if many variables contribute.


