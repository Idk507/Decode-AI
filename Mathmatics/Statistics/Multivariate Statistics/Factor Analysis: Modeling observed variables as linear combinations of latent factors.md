**Factor Analysis (FA)** is a statistical method used to model observed variables as linear combinations of a smaller number of unobserved (latent) factors, plus error terms. It aims to explain the correlations among observed variables by identifying underlying factors that account for their shared variance. Below is a concise explanation of FA in the context of modeling observed variables:

### Key Concepts
- **Objective**: FA assumes that observed variables are influenced by a few latent factors, which are not directly measurable but explain the correlations among the variables.
- **Model**: For a set of *p* observed variables **X** = (X₁, X₂, ..., Xₚ)ᵀ, FA models each variable as:
 <img width="306" height="86" alt="image" src="https://github.com/user-attachments/assets/afddcbf8-67d9-49b1-8943-cd3a47fc2d1a" />

  where:
  - **μᵢ**: Mean of variable Xᵢ.
  - **Fⱼ**: Latent factors (j = 1, ..., m, where m < p).
  - **λᵢⱼ**: Factor loadings, representing the relationship between variable Xᵢ and factor Fⱼ.
  - **εᵢ**: Unique error term (specific to Xᵢ), capturing variance not explained by the factors.
- **Assumptions**:
  - Factors **F** are typically assumed to follow a multivariate normal distribution, often standardized: **F ~ N(0, I)**.
  - Errors **ε** are uncorrelated with each other and with the factors, with **E(εᵢ) = 0** and **Var(εᵢ) = ψᵢ** (specific variance).
  - Observed variables are often centered (mean subtracted) for simplicity.

### Key Components
1. **Factor Loadings (Λ)**:
   - The matrix **Λ** (p × m) contains the loadings λᵢⱼ, which indicate how strongly each variable is influenced by each factor.
   - High loadings suggest a variable is strongly associated with a factor.
2. **Common Variance (Communality)**:
   - The variance of Xᵢ explained by the factors: $\( h_i^2 = \sum_{j=1}^m \lambda_{ij}^2 \)$.
   - Total variance of Xᵢ = communality + specific variance (ψᵢ).
3. **Specific Variance (Ψ)**:
   - The diagonal matrix **Ψ** contains the unique variances ψᵢ for each variable.
4. **Covariance Structure**:
   - The covariance matrix of the observed variables is modeled as:
     <img width="196" height="53" alt="image" src="https://github.com/user-attachments/assets/5416db3e-18bc-4a40-b5a2-2da93386ad3a" />

     where **ΛΛᵀ** captures the covariance due to the common factors, and **Ψ** accounts for unique variances.

### Steps in Factor Analysis
1. **Standardize the Data**: Center and possibly scale the variables to ensure comparability.
2. **Estimate the Model**:
   - Use methods like **Principal Axis Factoring**, **Maximum Likelihood**, or **Principal Component Method** to estimate **Λ** and **Ψ**.
   - Choose the number of factors *m* (often based on eigenvalues > 1, scree plots, or interpretability).
3. **Factor Rotation**:
   - Rotate factors to improve interpretability (e.g., **Varimax** for orthogonal rotation, **Promax** for oblique rotation).
   - Rotation does not change the model fit but makes loadings more distinct.
4. **Interpret Factors**:
   - Examine loadings to identify which variables are associated with each factor.
   - Assign meaning to factors based on the variables with high loadings.
5. **Compute Factor Scores**:
   - Estimate values of the latent factors for each observation, often for use in further analysis.

### Differences from PCA
- **PCA** focuses on maximizing variance explained and is a dimensionality reduction technique, not assuming a specific model.
- **FA** assumes a latent factor model, aiming to explain correlations among variables via underlying factors.
- PCA components are linear combinations of all variables, while FA factors are inferred constructs.

### Applications
- **Psychology**: Identifying latent traits (e.g., intelligence, personality) from questionnaire responses.
- **Finance**: Modeling unobserved factors (e.g., market trends) driving asset returns.
- **Social Sciences**: Reducing complex survey data to underlying constructs (e.g., socioeconomic status).

### Example
Suppose you have 5 observed variables (e.g., test scores) with high correlations. FA might reveal 2 latent factors:
- Factor 1: High loadings on math-related scores (e.g., algebra, calculus).
- Factor 2: High loadings on verbal-related scores (e.g., reading, writing).
The model explains the correlations among scores as driven by these two latent abilities, plus unique variances.

### Estimation Challenges
- **Number of Factors**: Choosing *m* is subjective; scree plots or fit indices (e.g., chi-square for ML estimation) help.
- **Model Fit**: Assessed via goodness-of-fit tests or residual correlations.
- **Non-uniqueness**: Factor loadings are not unique without constraints (e.g., rotation or orthogonality).
