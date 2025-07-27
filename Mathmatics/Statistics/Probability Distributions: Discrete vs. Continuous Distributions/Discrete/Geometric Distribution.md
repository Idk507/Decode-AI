### Geometric Distribution: A Comprehensive Guide

The **Geometric Distribution** is a discrete probability distribution that models the number of trials required to achieve the first success in a sequence of independent Bernoulli trials, each with a constant probability of success. It is widely used in statistics and data science to analyze scenarios involving repeated attempts until a specific event occurs, such as the number of customer contacts until a sale or the number of attempts until a system failure. This explanation provides a detailed, end-to-end exploration of the geometric distribution, including its mathematical formulation, properties, use cases in data science, visualizations, and practical examples.

---

### 1. Definition and Key Characteristics

The geometric distribution describes the probability of requiring $\( k \)$ trials to achieve the first success, where each trial is a Bernoulli trial with success probability $\( p \).$ There are two common formulations, depending on whether the count starts with the first trial or the number of failures before the first success.

#### Key Assumptions
1. **Independent Trials**: Each trial is independent; the outcome of one trial does not affect others.
2. **Binary Outcomes**: Each trial has two outcomes: success (with probability $\( p \))$ or failure (with probability $\( 1-p \)).$
3. **Constant Probability**: The success probability $\( p \)$ remains constant across trials.
4. **First Success**: The distribution focuses on the trial where the first success occurs.

#### Two Variants
1. **Number of Trials Until First Success** (most common):
   - $\( X \)$ : Number of trials until (and including) the first success.
   - $\( X = 1, 2, 3, \ldots \).$
2. **Number of Failures Before First Success**:
   - $\( Y \)$ : Number of failures before the first success.
   - $\( Y = 0, 1, 2, \ldots \).$
   - $\( Y = X - 1 \).$

This explanation focuses on the first variant $(\( X \)$ : number of trials until first success), as it is more commonly used, but both will be addressed where relevant.

#### Notation
- $\( X \sim \text{Geometric}(p) \): \( X \)$  is the number of trials until the first success, with success probability $\( p \).$
- $\( 0 < p \leq 1 \)$ : Probability of success in each trial.

---

### 2. Mathematical Formulation

The geometric distribution is defined by its **Probability Mass Function (PMF)**, which gives the probability of achieving the first success on the $\( k \)$ -th trial.

#### Probability Mass Function (PMF)
For $\( X \):$  Number of trials until the first success:
<img width="466" height="60" alt="image" src="https://github.com/user-attachments/assets/8dff7cdb-b4bf-441b-9df9-d6a4ce585e4b" />

Where:
<img width="706" height="147" alt="image" src="https://github.com/user-attachments/assets/f69638b5-cd86-4d69-8064-07ba1e58830c" />

For $\( Y \):$ Number of failures before the first success:
<img width="522" height="68" alt="image" src="https://github.com/user-attachments/assets/eb151a6a-7651-462b-a30c-574a1fd7fa88" />


#### Cumulative Distribution Function (CDF)
The CDF gives the probability of achieving the first success within the first \( k \) trials:
<img width="507" height="58" alt="image" src="https://github.com/user-attachments/assets/7cccb2d6-1779-42c9-8c38-d630192834a3" />

This is derived from summing the PMF:
<img width="855" height="95" alt="image" src="https://github.com/user-attachments/assets/6a05e7fc-fdc3-4217-a2f0-f9f401755524" />

#### Parameters
- **Mean (Expected Value)**:
  For $\( X \):$
  <img width="235" height="84" alt="image" src="https://github.com/user-attachments/assets/606a7860-f9e7-4eaf-8849-b280a4a17a06" />

  The expected number of trials until the first success.
  For $\( Y \):$
 <img width="191" height="86" alt="image" src="https://github.com/user-attachments/assets/4f264f02-0038-4e15-a4b6-700adf56fe47" />

- **Variance**:
  For $\( X \):$
 <img width="282" height="94" alt="image" src="https://github.com/user-attachments/assets/139d2aad-9da5-43bd-8265-1c4b6e493ef3" />

  For $\( Y \):$
 <img width="248" height="99" alt="image" src="https://github.com/user-attachments/assets/751c4f55-48cb-442b-85d3-fc795cda3ba5" />

  (Same as for $\( X \),$ as the variance depends on failures).
- **Standard Deviation**:
 <img width="199" height="109" alt="image" src="https://github.com/user-attachments/assets/5f40d7c1-4c5e-41eb-8863-c339a3b832d9" />


#### Example Calculation
Suppose a salesperson has a 20% chance $(\( p = 0.2 \))$ of making a sale per customer contact. What is the probability of making the first sale on the 3rd contact?
<img width="707" height="67" alt="image" src="https://github.com/user-attachments/assets/7afcf08c-c901-4828-817c-0866836b7203" />

There’s a 12.8% chance the first sale occurs on the 3rd contact.

What is the probability of making the first sale within 5 contacts?
<img width="707" height="66" alt="image" src="https://github.com/user-attachments/assets/95dcf2f6-f265-44ee-b1d8-e1338a90478d" />

There’s a 67.2% chance of at least one sale in 5 contacts.

---

### 3. Properties of the Geometric Distribution

1. **Memoryless Property**:
   - The geometric distribution is the only discrete distribution with the memoryless property: the probability of success on the next trial does not depend on previous failures.
   - Mathematically:
    <img width="490" height="68" alt="image" src="https://github.com/user-attachments/assets/e87862a3-ac5f-45d0-9e6a-04342ac3ae76" />

   - Example: If no sale is made in the first 10 contacts, the probability of a sale on the 11th contact is still $\( p \).$

2. **Right-Skewed Shape**:
   - The distribution is always right-skewed, with higher probabilities for smaller $\( k \).$
   - As $\( p \)$ increases, the skew decreases, and probabilities concentrate around smaller $\( k \).$

3. **Range**:
   -<img width="216" height="81" alt="image" src="https://github.com/user-attachments/assets/9026a9cc-3775-4717-a00a-ba438812d625" />

4. **No Upper Bound**:
   - The number of trials until the first success is theoretically unbounded, though probabilities decrease exponentially for large $\( k \).$

#### Example: Expected Value and Variance
<img width="764" height="435" alt="image" src="https://github.com/user-attachments/assets/b86f51ec-b37a-45a2-91e9-53a306266022" />


---

### 4. Use Cases in Data Science

The geometric distribution is valuable for modeling scenarios involving repeated trials until a specific event occurs. Below are key applications:

1. **Customer Conversion Modeling**:
   - **Scenario**: Predict the number of website visitors until the first purchase.
   - **Application**: Model each visitor as a Bernoulli trial with success probability $\( p \)$. Use the geometric distribution to estimate the expected number of visitors or the probability of a purchase within a certain number of visits.
   - **Example**: If the purchase probability is 5% $(\( p = 0.05 \))$, calculate the expected number of visitors until a purchase $(\( \frac{1}{0.05} = 20 \)).$

2. **Reliability and Failure Analysis**:
   - **Scenario**: Determine the number of tests until a system fails.
   - **Application**: Model each test as a Bernoulli trial with a small failure probability. Use the geometric distribution to estimate maintenance schedules or reliability.
   - **Example**: If a machine fails with probability $\( p = 0.01 \)$, the expected number of tests until failure is $\( \frac{1}{0.01} = 100 \).$

3. **Marketing Campaigns**:
   - **Scenario**: Analyze the number of email attempts until a customer responds.
   - **Application**: Model each email as a Bernoulli trial. Calculate probabilities to optimize campaign strategies.
   - **Example**: If the response probability is 10% $(\( p = 0.1 \))$, calculate the probability of a response within 5 emails.

4. **Natural Language Processing (NLP)**:
   - **Scenario**: Model the number of words scanned in a document until a specific keyword appears.
   - **Application**: Treat each word as a trial with a probability of being the keyword. Use the geometric distribution to estimate search efficiency.
   - **Example**: If a keyword appears with probability $\( p = 0.02 \)$, calculate the expected number of words scanned $(\( \frac{1}{0.02} = 50 \)).$
   - 
5. **Quality Control**:
   - **Scenario**: Test items until the first defective item is found.
   - **Application**: Model inspections as Bernoulli trials with a defect probability. Use the geometric distribution to assess testing effort.
   - **Example**: If the defect rate is 2% $(\( p = 0.02 \)),$ calculate the probability of finding a defect within 10 tests.

---

### 5. Visualization

The geometric distribution is visualized as a bar chart, where each bar represents $\( P(X = k) \) for \( k = 1, 2, 3, \ldots \).$ The probabilities decrease geometrically, reflecting the right-skewed nature.

#### Example Visualization
For a geometric distribution with $\( p = 0.2 \):$
<img width="454" height="262" alt="image" src="https://github.com/user-attachments/assets/018505c5-802d-415d-a1dc-9178f6589fb3" />

<img width="970" height="539" alt="image" src="https://github.com/user-attachments/assets/6e18affc-942a-4b0a-b6b6-0260cae400a3" />



This chart shows a right-skewed distribution with decreasing probabilities as $\( k \)$ increases.

---

### 6. Practical Example in Data Science

**Scenario**: A telemarketer has a 10% chance $(\( p = 0.1 \))$ of closing a sale per call. The company wants to predict the number of calls needed to make the first sale.

1. **Expected Number of Calls**:
 <img width="262" height="111" alt="image" src="https://github.com/user-attachments/assets/1b743445-e51d-407d-888b-f09872f97369" />

   On average, it takes 10 calls to make a sale.

2. **Variance and Standard Deviation**:
  <img width="342" height="135" alt="image" src="https://github.com/user-attachments/assets/74f1eccc-6276-4064-8393-ac2369cca977" />

   The number of calls varies significantly around the mean.

3. **Probability of First Sale on the 5th Call**:
 <img width="700" height="55" alt="image" src="https://github.com/user-attachments/assets/cfdf55b7-2bbb-453f-991c-f0af11c4812b" />

   There’s a 6.56% chance the first sale occurs on the 5th call.

4. **Probability of First Sale Within 10 Calls**:
   <img width="589" height="72" alt="image" src="https://github.com/user-attachments/assets/c8662003-d23d-436e-ab46-0cc444549a74" />

   There’s a 65.13% chance of making a sale within 10 calls.

5. **Actionable Insights**:
   - Expect around 10 calls for a sale, but prepare for variability (up to ~19 calls within one standard deviation).
   - Use the probability of success within 10 calls to set realistic sales targets.
   - Visualize the distribution to communicate expectations to the sales team.

**Python Implementation**:
```python
from scipy.stats import geom

p = 0.1
# PMF: P(X = 5)
print(geom.pmf(5, p))  # ~0.06561
# CDF: P(X <= 10)
print(geom.cdf(10, p))  # ~0.6513
# Mean and variance
print(geom.mean(p))    # 10
print(geom.var(p))     # 90
```

---

### 7. Relationships to Other Distributions

1. **Bernoulli Distribution**:
   - Each trial in the geometric distribution is a Bernoulli trial with success probability $\( p \).$
   - The geometric distribution models the sequence of Bernoulli trials until the first success.

2. **Negative Binomial Distribution**:
   - The geometric distribution is a special case of the negative binomial distribution, where the number of successes required is 1.
   - For $\( r \) successes, use \( \text{Negative Binomial}(r, p) \).$

3. **Exponential Distribution**:
   - The continuous analog of the geometric distribution is the exponential distribution, which models the time until the first event in a Poisson process.

4. **Poisson Approximation**:
   - In rare cases, for very small $\( p \),$ the geometric distribution can be approximated by a Poisson distribution when considering rare successes over many trials.

---

### 8. Limitations and Considerations

1. **Assumption Violations**:
   - If trials are not independent (e.g., customer responses depend on previous interactions), the geometric model may fail.
   - If $\( p \)$ varies across trials, consider a model like the beta-geometric distribution.

2. **Right-Skewed Nature**:
   - For small $\( p \),$ the distribution is highly skewed, with long tails, making large $\( k \)$ values possible but unlikely.

3. **Discrete Nature**:
   - The geometric distribution applies to discrete trials, not continuous time. For continuous time until an event, use the exponential distribution.

4. **Memoryless Property**:
   - While useful, the memoryless property assumes no learning or fatigue effects, which may not hold in real-world scenarios (e.g., a salesperson improving with experience).

---

### Conclusion

The geometric distribution is a powerful tool for modeling the number of trials until the first success in a sequence of independent Bernoulli trials. Its mathematical simplicity (PMF, mean, variance) and memoryless property make it ideal for scenarios like customer conversion, reliability analysis, and marketing. Visualizations and software tools (e.g., Python, R) enable data scientists to compute probabilities, estimate expectations, and communicate insights effectively. By understanding its properties and limitations, practitioners can apply the geometric distribution to analyze repeated-trial scenarios and inform decision-making.

