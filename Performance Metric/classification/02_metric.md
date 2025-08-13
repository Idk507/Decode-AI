
---

### 1. **Accuracy**

**Formula**:
<img width="327" height="50" alt="image" src="https://github.com/user-attachments/assets/195e6ee5-6dd4-45d2-99c9-5859605e8ecf" />

Where:
- **TP**: True Positives (correctly predicted positive cases)
- **TN**: True Negatives (correctly predicted negative cases)
- **FP**: False Positives (incorrectly predicted positive cases)
- **FN**: False Negatives (incorrectly predicted negative cases)

**Explanation**:
Accuracy measures the proportion of correct predictions (both positive and negative) out of the total predictions made by the model. It’s a straightforward metric that indicates overall model performance.

**When and Where Used**:
- Used in balanced datasets where the classes (positive and negative) are roughly equally represented.
- Common in tasks like image classification or text classification when class imbalance is not a concern.
- Often reported as a first glance at model performance in introductory analyses.

**Why It’s Important**:
- Provides a quick summary of how well the model performs across all classes.
- Easy to interpret and communicate to non-technical stakeholders.

**Role and Limitations**:
- Accuracy can be misleading in imbalanced datasets. For example, in a dataset with 95% negative and 5% positive cases, a model that always predicts negative can achieve 95% accuracy but fails to detect positives.
- Does not provide insight into class-specific performance (e.g., how well the model identifies positives vs. negatives).

**Example**:
In a medical test for a disease (positive = diseased, negative = healthy), if a model correctly identifies 90 out of 100 healthy patients (TN) and 8 out of 10 diseased patients (TP), with 2 false positives (FP) and 2 false negatives (FN), accuracy is:
<img width="480" height="45" alt="image" src="https://github.com/user-attachments/assets/852c49b6-509e-4401-bf43-7917a62cbc8c" />

---

### 2. **Precision**

**Formula**:
<img width="220" height="54" alt="image" src="https://github.com/user-attachments/assets/2337649d-63bd-4de1-80af-4bce60f3cf69" />

**Explanation**:
Precision measures the proportion of true positive predictions out of all positive predictions made by the model. It answers, “Of all the instances the model labeled as positive, how many were actually positive?”

**When and Where Used**:
- Used in scenarios where false positives are costly, such as email spam detection (labeling a legitimate email as spam is undesirable) or fraud detection.
- Common in information retrieval systems (e.g., search engines) to ensure retrieved results are relevant.

**Why It’s Important**:
- High precision indicates that the model is reliable when it predicts a positive class, minimizing false alarms.
- Useful when the cost of acting on a false positive is high (e.g., unnecessary medical treatments).

**Role and Limitations**:
- Precision focuses only on positive predictions and ignores false negatives, so it doesn’t capture the model’s ability to identify all positive cases.
- Often used in conjunction with recall to balance the trade-off between false positives and false negatives.

**Example**:
In spam detection, if a model flags 50 emails as spam, with 40 being actual spam (TP) and 10 being legitimate emails (FP), precision is:
<img width="412" height="44" alt="image" src="https://github.com/user-attachments/assets/f3e9ffb4-5578-4e2f-b473-628761219c34" />

---

### 3. **Recall (Sensitivity, True Positive Rate)**

**Formula**:
<img width="191" height="49" alt="image" src="https://github.com/user-attachments/assets/b4d4a0e6-d1df-454c-ad93-38bc4f296f80" />

**Explanation**:
Recall measures the proportion of actual positive cases that the model correctly identifies. It answers, “Of all the actual positive instances, how many did the model correctly predict?”

**When and Where Used**:
- Used in scenarios where missing a positive case is costly, such as disease detection (e.g., cancer screening) or anomaly detection.
- Common in medical diagnostics, where identifying all positive cases (e.g., patients with a disease) is critical.

**Why It’s Important**:
- High recall ensures that most positive cases are captured, reducing the risk of missing critical instances.
- Crucial in safety-critical applications where false negatives have severe consequences.

**Role and Limitations**:
- Recall focuses on capturing positive cases but ignores false positives, so a model with high recall may produce many false positives.
- Often paired with precision to evaluate the trade-off between capturing positives and avoiding false positives.

**Example**:
In cancer screening, if there are 100 patients with cancer, and the model correctly identifies 90 (TP) but misses 10 (FN), recall is:
<img width="371" height="59" alt="image" src="https://github.com/user-attachments/assets/621b78f3-f6d3-4cac-af5b-74585495fd47" />

---

### 4. **Specificity (True Negative Rate)**

**Formula**:
<img width="245" height="48" alt="image" src="https://github.com/user-attachments/assets/97793ada-4448-4776-bb03-3b34260c2fe7" />

**Explanation**:
Specificity measures the proportion of actual negative cases that the model correctly identifies as negative. It answers, “Of all the actual negative instances, how many did the model correctly predict?”

**When and Where Used**:
- Used in scenarios where correctly identifying negatives is important, such as ruling out diseases in medical diagnostics or identifying non-fraudulent transactions.
- Often used alongside recall in medical and fraud detection tasks.

**Why It’s Important**:
- High specificity ensures that negative cases are correctly identified, reducing unnecessary interventions (e.g., avoiding false alarms in security systems).
- Balances recall in applications where both positive and negative predictions matter.

**Role and Limitations**:
- Specificity focuses on negative cases and ignores positive cases, so it must be used with other metrics like recall or precision.
- Less emphasized in imbalanced datasets where the positive class is the primary focus.

**Example**:
In a disease test with 200 healthy patients, if the model correctly identifies 180 as healthy (TN) but incorrectly flags 20 as diseased (FP), specificity is:
<img width="430" height="62" alt="image" src="https://github.com/user-attachments/assets/de3a4f49-0a50-478b-a9d0-ff9bea53d0eb" />

---

### 5. **F1 Score**

**Formula**:
<img width="333" height="56" alt="image" src="https://github.com/user-attachments/assets/01c570a8-94e1-4a04-8efb-60cc35bb77fc" />

**Explanation**:
The F1 Score is the harmonic mean of precision and recall, providing a single metric that balances the trade-off between the two. It is particularly useful when precision and recall are equally important.

**When and Where Used**:
- Used in imbalanced datasets where both false positives and false negatives matter, such as in information retrieval, medical diagnostics, or fraud detection.
- Common in machine learning competitions and evaluations where a single metric is desired.

**Why It’s Important**:
- Combines precision and recall into a single number, making it easier to compare models.
- Penalizes models that have a large imbalance between precision and recall.

**Role and Limitations**:
- The F1 Score assumes that precision and recall are equally important, which may not always be the case (e.g., in some medical tasks, recall is prioritized over precision).
- Does not account for true negatives, so it’s less informative in balanced datasets where specificity matters.

**Example**:
Using the precision (0.8) and recall (0.9) from earlier examples, the F1 Score is:
<img width="528" height="48" alt="image" src="https://github.com/user-attachments/assets/3b3436ab-9d95-4713-8c23-e581c142935c" />

---

### 6. **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**

**Formula**:
AUC-ROC is the area under the ROC curve, which plots the True Positive Rate (Recall) against the False Positive Rate $(\( \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} \))$ at various classification thresholds.

**Explanation**:
The ROC curve visualizes the trade-off between sensitivity (recall) and (1 - specificity) as the classification threshold changes. AUC-ROC quantifies the overall ability of the model to distinguish between positive and negative classes, with a value between 0 and 1 (0.5 = random guessing, 1 = perfect discrimination).

**When and Where Used**:
- Used in binary classification tasks to evaluate a model’s ability to rank predictions correctly, such as in credit scoring, medical diagnostics, or spam detection.
- Common in imbalanced datasets because it is insensitive to class distribution.

**Why It’s Important**:
- AUC-ROC summarizes model performance across all possible thresholds, making it robust to threshold selection.
- A high AUC indicates that the model can effectively separate positive and negative classes.

**Role and Limitations**:
- AUC-ROC is useful for ranking predictions but does not directly measure the quality of calibrated probabilities.
- Can be misleading in highly imbalanced datasets, where Precision-Recall AUC is preferred.

**Example**:
If a model’s ROC curve is plotted and the area under the curve is calculated as 0.85, it means the model has an 85% chance of correctly distinguishing a positive instance from a negative one.

---

### 7. **Log Loss (Logarithmic Loss)**

**Formula**:
For binary classification:
<img width="608" height="224" alt="image" src="https://github.com/user-attachments/assets/8b9f4bda-730e-4bcd-a146-6a5d1306f866" />

**Explanation**:
Log Loss measures the uncertainty of a model’s probability predictions, penalizing confident incorrect predictions more heavily than less confident ones. Lower log loss indicates better-calibrated probabilities.

**When and Where Used**:
- Used in tasks requiring well-calibrated probabilities, such as probabilistic forecasting, ranking tasks, or when models output probabilities (e.g., logistic regression).
- Common in machine learning competitions like Kaggle, where probability outputs are evaluated.

**Why It’s Important**:
- Encourages models to produce accurate and well-calibrated probability estimates.
- Sensitive to the confidence of predictions, unlike accuracy or F1 Score.

**Role and Limitations**:
- Log Loss is sensitive to extreme probabilities (e.g., predicting 0.999 when the true label is 0 results in a large penalty).
- Less intuitive for non-technical audiences compared to accuracy or F1 Score.

**Example**:
For a single prediction where the true label is 1 and the predicted probability is 0.9, the log loss contribution is:
<img width="674" height="57" alt="image" src="https://github.com/user-attachments/assets/9e701ac9-66ae-4042-b1fe-f7c22cfb87b2" />

---

### 8. **Cohen’s Kappa**

**Formula**:
<img width="663" height="190" alt="image" src="https://github.com/user-attachments/assets/f66d0de5-be92-4c9a-9995-95927421412a" />


**Explanation**:
Cohen’s Kappa measures the agreement between predicted and actual labels, adjusted for chance agreement. It ranges from -1 to 1, with 0 indicating no agreement beyond chance and 1 indicating perfect agreement.

**When and Where Used**:
- Used in tasks with imbalanced datasets or when chance agreement is a concern, such as in inter-rater agreement studies or text classification.
- Common in evaluating models for tasks like sentiment analysis or medical diagnosis annotation.

**Why It’s Important**:
- Accounts for the possibility of random agreement, making it more robust than accuracy in imbalanced datasets.
- Useful for comparing models across datasets with different class distributions.

**Role and Limitations**:
- Kappa penalizes models that perform well by chance, providing a more realistic measure of performance.
- Can be sensitive to class distribution and may be less interpretable in highly skewed datasets.

**Example**:
If a model’s accuracy $(\( p_o \))$ is 0.8 and the expected agreement by chance $(\( p_e \))$ is 0.5, Cohen’s Kappa is:
<img width="287" height="67" alt="image" src="https://github.com/user-attachments/assets/044565f0-6be1-40b9-bfec-3b7a12eca426" />

---

### 9. **Matthews Correlation Coefficient (MCC)**

**Formula**:
<img width="478" height="71" alt="image" src="https://github.com/user-attachments/assets/666c59ab-1b2f-4fb7-95e8-ad2b393be3fc" />

**Explanation**:
MCC is a correlation coefficient between the observed and predicted binary classifications, ranging from -1 to 1. A value of 1 indicates perfect prediction, 0 indicates random prediction, and -1 indicates total disagreement.

**When and Where Used**:
- Used in imbalanced datasets where all four outcomes (TP, TN, FP, FN) are important, such as in bioinformatics, medical diagnostics, or fraud detection.
- Preferred when a single metric is needed to summarize model performance across all classes.

**Why It’s Important**:
- MCC considers all elements of the confusion matrix, making it robust to class imbalance.
- Provides a balanced measure of performance, even when positive and negative classes are skewed.

**Role and Limitations**:
- MCC is a comprehensive metric but can be harder to interpret than accuracy or F1 Score for non-technical audiences.
- Sensitive to small sample sizes or extreme class imbalances.

**Example**:
<img width="646" height="100" alt="image" src="https://github.com/user-attachments/assets/e942f167-737d-4fb4-b4a0-9e98549c445c" />

---

### 10. **Precision-Recall AUC**

**Formula**:
Precision-Recall AUC is the area under the curve plotting precision against recall at various thresholds.

**Explanation**:
The Precision-Recall curve focuses on the trade-off between precision and recall, particularly for the positive class. The AUC summarizes the model’s performance across all thresholds, with a value of 1 indicating perfect precision and recall.

**When and Where Used**:
- Used in highly imbalanced datasets where the positive class is rare (e.g., fraud detection, rare disease diagnosis).
- Preferred over AUC-ROC when the negative class dominates, as it focuses on positive class performance.

**Why It’s Important**:
- Provides a threshold-independent measure of model performance for the positive class.
- More informative than AUC-ROC in imbalanced datasets.

**Role and Limitations**:
- Ignores true negatives, so it’s less informative about negative class performance.
- Requires probability outputs or ranking, which may not be available in all models.

**Example**:
If a model’s Precision-Recall curve is plotted and the area under the curve is 0.9, it indicates strong performance in identifying positive cases with high precision and recall.

---

### 11. **Balanced Accuracy**

**Formula**:
<img width="561" height="45" alt="image" src="https://github.com/user-attachments/assets/166c2d1d-3b62-44d2-867b-683692e2be11" />

**Explanation**:
Balanced Accuracy is the average of recall and specificity, providing a single metric that accounts for performance on both positive and negative classes. It is particularly useful in imbalanced datasets.

**When and Where Used**:
- Used in imbalanced datasets where accuracy alone is misleading, such as in fraud detection or medical diagnostics.
- Common in tasks requiring balanced performance across classes.

**Why It’s Important**:
- Balances the model’s ability to correctly identify both positive and negative cases.
- More robust than accuracy in imbalanced datasets.

**Role and Limitations**:
- Provides a single metric but does not capture the trade-off between precision and recall.
- May not be as informative as F1 Score or MCC in certain contexts.

**Example**:
Using recall (0.9) and specificity (0.9) from earlier examples:
<img width="450" height="62" alt="image" src="https://github.com/user-attachments/assets/0a5eabe7-ccc6-407a-9f69-0d55df3eaa2f" />


---

### 12. **Gini Coefficient**

**Formula**:
<img width="299" height="38" alt="image" src="https://github.com/user-attachments/assets/575733c3-0096-4b40-939e-22156515f818" />


**Explanation**:
The Gini Coefficient is derived from the AUC-ROC and measures the model’s ability to distinguish between classes. It ranges from 0 (random guessing) to 1 (perfect discrimination).

**When and Where Used**:
- Commonly used in credit scoring and financial modeling to evaluate ranking performance.
- Used in tasks where AUC-ROC is relevant but a linear scale (0 to 1) is preferred.

**Why It’s Important**:
- Provides a simple transformation of AUC-ROC, making it easier to interpret in some contexts.
- Emphasizes the model’s ability to rank predictions correctly.

**Role and Limitations**:
- Shares the same limitations as AUC-ROC, such as reduced effectiveness in highly imbalanced datasets.
- Less commonly used than AUC-ROC in machine learning.

**Example**:
If AUC-ROC is 0.85:
<img width="293" height="52" alt="image" src="https://github.com/user-attachments/assets/5a6f966e-46ae-42c4-836d-065ba9bab867" />


---

### 13. **Brier Score**

**Formula**:
<img width="350" height="216" alt="image" src="https://github.com/user-attachments/assets/e97e8bd3-a603-43d3-a3a1-8a0d8157bb11" />


**Explanation**:
The Brier Score measures the mean squared difference between predicted probabilities and actual outcomes, evaluating both calibration and discrimination. Lower scores indicate better performance.

**When and Where Used**:
- Used in tasks requiring well-calibrated probabilities, such as weather forecasting or medical risk prediction.
- Common in probabilistic models like logistic regression.

**Why It’s Important**:
- Evaluates the accuracy of probability predictions, penalizing overconfident or poorly calibrated predictions.
- Useful for comparing models based on probability outputs.

**Role and Limitations**:
- Sensitive to class imbalance, as it penalizes errors in the majority class more heavily.
- Less intuitive than accuracy or F1 Score for non-technical audiences.

**Example**:
For a prediction with true label 1 and predicted probability 0.9:
<img width="476" height="33" alt="image" src="https://github.com/user-attachments/assets/1dffa291-9d33-4c7e-96e7-613a5b768b88" />


---

### 14. **Hamming Loss**

**Formula**:
<img width="472" height="266" alt="image" src="https://github.com/user-attachments/assets/0c7b3493-6cee-492f-a245-6dd393a0f628" />


**Explanation**:
Hamming Loss measures the fraction of incorrect predictions (i.e., the proportion of labels that are misclassified). In multiclass or multilabel settings, it averages the error across all labels.

**When and Where Used**:
- Used in multiclass or multilabel classification tasks, such as text categorization or image tagging.
- Common in evaluating models with multiple outputs.

**Why It’s Important**:
- Provides a simple measure of prediction error in complex classification tasks.
- Easy to compute and interpret.

**Role and Limitations**:
- Does not account for class imbalance or the severity of errors.
- Less informative than metrics like F1 Score or MCC in binary classification.

**Example**:
In a dataset with 100 samples, if 20 are misclassified:
<img width="403" height="63" alt="image" src="https://github.com/user-attachments/assets/42da4971-971c-48ad-a1c7-11d0ac4b9140" />


---

### 15. **Jaccard Index (Intersection over Union)**

**Formula**:
<img width="323" height="54" alt="image" src="https://github.com/user-attachments/assets/e55a9e6d-df2e-415e-b968-8825b82bb21f" />

**Explanation**:
The Jaccard Index measures the similarity between the predicted and actual positive labels, defined as the size of their intersection divided by the size of their union. It ranges from 0 to 1.

**When and Where Used**:
- Used in tasks like image segmentation, text classification, or recommendation systems where overlap between predicted and actual labels is important.
- Common in evaluating clustering or multilabel classification.

**Why It’s Important**:
- Quantifies the overlap between predicted and true positive sets, making it useful for tasks with set-like outputs.
- Robust to class imbalance when focusing on the positive class.

**Role and Limitations**:
- Ignores true negatives, so it’s less informative in balanced datasets.
- Similar to precision and recall but focuses on set overlap.

**Example**:
For TP = 40, FP = 10, FN = 10:
<img width="518" height="44" alt="image" src="https://github.com/user-attachments/assets/346a492b-b211-4897-bfc4-f1707875d8dc" />


---

### 16. **Zero-One Loss**

**Formula**:
<img width="408" height="44" alt="image" src="https://github.com/user-attachments/assets/611baefe-f7f7-498b-8136-f65d2dd1c6df" />

**Explanation**:
Zero-One Loss is identical to Hamming Loss in binary or multiclass classification, measuring the fraction of incorrect predictions. It assigns a loss of 1 for each misclassification and 0 for correct predictions.

**When and Where Used**:
- Used in any classification task to measure raw error rate, such as in image or text classification.
- Common in theoretical analyses or when simplicity is desired.

**Why It’s Important**:
- Provides a straightforward measure of classification error.
- Directly related to accuracy $(\( \text{Accuracy} = 1 - \text{Zero-One Loss} \))$.

**Role and Limitations**:
- Does not account for class imbalance or the severity of errors.
- Less informative than metrics that consider probabilities or class-specific performance.

**Example**:
If 30 out of 100 samples are misclassified:
<img width="381" height="53" alt="image" src="https://github.com/user-attachments/assets/806a0860-f031-440c-949d-471dd597d3a2" />


---

### Summary Table

| **Metric**                 | **Focus**                     | **Best for Imbalanced Data?** | **Considers TN?** | **Probability-Based?** |
|----------------------------|-------------------------------|-------------------------------|-------------------|------------------------|
| Accuracy                   | Overall correctness           | No                            | Yes               | No                     |
| Precision                  | Positive prediction accuracy  | Yes                           | No                | No                     |
| Recall                     | Positive detection rate       | Yes                           | No                | No                     |
| Specificity                | Negative detection rate       | Yes                           | Yes               | No                     |
| F1 Score                   | Precision-recall balance      | Yes                           | No                | No                     |
| AUC-ROC                    | Class discrimination          | Yes                           | Yes               | Yes                    |
| Log Loss                   | Probability calibration       | Yes                           | Yes               | Yes                    |
| Cohen’s Kappa              | Agreement beyond chance      | Yes                           | Yes               | No                     |
| MCC                        | Balanced correlation          | Yes                           | Yes               | No                     |
| Precision-Recall AUC       | Positive class performance    | Yes                           | No                | Yes                    |
| Balanced Accuracy          | Class-balanced accuracy       | Yes                           | Yes               | No                     |
| Gini Coefficient           | Ranking performance           | Yes                           | Yes               | Yes                    |
| Brier Score                | Probability accuracy          | Yes                           | Yes               | Yes                    |
| Hamming Loss               | Misclassification rate        | Yes                           | Yes               | No                     |
| Jaccard Index              | Positive label overlap        | Yes                           | No                | No                     |
| Zero-One Loss              | Misclassification rate        | No                            | Yes               | No                     |

---

### General Guidance on Choosing Metrics
- **Imbalanced Datasets**: Use F1 Score, Precision-Recall AUC, MCC, or Balanced Accuracy to focus on the minority class.
- **Probability-Based Models**: Use Log Loss or Brier Score to evaluate calibration.
- **Ranking Tasks**: Use AUC-ROC or Gini Coefficient for threshold-independent evaluation.
- **Multilabel or Multiclass**: Use Hamming Loss, Jaccard Index, or extensions of F1 Score.
- **Medical or Safety-Critical Tasks**: Prioritize recall (to minimize false negatives) or specificity (to minimize false positives), depending on the cost of errors.

