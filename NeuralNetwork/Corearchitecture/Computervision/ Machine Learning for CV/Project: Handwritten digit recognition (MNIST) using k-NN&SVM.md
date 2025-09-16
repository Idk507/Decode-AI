### Project: Handwritten Digit Recognition on MNIST Using k-NN and SVM

The MNIST dataset is a benchmark for handwritten digit recognition in machine learning and computer vision. It consists of 70,000 grayscale images (28x28 pixels) of digits 0-9: 60,000 for training and 10,000 for testing. Each image is labeled with its digit. The goal is to classify these images using classical machine learning algorithms like k-Nearest Neighbors (k-NN) and Support Vector Machine (SVM). These are non-parametric (k-NN) and parametric (SVM) classifiers that work well on flattened pixel features (no need for advanced feature extraction like SIFT/HOG, as raw pixels suffice for MNIST due to its simplicity).

This project involves:
- **Data Loading and Preprocessing**: Normalize pixels, flatten images into 784D vectors.
- **Model Training**: Fit k-NN or SVM on training data.
- **Prediction and Evaluation**: Use metrics like accuracy, precision, recall, F1-score, and confusion matrix (since it's multi-class classification; mAP is more for detection, but we can compute per-class precision/recall).

k-NN achieves ~95-97% accuracy on MNIST, while SVM can reach ~98-99% with proper tuning. Below, I'll explain each algorithm end-to-end: concept, process, math, and Python code using NumPy/SciPy (for k-NN) and scikit-learn (for SVM, assuming it's available; if not, a basic SVM can be implemented via quadratic programming, but sklearn is standard). Code assumes MNIST is loaded (e.g., via `keras.datasets.mnist` or downloaded CSV; in practice, use `from sklearn.datasets import fetch_openml; mnist = fetch_openml('mnist_784')`).

#### Dataset Overview and Preprocessing
- **MNIST Details**: Images are 28x28 (784 pixels), values 0-255. Labels: 0-9.
- **Preprocessing**:
  1. Load data.
  2. Normalize: Divide pixels by 255 (to 0-1 range).
  3. Flatten: Reshape to (N, 784) vectors.
  4. Split: Train/test (already split in MNIST).
- **Why?** Reduces scale sensitivity in distance-based (k-NN) or margin-based (SVM) algorithms.

---

### 1. k-Nearest Neighbors (k-NN)

#### Concept
k-NN is a lazy, instance-based learner: It stores all training data and classifies a new sample by finding the k closest training examples (neighbors) based on a distance metric (e.g., Euclidean) and taking a majority vote. It's simple, non-parametric, and works well for low-dimensional data like flattened MNIST, but slow for large datasets (O(N) query time).

#### End-to-End Process
1. **Training**: Just store training data (X_train: features, y_train: labels).
2. **Prediction**:
   - For a test sample x, compute distances to all training samples.
   - Select k smallest distances (nearest neighbors).
   - Vote: Assign the most common label among neighbors.
3. **Hyperparameters**: k (e.g., 3-5), distance metric (Euclidean).
4. **Evaluation**: Compute accuracy, etc., on test set.
5. **Improvements**: Use KD-trees for faster search (but basic implementation here).

#### Math
<img width="981" height="182" alt="image" src="https://github.com/user-attachments/assets/c34b940e-c6e2-4186-ac97-69d77b0a04fc" />


#### Code Example
Basic implementation using NumPy (no sklearn for k-NN). Assumes MNIST loaded as X_train (60000x784), y_train (60000,), X_test (10000x784), y_test (10000,).

```python
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for x in X_test:
        distances = [euclidean_distance(x, x_train) for x_train in X_train]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [y_train[i] for i in nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)

# Example usage (assume data loaded and normalized)
# y_pred = knn_predict(X_train, y_train, X_test, k=5)
# accuracy = np.mean(y_pred == y_test)
# print(f"Accuracy: {accuracy * 100:.2f}%")
```

For full project: Load data, normalize, train (store), predict, evaluate. This is slow (~hours on CPU for full MNIST); use subsets or sklearn's KNeighborsClassifier for speed.

---

### 2. Support Vector Machine (SVM)

#### Concept
SVM is a supervised classifier that finds a hyperplane maximizing the margin between classes. For non-linear data (like MNIST digits), use kernels (e.g., RBF) to map to higher dimensions. It's effective for high-dimensional data and robust to overfitting with regularization. For multi-class, use one-vs-one or one-vs-rest (sklearn defaults to one-vs-rest).

#### End-to-End Process
1. **Training**:
   - Flatten and normalize images.
   - Fit SVM on X_train, y_train with kernel (linear for speed, RBF for accuracy).
2. **Prediction**: Project test samples onto the hyperplane and classify.
3. **Hyperparameters**: C (regularization, e.g., 1.0), kernel ('rbf'), gamma (for RBF, e.g., 'scale').
4. **Evaluation**: Same as k-NN.
5. **Tuning**: Grid search for C, gamma (cross-validation on train set).

#### Math
<img width="781" height="288" alt="image" src="https://github.com/user-attachments/assets/3540d146-989a-4d29-ba28-b1e17bc91e2b" />


#### Code Example
Using scikit-learn (assumes installed; if not, use quadratic programming via cvxopt for basic linear SVM). Load MNIST as above.

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Assume X_train, y_train, X_test, y_test loaded and normalized (X = X / 255.0)

# Train SVM (RBF kernel for non-linearity)
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Per-class precision, recall, F1 (macro average for multi-class)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
print(f"Macro Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# Confusion Matrix (10x10 for digits)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
```

For loading MNIST in code (if using sklearn):
```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```

---

### Full Project Workflow and Evaluation
1. **Load and Preprocess**: As above.
2. **Train and Test**:
   - For k-NN: Store data, predict (slow; subsample for demo, e.g., 10k train).
   - For SVM: Fit and predict (faster with linear kernel, but RBF better).
3. **Evaluation Metrics** (Multi-Class Adaptation):
   - **Accuracy**: $\( \frac{\text{Correct Predictions}}{\text{Total Samples}} \)$.
   - **Precision/Recall/F1 per Class**: For class c, treat as binary (one-vs-rest): Precision_c = TP_c / (TP_c + FP_c), etc. Macro-average: Mean over classes.
   - **Confusion Matrix**: 10x10 table showing predicted vs. true labels (diagonal = correct).
   - **Other**: Micro-average (global TP/FP), ROC-AUC (but for multi-class, one-vs-rest).
   - Typical Results: k-NN (k=5) ~96.9%, SVM (RBF) ~98.5% accuracy.
4. **Improvements**:
   - Dimensionality reduction: PCA to reduce 784D (e.g., to 50D) for speed.
   - Cross-Validation: Tune k or C using 5-fold on train set.
   - Visualization: Plot misclassified images using matplotlib.

#### Comparison Table
| Algorithm | Strengths | Weaknesses | Accuracy on MNIST | Time Complexity |
|-----------|-----------|------------|-------------------|-----------------|
| k-NN     | Simple, no training time | Slow prediction, memory-intensive | ~95-97% | Predict: O(N*D) |
| SVM      | High accuracy, robust margins | Training can be slow (O(N^2) for kernel), parameter-sensitive | ~98-99% | Train: O(N^2 - N^3) |

This project demonstrates classical ML for CV classification. For production, consider CNNs (e.g., via PyTorch) for >99% accuracy. If implementing, test on a subset to avoid long runtimes.
