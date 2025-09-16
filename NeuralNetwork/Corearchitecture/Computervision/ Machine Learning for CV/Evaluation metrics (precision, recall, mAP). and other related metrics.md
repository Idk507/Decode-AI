### Evaluation Metrics for Object Detection in Computer Vision

In object detection tasks, evaluation metrics quantify how well a model performs in identifying and localizing objects in images or videos. Precision, recall, and mean Average Precision (mAP) are among the most widely used metrics, especially in classical and modern computer vision (e.g., for evaluating Haar Cascades, Template Matching, or deep learning models like YOLO). Other related metrics, such as F1-score, Intersection over Union (IoU), and Average Precision (AP), are also critical. Below, I’ll explain these metrics end-to-end: their concepts, detailed calculations, underlying math, and provide Python code examples to compute them. Since object detection involves both classification (is it an object?) and localization (where is it?), these metrics assess both aspects.

---

### 1. Precision and Recall

#### Concept
- **Precision**: Measures the accuracy of positive predictions, i.e., the fraction of detected objects that are correct. It answers: "Of all detected objects, how many are actually objects?"
- **Recall**: Measures the fraction of actual objects correctly detected. It answers: "Of all true objects, how many did we find?"

These metrics are derived from a confusion matrix for binary classification (positive = object, negative = non-object) but adapted for detection by considering bounding box overlap (via IoU).

#### End-to-End Process
1. **Ground Truth and Predictions**: Collect ground truth (GT) bounding boxes (with class labels) and predicted bounding boxes (with confidence scores and labels).
2. **IoU Threshold**: Define a threshold (e.g., IoU ≥ 0.5) to determine if a predicted box matches a GT box (True Positive, TP). IoU is the overlap between predicted and GT boxes divided by their union.
3. **Confusion Matrix**:
   - **True Positive (TP)**: Predicted box has IoU ≥ threshold with a GT box of the same class.
   - **False Positive (FP)**: Predicted box has no matching GT box or wrong class.
   - **False Negative (FN)**: GT box not matched by any predicted box.
   - **True Negative (TN)**: Often ignored in detection, as non-object regions are vast.
4. **Compute Metrics**:
   - Precision = $\( \frac{TP}{TP + FP} \)$
   - Recall = $\( \frac{TP}{TP + FN} \)$
5. **Vary Confidence**: Adjust the confidence threshold to plot a Precision-Recall curve, as higher thresholds increase precision but may reduce recall.

#### Math
<img width="1009" height="343" alt="image" src="https://github.com/user-attachments/assets/f5f68bca-1eb2-44a1-bfd4-1be275a852b3" />


#### Code Example
Compute precision and recall for a set of predicted and GT bounding boxes.

```python
import numpy as np

def compute_iou(boxA, boxB):
    # box format: [x_min, y_min, x_max, y_max]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = inter_area / (boxA_area + boxB_area - inter_area + 1e-6)
    return iou

def precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    TP, FP, FN = 0, 0, len(gt_boxes)
    matched_gt = set()
    
    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= iou_threshold and best_gt_idx != -1:
            TP += 1
            FN -= 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1
            
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    return precision, recall

# Example: [x_min, y_min, x_max, y_max]
gt_boxes = [[50, 50, 100, 100], [150, 150, 200, 200]]
pred_boxes = [[48, 48, 98, 98], [140, 140, 190, 190], [300, 300, 350, 350]]  # Last is FP

precision, recall = precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
```

---

### 2. Mean Average Precision (mAP)

#### Concept
mAP is the standard metric for object detection, summarizing the Precision-Recall curve across classes and IoU thresholds. It computes the **Average Precision (AP)** for each class (area under the Precision-Recall curve) and averages it across all classes (mAP). It’s widely used in benchmarks like COCO and PASCAL VOC Arnold.

#### End-to-End Process
1. **Collect Predictions**: For each image, gather predicted boxes, confidence scores, and class labels, along with GT boxes.
2. **Rank Predictions**: Sort predictions by confidence score (descending).
3. **Compute Precision-Recall Points**:
   - For each confidence threshold, compute TP, FP, FN based on IoU ≥ threshold (e.g., 0.5).
   - Generate precision and recall at each threshold to form the PR curve.
4. **Average Precision (AP)**: Compute area under the interpolated PR curve for each class.
   - **Interpolation**: For each recall level $\( r \)$, take max precision for all higher recalls: $\( p_{\text{interp}}(r) = \max_{r' \geq r} p(r') \)$.
   - AP = sum of interpolated precision at recall levels (e.g., 0.0 to 1.0 in steps of 0.1).
5. **mAP**: Average AP across all classes. For COCO, mAP is often averaged over multiple IoU thresholds (0.5:0.05:0.95).
6. **Output**: mAP score (0 to 1, higher is better).

#### Math
<img width="1000" height="464" alt="image" src="https://github.com/user-attachments/assets/4d123d74-c5be-4335-b706-1c72aa8af811" />


#### Code Example
Compute AP for a single class (extend to mAP by averaging over classes).

```python
import numpy as np
from sklearn.metrics import auc

def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    # Sort predictions by confidence (descending)
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    pred_scores = [pred_scores[i] for i in sorted_indices]
    
    TP, FP, FN = 0, 0, len(gt_boxes)
    matched_gt = set()
    precisions, recalls = [], []
    
    for i, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_threshold and best_gt_idx != -1:
            TP += 1
            FN -= 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1
        precisions.append(TP / (TP + FP + 1e-6))
        recalls.append(TP / (TP + FN + 1e-6))
    
    # Interpolate precision
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    interpolated_precisions = [max(precisions[i:]) for i in range(len(precisions))]
    recall_points = np.linspace(0, 1, 11)
    interp_prec = [max(interpolated_precisions[recalls >= r], default=0) for r in recall_points]
    ap = np.mean(interp_prec)
    return ap

# Example
gt_boxes = [[50, 50, 100, 100], [150, 150, 200, 200]]
pred_boxes = [[48, 48, 98, 98], [140, 140, 190, 190], [300, 300, 350, 350]]
pred_scores = [0.9, 0.8, 0.7]

ap = compute_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
print(f"Average Precision (AP): {ap:.3f}")
```

For mAP, repeat for each class and average.

---

### 3. Other Related Metrics

#### F1-Score
- **Concept**: Harmonic mean of precision and recall, balancing both.
- **Formula**: $\( F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \)$
- **Use**: Single metric for balanced performance. Computed at a fixed confidence threshold or IoU.
- **Code Addition**:
  ```python
  f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
  print(f"F1-Score: {f1:.3f}")
  ```

#### Intersection over Union (IoU)
- **Concept**: Measures bounding box overlap accuracy (used in TP/FP decisions).
- **Formula**: See above.
- **Use**: Core component for matching predictions to GT. Common thresholds: 0.5 (PASCAL), 0.5:0.95 (COCO).
- **Code**: Already in `compute_iou` above.

#### Average Recall (AR)
- **Concept**: Recall averaged over IoU thresholds (COCO-specific).
- **Formula**: $\( AR = \frac{1}{10} \sum_{\text{IoU}=0.5}^{0.95} \text{Recall}_{\text{IoU}} \)$.
- **Use**: Complements mAP, focuses on localization quality.

#### Precision at k (P@k)
- **Concept**: Precision for top-k predictions (useful in ranking tasks).
- **Formula**: $\( P@k = \frac{\text{Number of TPs in top } k}{k} \)$.
- **Use**: Evaluates ranking quality (e.g., in image retrieval).

#### False Positive Rate (FPR), False Negative Rate (FNR)
- **FPR**: $\( \frac{FP}{FP + TN} \) (rarely used, as TN is large)$.
- **FNR**: $\( \frac{FN}{TP + FN} = 1 - \text{Recall} \)$.
- **Use**: Assess specific error types.

---

### Notes for Classical Methods
- **Haar Cascades**: High FP rate due to sliding window; precision/recall depend on cascade stages and scale factor. mAP less common, as it’s typically single-class (e.g., faces).
- **Template Matching**: Precision/recall vary by similarity metric (e.g., NCC vs. SSD). No training, so no AP/mAP unless confidence scores are defined. Sensitive to scale/rotation, lowering recall.
- **Challenges**: Classical methods struggle with occlusion, lighting, or deformation, reducing recall. mAP is lower than deep learning (e.g., YOLO achieves 50-90% mAP on COCO, vs. ~20-40% for Haar).

These metrics are computed similarly in classical and modern CV, but deep learning models dominate due to better generalization. For further details, check COCO or PASCAL VOC evaluation protocols.
