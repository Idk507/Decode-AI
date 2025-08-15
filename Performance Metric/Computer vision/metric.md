
---

### 1. Intersection over Union (IoU)

**Formula:**
<img width="236" height="113" alt="image" src="https://github.com/user-attachments/assets/8acf7543-8247-477d-b3fd-035907a81aa4" />
<img width="574" height="190" alt="image" src="https://github.com/user-attachments/assets/89bc0d1d-d21f-4b56-b995-72d7b09dee4e" />


**Explanation:**
IoU measures the overlap between a predicted region (e.g., bounding box or segmentation mask) and the ground truth, expressed as the ratio of their intersection to their union. It ranges from 0 (no overlap) to 1 (perfect overlap).

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate localization accuracy in object detection, instance segmentation, and semantic segmentation.
- **Where**: Common in datasets like COCO, Pascal VOC, and medical imaging for bounding box or mask evaluation.
- **Why**: IoU provides a simple, intuitive measure of spatial alignment between predicted and true regions.
- **Role**: Standard metric for assessing localization precision in detection and segmentation tasks.

**Advantages:**
- Intuitive and easy to compute.
- Scale-invariant, applicable to various object sizes.
- Widely used, enabling comparison across models.

**Limitations:**
- Does not account for classification accuracy (e.g., correct class label).
- Sensitive to small misalignments for small objects.
- Zero IoU for non-overlapping regions, even if close.

**Example Use Case:**
In object detection (e.g., COCO dataset), IoU evaluates how well a predicted bounding box for a car aligns with the ground truth box.

---

### 2. Mean Average Precision (mAP)

**Formula:**
<img width="735" height="422" alt="image" src="https://github.com/user-attachments/assets/9fb301c2-bc3e-42a9-8799-17e3f3243739" />


**Explanation:**
mAP is the mean of Average Precision (AP) across all classes, where AP measures the area under the Precision-Recall curve for a given class. It evaluates both classification and localization accuracy in object detection.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate object detection and instance segmentation models.
- **Where**: Standard in datasets like COCO, Pascal VOC, and Open Images.
- **Why**: mAP balances precision and recall, accounting for ranking quality and localization (via IoU thresholds, e.g., mAP@0.5).
- **Role**: Primary metric for comparing object detection models.

**Advantages:**
- Comprehensive, combining precision, recall, and localization.
- Robust to class imbalance when averaged across classes.
- Standard across object detection benchmarks.

**Limitations:**
- Complex to compute due to ranking and IoU thresholding.
- May mask class-specific performance differences.
- Sensitive to IoU threshold choice (e.g., mAP@0.5 vs. mAP@0.5:0.95).

**Example Use Case:**
In COCO object detection, mAP@0.5:0.95 (averaged across IoU thresholds from 0.5 to 0.95) evaluates a model’s detection performance across multiple classes.

---

### 3. Pixel Accuracy

**Formula:**
<img width="804" height="255" alt="image" src="https://github.com/user-attachments/assets/7bda9b8a-c045-4a61-9a3e-37a3dd420fac" />


**Explanation:**
Pixel Accuracy measures the proportion of correctly classified pixels in semantic segmentation tasks, treating each pixel as an independent classification.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate semantic segmentation models.
- **Where**: Common in datasets like Cityscapes, ADE20K, and medical imaging.
- **Why**: Pixel Accuracy provides a simple measure of overall classification accuracy at the pixel level.
- **Role**: Basic metric for assessing segmentation performance, often used alongside IoU.

**Advantages:**
- Simple and intuitive.
- Easy to compute for segmentation tasks.

**Limitations:**
- Sensitive to class imbalance (e.g., overemphasizes background pixels).
- Does not account for spatial structure or boundary accuracy.
- Less informative than IoU for small objects.

**Example Use Case:**
In semantic segmentation of urban scenes (e.g., Cityscapes), Pixel Accuracy measures how many pixels are correctly labeled as “road,” “car,” etc.

---

### 4. Dice Coefficient (F1-Score for Segmentation)

**Formula:**
<img width="769" height="228" alt="image" src="https://github.com/user-attachments/assets/7fe55794-8919-4565-ab90-f6610bf97a6e" />


**Explanation:**
The Dice Coefficient measures the overlap between predicted and ground truth segmentation masks, equivalent to the F1-score in binary classification. It balances precision and recall.

**When, Where, and Why It Is Used:**
- **When**: Used in semantic and instance segmentation, especially in medical imaging.
- **Where**: Common in datasets like BraTS, ISIC, and other medical imaging tasks.
- **Why**: Dice focuses on overlap, making it suitable for imbalanced datasets where foreground objects are small.
- **Role**: Standard metric for segmentation, particularly in medical applications.

**Advantages:**
- Robust to class imbalance, emphasizing foreground objects.
- Interpretable as a balance of precision and recall.
- Equivalent to F1-score, widely understood.

**Limitations:**
- Sensitive to small objects or boundary errors.
- Does not account for spatial distance between errors.

**Example Use Case:**
In tumor segmentation from MRI scans, Dice evaluates how well the predicted tumor mask overlaps with the ground truth.

---

### 5. Hausdorff Distance

**Formula:**
<img width="598" height="96" alt="image" src="https://github.com/user-attachments/assets/669d75b0-59b3-4b4f-a19e-9418dabce4e6" />

- $\( A, B \)$: Sets of points in predicted and ground truth segmentation boundaries.
- $\( d(a, b) \)$: Euclidean distance between points $\( a \)$ and $\( b \)$.
- $\( \sup \)$: Supremum (maximum), $\( \inf \)$: Infimum (minimum).

**Explanation:**
Hausdorff Distance measures the maximum distance between any point on the predicted segmentation boundary and the closest point on the ground truth boundary, capturing the worst-case boundary misalignment.

**When, Where, and Why It Is Used:**
- **When**: Used in segmentation tasks where boundary accuracy is critical.
- **Where**: Common in medical imaging (e.g., organ or tumor segmentation).
- **Why**: Hausdorff Distance focuses on the largest boundary error, useful for applications where precise boundaries matter.
- **Role**: Evaluates boundary alignment in segmentation tasks.

**Advantages:**
- Sensitive to boundary errors, capturing worst-case misalignment.
- Useful for precise boundary evaluation in medical imaging.

**Limitations:**
- Sensitive to outliers, as it focuses on the maximum distance.
- Computationally expensive for large point sets.
- Does not account for overall overlap.

**Example Use Case:**
In cardiac segmentation from CT scans, Hausdorff Distance evaluates the maximum boundary error between predicted and true heart contours.

---

### 6. PSNR (Peak Signal-to-Noise Ratio)

**Formula:**
<img width="393" height="80" alt="image" src="https://github.com/user-attachments/assets/c69f82e4-8eee-400d-b7c2-25371b6cb734" />

- $\( \text{MAX}_I \)$: Maximum possible pixel value (e.g., 255 for 8-bit images).
- $\( \text{MSE} \)$ : Mean Squared Error between predicted and ground truth images:
<img width="468" height="96" alt="image" src="https://github.com/user-attachments/assets/483fc407-3cf1-4788-97f7-22c805b8693b" />

- $\( I, \hat{I} \)$ : Ground truth and predicted images.

**Explanation:**
PSNR measures the quality of a reconstructed or generated image by comparing it to the ground truth, expressed as the ratio of the maximum signal power to the mean squared error, in decibels (dB).

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate image reconstruction, denoising, or super-resolution.
- **Where**: Common in image processing tasks like compression, denoising, and generative models.
- **Why**: PSNR quantifies pixel-level fidelity, with higher values indicating better quality.
- **Role**: Standard metric for assessing image quality in reconstruction tasks.

**Advantages:**
- Simple and widely used for pixel-level comparison.
- High values indicate low distortion.

**Limitations:**
- Does not capture perceptual quality or structural similarity.
- Sensitive to small pixel differences that may not be visually significant.

**Example Use Case:**
In image denoising, PSNR evaluates how closely a denoised image matches the original clean image.

---

### 7. SSIM (Structural Similarity Index)

**Formula:**
<img width="853" height="282" alt="image" src="https://github.com/user-attachments/assets/dbe75085-7fa2-402d-b037-2e46eb14d891" />

**Explanation:**
SSIM measures the structural similarity between two images by comparing luminance, contrast, and structure, designed to align with human visual perception.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate image quality in tasks like super-resolution, compression, or denoising.
- **Where**: Common in image processing and generative models.
- **Why**: SSIM captures perceptual similarity, making it more aligned with human judgments than PSNR.
- **Role**: Assesses image quality based on structural and perceptual fidelity.

**Advantages:**
- Correlates better with human perception than PSNR.
- Captures structural and contrast differences.

**Limitations:**
- Computationally more complex than PSNR.
- Less effective for images with significant distortions or non-natural content.

**Example Use Case:**
In super-resolution, SSIM evaluates how well a high-resolution output preserves structural details compared to the ground truth.

---

### 8. FID (Fréchet Inception Distance)

**Formula:**
<img width="818" height="201" alt="image" src="https://github.com/user-attachments/assets/801fdbb0-e52d-41d8-b9c3-e3113b296382" />


**Explanation:**
FID measures the similarity between feature distributions of real and generated images, using features extracted from a pre-trained Inception V3 model. Lower FID indicates better similarity.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate generative models (e.g., GANs, VAEs).
- **Where**: Common in image generation tasks (e.g., ImageNet, CelebA datasets).
- **Why**: FID captures both visual quality and diversity, aligning with human perception of generated images.
- **Role**: Standard metric for assessing generative model performance.

**Advantages:**
- Sensitive to both quality and diversity of generated images.
- Correlates well with human judgments.

**Limitations:**
- Dependent on the Inception V3 model, which may not generalize to all domains.
- Computationally expensive due to feature extraction.
- Assumes Gaussian feature distributions.

**Example Use Case:**
In evaluating a GAN for face generation, FID compares the distribution of generated faces to real faces in a dataset.

---

### 9. LPIPS (Learned Perceptual Image Patch Similarity)

**Formula:**
<img width="813" height="231" alt="image" src="https://github.com/user-attachments/assets/c328c0d2-9c09-450d-9ef3-1ad794d0db62" />


**Explanation:**
LPIPS measures perceptual similarity between two images using features from a pre-trained deep neural network, weighted to align with human perception.

**When, Where, and Why It Is Used:**
- **When**: Used to evaluate image quality in generative models, super-resolution, or style transfer.
- **Where**: Common in tasks requiring perceptual quality assessment.
- **Why**: LPIPS captures perceptual differences, outperforming PSNR and SSIM in aligning with human judgments.
- **Role**: Evaluates perceptual fidelity in image generation and processing tasks.

**Advantages:**
- Aligns closely with human visual perception.
- Robust to small pixel-level differences.

**Limitations:**
- Requires a pre-trained network, limiting generalizability.
- Computationally expensive.
- Dependent on the choice of network and weights.

**Example Use Case:**
In style transfer, LPIPS evaluates how well the stylized image preserves content while applying the target style.

---

### 10. Average Precision (AP)

**Formula:**
<img width="374" height="82" alt="image" src="https://github.com/user-attachments/assets/c5da94b5-c60f-4f35-ba28-01f976d40fd1" />

- $\( P_k \)$: Precision at recall $\( R_k \)$.
- $\( R_k \)$ : Recall at rank $\( k \)$.

**Explanation:**
AP is the area under the Precision-Recall curve for a single class, measuring the quality of ranked predictions in object detection or classification tasks.

**When, Where, and Why It Is Used:**
- **When**: Used in object detection and information retrieval to evaluate ranking quality.
- **Where**: Common in datasets like COCO, Pascal VOC, and ranking tasks.
- **Why**: AP balances precision and recall, providing a single metric for detection performance.
- **Role**: Core component of mAP, used to assess per-class detection performance.

**Advantages:**
- Balances precision and recall.
- Robust to ranking and threshold variations.

**Limitations:**
- Requires ranked predictions, not applicable to all tasks.
- Sensitive to IoU threshold in detection tasks.

**Example Use Case:**
In object detection, AP for the “dog” class evaluates how well the model ranks and localizes dogs in images.

---

### 11. Recall@k

**Formula:**
<img width="531" height="78" alt="image" src="https://github.com/user-attachments/assets/a960eb2f-f0e2-48e4-86a6-bf18886f7b6e" />

**Explanation:**
Recall@k measures the proportion of relevant items (e.g., correct detections or classifications) retrieved in the top \( k \) predictions.

**When, Where, and Why It Is Used:**
- **When**: Used in ranking tasks, such as image retrieval or recommendation systems.
- **Where**: Common in information retrieval and multi-label classification.
- **Why**: Recall@k evaluates the ability to retrieve relevant items within a fixed number of predictions.
- **Role**: Assesses ranking quality in top-k predictions.

**Advantages:**
- Simple and focused on retrieval performance.
- Useful for tasks with a fixed number of results.

**Limitations:**
- Does not account for precision or ranking quality within top $\( k \)$.
- Dependent on the choice of $\( k \)$.

**Example Use Case:**
In image retrieval, Recall@5 evaluates whether relevant images are retrieved within the top 5 results.

---

### 12. Top-1/Top-5 Accuracy

**Formula:**
<img width="832" height="160" alt="image" src="https://github.com/user-attachments/assets/a6acf4c5-93ee-4e4c-af86-8d85d2c50699" />


**Explanation:**
Top-1 Accuracy measures whether the model’s top prediction matches the ground truth label. Top-5 Accuracy checks if the correct label is among the top 5 predictions.

**When, Where, and Why It Is Used:**
- **When**: Used in image classification tasks.
- **Where**: Common in datasets like ImageNet, CIFAR.
- **Why**: Top-1 is strict, while Top-5 accounts for cases where multiple labels may be plausible.
- **Role**: Standard metrics for evaluating classification performance.

**Advantages:**
- Simple and interpretable.
- Top-5 is more forgiving, useful for complex datasets.

**Limitations:**
- Top-1 is overly strict for ambiguous cases.
- Top-5 may overstate performance if lower-ranked predictions are irrelevant.

**Example Use Case:**
In ImageNet classification, Top-1 and Top-5 Accuracy evaluate whether a model correctly identifies an image’s class (e.g., “cat”).

---

### 13. Boundary F1 (BF1)

**Formula:**
<img width="298" height="87" alt="image" src="https://github.com/user-attachments/assets/c793ed0f-0d7b-4517-8a22-42644a65876b" />

- $\( \text{BP} \)$ : Boundary Precision (fraction of predicted boundary pixels within a threshold distance of true boundaries).
- $\( \text{BR} \)$ : Boundary Recall (fraction of true boundary pixels within a threshold distance of predicted boundaries).

**Explanation:**
BF1 evaluates the accuracy of segmentation boundaries by measuring precision and recall of boundary pixels within a specified distance threshold.

**When, Where, and Why It Is Used:**
- **When**: Used in segmentation tasks where boundary accuracy is critical.
- **Where**: Common in medical imaging and instance segmentation.
- **Why**: BF1 focuses on boundary alignment, which is crucial for precise segmentation tasks.
- **Role**: Assesses boundary quality in segmentation models.

**Advantages:**
- Specifically evaluates boundary accuracy.
- Robust to small misalignments within the threshold.

**Limitations:**
- Requires setting a distance threshold, which is dataset-dependent.
- Does not assess overall region overlap.

**Example Use Case:**
In cell segmentation for microscopy images, BF1 evaluates how well predicted cell boundaries align with true boundaries.

---

### 14. MOTA (Multi-Object Tracking Accuracy)

**Formula:**
<img width="445" height="90" alt="image" src="https://github.com/user-attachments/assets/4484aa8f-17e6-44b9-93b6-17d82a315d46" />

- FN: False Negatives (missed detections).
- FP: False Positives (incorrect detections).
- IDS: Identity Switches (when a tracker assigns the wrong ID to an object).
- GT: Total number of ground truth objects.

**Explanation:**
MOTA measures the accuracy of multi-object tracking by accounting for missed detections, false positives, and identity switches.

**When, Where, and Why It Is Used:**
- **When**: Used in multi-object tracking tasks.
- **Where**: Common in datasets like MOTChallenge, KITTI, and autonomous driving.
- **Why**: MOTA combines detection and tracking errors, providing a comprehensive tracking evaluation.
- **Role**: Standard metric for assessing tracking performance.

**Advantages:**
- Comprehensive, capturing detection and tracking errors.
- Widely used in tracking benchmarks.

**Limitations:**
- Does not account for localization precision (e.g., IoU).
- Sensitive to identity switches, which may dominate errors.

**Example Use Case:**
In autonomous driving, MOTA evaluates how well a system tracks multiple vehicles across video frames.

---

### Summary Table

| Metric            | Focus                              | Task Type                      | Advantages                          | Limitations                         |
|-------------------|------------------------------------|--------------------------------|-------------------------------------|-------------------------------------|
| IoU               | Spatial overlap                   | Detection, segmentation        | Intuitive, scale-invariant          | Ignores classification, sensitive to small objects |
| mAP               | Precision-recall ranking          | Object detection              | Comprehensive, robust to imbalance  | Complex, sensitive to IoU threshold |
| Pixel Accuracy    | Pixel-level classification        | Semantic segmentation         | Simple, intuitive                   | Class imbalance, ignores structure  |
| Dice Coefficient  | Overlap (F1-score)                | Segmentation                  | Robust to imbalance, interpretable  | Sensitive to boundaries             |
| Hausdorff Distance| Boundary misalignment             | Segmentation                  | Captures worst-case boundary errors | Sensitive to outliers, complex      |
| PSNR              | Pixel-level fidelity              | Image reconstruction          | Simple, widely used                 | Ignores perceptual quality          |
| SSIM              | Structural similarity             | Image reconstruction          | Aligns with human perception        | Less effective for heavy distortions |
| FID               | Feature distribution similarity   | Image generation              | Captures quality and diversity      | Model-dependent, computationally heavy |
| LPIPS             | Perceptual similarity             | Image generation, processing  | Aligns with human perception        | Model-dependent, computationally heavy |
| AP                | Precision-recall for one class    | Object detection              | Balances precision and recall       | Requires ranking, threshold-sensitive |
| Recall@k          | Retrieval in top k                | Image retrieval               | Simple, focused on retrieval        | Ignores precision, k-dependent      |
| Top-1/Top-5 Acc.  | Classification correctness        | Image classification          | Simple, interpretable               | Top-1 strict, Top-5 forgiving      |
| Boundary F1 (BF1) | Boundary accuracy                | Segmentation                  | Boundary-focused, robust to small errors | Threshold-dependent, no region overlap |
| MOTA              | Tracking accuracy                 | Multi-object tracking         | Comprehensive for tracking          | Ignores localization precision      |

---

### Conclusion
Each Computer Vision metric serves a specific purpose depending on the task and evaluation needs. IoU, mAP, and AP are critical for object detection, while Pixel Accuracy, Dice, and Hausdorff Distance are key for segmentation. PSNR, SSIM, FID, and LPIPS evaluate image quality in reconstruction and generation tasks, with FID and LPIPS focusing on perceptual quality. Recall@k and Top-1/Top-5 Accuracy are suited for retrieval and classification, while BF1 and MOTA address boundary and tracking accuracy, respectively. Choosing the right metric depends on the task, data, and whether localization, classification, perceptual quality, or tracking performance is prioritized.
