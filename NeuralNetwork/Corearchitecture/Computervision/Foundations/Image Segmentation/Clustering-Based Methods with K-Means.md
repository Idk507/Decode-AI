### Image Segmentation: Clustering-Based Methods with K-Means

**Image segmentation** divides an image into meaningful regions based on pixel characteristics like intensity, color, or texture. **Clustering-based methods**, such as **K-means clustering**, group pixels into clusters based on feature similarity, effectively segmenting the image into regions with similar properties. K-means is a popular unsupervised clustering algorithm that partitions pixels into \( K \) clusters by minimizing the variance within each cluster. This explanation covers the mathematical concepts, practical applications, and a detailed Python code implementation using OpenCV and NumPy to demonstrate K-means clustering for image segmentation.

---

### 1. What is Clustering-Based Segmentation?

Clustering-based segmentation groups pixels into clusters based on features (e.g., color, intensity) using unsupervised learning algorithms. K-means clustering is widely used due to its simplicity and effectiveness for image segmentation.

- **Definition**: K-means clustering assigns each pixel to one of $\( K \)$ clusters by minimizing the sum of squared distances between pixels and their assigned cluster centroids in the feature space.
- **Purpose**: Segments an image into regions with similar visual properties, such as color or intensity, for object detection, analysis, or visualization.
- **Applications**:
  - Color-based segmentation (e.g., separating objects in natural images).
  - Medical imaging (e.g., identifying tissue types in MRI scans).
  - Image compression (e.g., reducing colors to $\( K \)$ representative colors).
  - Object recognition and scene understanding.

#### K-Means Clustering
K-means clustering iteratively assigns pixels to clusters and updates cluster centroids to minimize intra-cluster variance. It operates on a feature vector for each pixel, typically RGB colors or intensity values.

---

### 2. Mathematical Concepts of K-Means Clustering

K-means clustering partitions $\( N \)$ pixels into $\( K \)$ clusters by minimizing the within-cluster sum of squares (WCSS), also known as the objective function.

#### Mathematical Formulation
1. **Feature Space**:
<img width="753" height="179" alt="image" src="https://github.com/user-attachments/assets/5acb3f96-9aae-4cd2-9a86-7d9ef810de73" />


2. **Objective Function**:
<img width="656" height="330" alt="image" src="https://github.com/user-attachments/assets/1e739cb5-4106-4087-b3d5-c00acaa5ba9e" />


3. **Algorithm Steps**:
  <img width="975" height="418" alt="image" src="https://github.com/user-attachments/assets/07c35659-c5b0-4bb6-8075-a6f03339c97c" />


4. **Segmentation**:
   - Assign each pixel the color or intensity of its cluster’s centroid to create the segmented image.
   - Alternatively, assign unique colors to each cluster for visualization.

#### Example
Consider a 3x3 RGB image (simplified):
<img width="551" height="111" alt="image" src="https://github.com/user-attachments/assets/7b882cfc-1b8e-4a4b-acc2-58543d1ed16b" />

- **Feature Space**: Each pixel is a vector $\( [R, G, B] \)$, e.g., $\( [100, 100, 100] \)$.
- **K=2 Clustering**:
  - Initial centroids (random): $\( \boldsymbol{\mu}_1 = [100, 100, 100] \), \( \boldsymbol{\mu}_2 = [200, 200, 200] \)$.
  - Assign pixels:
    - Pixels near [100, 100, 100] (e.g., [100, 100, 100], [110, 110, 110]) to cluster 1.
    - Pixels near [200, 200, 200] (e.g., [200, 200, 200], [205, 205, 205]) to cluster 2.
  - Update centroids:
    - Cluster 1: Mean of [100, 100, 100], [110, 110, 110], [105, 105, 105], [115, 115, 115] ≈ [107.5, 107.5, 107.5].
    - Cluster 2: Mean of [200, 200, 200], [205, 205, 205], [190, 190, 190], [195, 195, 195], [210, 210, 210] ≈ [200, 200, 200].
  - Iterate until convergence.
- **Output** (assign centroid colors):
 <img width="645" height="106" alt="image" src="https://github.com/user-attachments/assets/8448d78e-cb72-47a6-bbc6-671333893ce5" />

- **Result**: Two regions with distinct colors, segmenting the image into “dark” and “bright” clusters.

---

### 3. Practical Considerations
- **Choosing $\( K \)$**:
  - $\( K \)$ (number of clusters) must be specified.
  - Too few clusters may merge distinct regions; too many may over-segment.
  - Use domain knowledge or elbow method (plot WCSS vs. $\( K \)$) to choose $\( K \)$.
- **Feature Space**:
  - RGB colors are common for color images.
  - Include spatial coordinates (e.g., $\( [x, y, R, G, B] \)$) for spatial coherence.
  - Normalize features (e.g., scale RGB to [0, 1]) to balance contributions.
- **Initialization**:
  - Random initialization may lead to different results.
  - Use K-means++ (`cv2.kmeans` with `KMEANS_PP_CENTERS`) for better initial centroids.
- **Preprocessing**:
  - Apply Gaussian blur to reduce noise:
    ```python
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ```
  - Convert to appropriate color space (e.g., LAB for perceptual uniformity).
- **Performance**:
  - K-means is computationally intensive for large images due to iterative assignments.
  - Subsample pixels or reduce image resolution for faster processing.
- **Applications**:
  - Color quantization (reduce image colors to $\( K \))$.
  - Object segmentation (e.g., separating sky, ground, objects).
  - Medical imaging (e.g., identifying tissue types).

---

### 4. Code Implementation in Python

Below is a Python code example using OpenCV and NumPy to demonstrate K-means clustering for image segmentation. The code:
- Loads a color image and converts it to a feature vector (RGB or LAB).
- Applies K-means clustering with $\( K = 3 \)$ clusters.
- Visualizes the original and segmented images, along with a color bar showing cluster centroids.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image_path = 'image.jpg'  # Replace with your image path
img = cv2.imread(image_path)

# Check if image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Optional: Convert to LAB color space for better perceptual uniformity
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Reshape image to a 2D array of pixels (feature vectors)
pixels = img_lab.reshape(-1, 3).astype(np.float32)

# Define K-means parameters
K = 3  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
attempts = 10
flags = cv2.KMEANS_PP_CENTERS

# Apply K-means clustering
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, attempts, flags)

# Convert centers to uint8 and back to original color space
centers = np.uint8(centers)
centers_rgb = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)

# Create segmented image by assigning each pixel its cluster centroid color
segmented = centers[labels.flatten()]
segmented = segmented.reshape(img.shape)

# 5. Display Results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

# Segmented Image
plt.subplot(1, 3, 2)
plt.title(f'K-Means Segmentation (K={K})')
plt.imshow(segmented)
plt.axis('off')

# Cluster Centroids (Color Bar)
plt.subplot(1, 3, 3)
plt.title('Cluster Centroids')
color_bar = np.zeros((100, 300, 3), dtype=np.uint8)
width = 300 // K
for i in range(K):
    color_bar[:, i*width:(i+1)*width] = centers_rgb[i]
plt.imshow(color_bar)
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values and labels at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original RGB: {img_rgb[y, x]}")
print(f"Segmented RGB: {segmented[y, x]}")
print(f"Cluster Label: {labels[y * img.shape[1] + x]}")
print(f"Centroid Colors (RGB): {centers_rgb}")
</xaiArtifact>

---

### Code Explanation
1. **Loading Image**:
   - Loads a color image using `cv2.imread`.
   - Converts from BGR to RGB for display and to LAB for clustering (LAB ensures perceptual uniformity).

2. **Feature Preparation**:
   - Reshapes the image into a 2D array of pixels, where each pixel is a 3D feature vector ([L, A, B]).
   - Converts to `float32` for K-means compatibility.

3. **K-Means Clustering**:
   - Applies `cv2.kmeans` with:
     - $\( K = 3 \)$ clusters.
     - Termination criteria: 100 iterations or 0.2 epsilon.
     - K-means++ initialization (`KMEANS_PP_CENTERS`) for better centroids.
   - Returns labels (cluster index for each pixel) and centroids (mean colors).

4. **Segmentation**:
   - Assigns each pixel the color of its cluster centroid.
   - Reshapes the result back to the image’s dimensions.

5. **Visualization**:
   - Displays the original image, segmented image, and a color bar showing the \( K \) centroid colors.

6. **Pixel Value Output**:
   - Prints the original and segmented RGB values, cluster label, and centroid colors at (100, 100).

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide an image file (`image.jpg`) in the same directory or update the `image_path` variable. Use a color image with distinct regions (e.g., objects, sky, ground) for clear results.

---

### Expected Output
- A figure with three subplots:
  - Original RGB image.
  - Segmented image with $\( K = 3 \)$ colors (each pixel assigned its cluster centroid color).
  - Color bar showing the $\( K \)$ centroid colors.
- Printed pixel values, e.g.:
  ```
  Pixel at (100, 100):
  Original RGB: [150 100  50]
  Segmented RGB: [145  95  45]
  Cluster Label: 1
  Centroid Colors (RGB): [[145  95  45]
                         [200 150 100]
                         [ 50  50  50]]
  ```

---

### Practical Notes
- **Choosing $\( K \)$ **:
  - Experiment with $\( K \)$ (e.g., 2–10) based on the number of distinct regions.
  - Use the elbow method (plot WCSS vs. $\( K \)$) for optimal $\( K \)$.
- **Color Space**:
  - LAB is preferred for perceptual uniformity.
  - RGB can be used but may lead to less intuitive clusters.
  - HSV or grayscale can be used for specific tasks.
- **Preprocessing**:
  - Apply Gaussian blur to reduce noise:
    ```python
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ```
  - Normalize pixel values to [0, 1] for balanced clustering.
- **Performance**:
  - Subsample pixels (e.g., every 10th pixel) for faster processing on large images.
- **Applications**:
  - Color quantization for image compression.
  - Segmenting objects (e.g., separating sky from ground).
  - Medical imaging (e.g., clustering tissue types).

---

### Mathematical Insights
- **Objective Function**: Minimizes $\( J = \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2 \)$.
- **Euclidean Distance**: Measures similarity in the feature space (e.g., RGB or LAB).
- **Convergence**: Iterates until centroids stabilize or variance reduction is minimal.

---

### Extensions
- **Spatial Features**: Include pixel coordinates $\( [x, y, R, G, B] \)$ for spatially coherent clusters.
- **Other Clustering Methods**: Explore Mean-Shift or DBSCAN for density-based segmentation.
- **Post-processing**: Apply morphological operations (e.g., opening) to smooth segmented regions.
- **Quantitative Analysis**: Evaluate segmentation quality using metrics like Silhouette Score.

