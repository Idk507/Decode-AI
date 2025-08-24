### Edge Detection in Image Processing: Sobel and Canny

Edge detection is a critical image processing technique used to identify boundaries or edges within an image, where significant changes in intensity occur. These edges often correspond to object boundaries, making edge detection essential for applications like object recognition, image segmentation, and computer vision. This explanation covers the **Sobel** and **Canny** edge detection methods, including their mathematical concepts, practical applications, and a detailed Python code implementation using OpenCV.

---

### 1. Overview of Edge Detection

Edges are regions in an image where pixel intensity changes rapidly, typically indicating transitions between objects or regions. Edge detection algorithms aim to highlight these transitions by analyzing intensity gradients.

#### Key Concepts:
- **Gradient**: The rate of change in pixel intensity, computed using derivatives in the horizontal (\(x\)) and vertical (\(y\)) directions.
- **Edge Strength**: Measured by the magnitude of the gradient.
- **Edge Direction**: Determined by the orientation of the gradient.
- **Applications**:
  - Object detection and tracking.
  - Image segmentation.
  - Feature extraction for machine learning.

The **Sobel** and **Canny** methods are two popular edge detection techniques, each with distinct approaches and strengths.

---

### 2. Sobel Edge Detection

The Sobel operator is a simple, gradient-based method for detecting edges by computing the intensity gradient of an image using convolution with specific kernels.

#### Mathematical Concept
The Sobel operator approximates the first derivative of the image intensity function in the horizontal $(\(G_x\))$ and vertical $(\(G_y\))$ directions using two 3x3 kernels:

- **Horizontal Kernel** $(\(K_x\))$ :
 <img width="277" height="135" alt="image" src="https://github.com/user-attachments/assets/a408d19b-e60b-4662-84f3-aedce218ab06" />

- **Vertical Kernel** $(\(K_y\))$ :
  <img width="311" height="116" alt="image" src="https://github.com/user-attachments/assets/e6e2e5b0-00b2-4a7d-8c6b-d1b9033355f3" />


For an image $\(I(x, y)\)$, the gradients are computed via convolution:

<img width="390" height="60" alt="image" src="https://github.com/user-attachments/assets/7cf50675-9a16-490a-8fc5-f19117c139d6" />


Where $\(*\)$ denotes convolution.

- **Gradient Magnitude**: Combines $\(G_x\) and \(G_y\)$ to measure edge strength:
 <img width="233" height="68" alt="image" src="https://github.com/user-attachments/assets/4bf74030-b86d-48e1-a07f-613b61b6b00e" />

  Alternatively, for computational efficiency:
  <img width="232" height="59" alt="image" src="https://github.com/user-attachments/assets/7c6bf899-fcf3-4740-8cd6-7f8d2527c073" />


- **Gradient Direction**: Indicates the orientation of the edge:
 <img width="268" height="86" alt="image" src="https://github.com/user-attachments/assets/d4a1f48b-e0a5-4ea6-91b6-a2d719f909c1" />

  $\(\theta\)$ ranges from 0 to $\(2\pi\)$, often quantized into discrete directions (e.g., 0°, 45°, 90°, 135°).

#### Properties:
- **Purpose**: Detects edges by emphasizing regions with high gradient magnitude.
- **Strengths**:
  - Simple and computationally efficient.
  - Provides both edge strength and direction.
- **Weaknesses**:
  - Sensitive to noise, as it directly uses derivatives.
  - Produces thick edges, which may require thresholding for refinement.
- **Applications**: Used in tasks requiring quick edge detection, such as feature extraction or as a preprocessing step for more advanced algorithms.

#### Example:
For a 3x3 neighborhood in a grayscale image:
<img width="228" height="129" alt="image" src="https://github.com/user-attachments/assets/5128e129-8a30-460c-af27-be66b23a21b5" />


Applying the horizontal Sobel kernel \(K_x\):
<img width="1029" height="110" alt="image" src="https://github.com/user-attachments/assets/2a9dc7c0-b882-436d-9f7e-665bd7ceffd7" />


Applying the vertical Sobel kernel \(K_y\):
<img width="1034" height="111" alt="image" src="https://github.com/user-attachments/assets/bff294ad-381c-4344-8729-ef0fb897653d" />

Gradient magnitude: $\( G = \sqrt{0^2 + 0^2} = 0 \)$, indicating no edge. If the central pixel’s intensity were different, a non-zero gradient would indicate an edge.

---

### 3. Canny Edge Detection

The Canny edge detector is a more sophisticated, multi-stage algorithm designed to produce thin, well-defined edges with minimal noise. Developed by John Canny in 1986, it is considered the gold standard for edge detection.

#### Mathematical Concept and Stages
The Canny algorithm involves the following steps:

1. **Noise Reduction**:
   - Apply a Gaussian blur to reduce noise, which could otherwise cause false edge detections.
   - Kernel: A Gaussian kernel (e.g., 5x5 with $\(\sigma = 1.4\)$) smooths the image:
    <img width="200" height="56" alt="image" src="https://github.com/user-attachments/assets/b40b3082-c542-4e75-ac2a-35f3fa2d6fc1" />

     Where $\(G\)$ is the Gaussian kernel.

2. **Gradient Computation**:
   - Compute intensity gradients using Sobel kernels $(\(G_x\), \(G_y\))$ to find edge strength and direction, as described above.
   - Gradient magnitude: $\( G = \sqrt{G_x^2 + G_y^2} \)$.
   - Gradient direction: $\(\theta = \arctan(G_y / G_x)\)$, quantized to 0°, 45°, 90°, or 135°.

3. **Non-Maximum Suppression**:
   - Suppress non-maximum gradient values to thin edges.
   - For each pixel, compare its gradient magnitude with neighbors in the direction of the gradient. If it’s not the maximum, set it to zero, preserving only local maxima (likely edges).

4. **Double Thresholding**:
   - Apply two thresholds: a low threshold $(\(T_{\text{low}}\))$ and a high threshold $(\(T_{\text{high}}\))$.
   - Pixels with $\( G > T_{\text{high}} \)$ are strong edges.
   - Pixels with $\( T_{\text{low}} < G \leq T_{\text{high}} \)$ are weak edges.
   - Pixels with $\( G \leq T_{\text{low}} \)$ are suppressed.

5. **Edge Tracking by Hysteresis**:
   - Connect weak edges to strong edges if they are spatially connected (e.g., via 8-connectivity).
   - Weak edges not connected to strong edges are discarded.

#### Properties:
- **Purpose**: Produces thin, continuous edges with high accuracy.
- **Strengths**:
  - Robust to noise due to Gaussian smoothing.
  - Thin edges via non-maximum suppression.
  - Accurate edge connectivity via hysteresis.
- **Weaknesses**:
  - Computationally more complex than Sobel.
  - Requires tuning of thresholds $(\(T_{\text{low}}\), \(T_{\text{high}}\))$.
- **Applications**: Used in precise edge detection tasks, such as object boundary detection, shape analysis, and autonomous driving.

#### Example:
Consider a smoothed image with a gradient magnitude array after Sobel filtering:
<img width="213" height="122" alt="image" src="https://github.com/user-attachments/assets/7048ee6c-0e09-4068-b990-839134cc1a8f" />


- **Non-Maximum Suppression**: If the gradient direction at the center pixel (80) is horizontal, compare it with neighbors (100, 30). Since 80 < 100, it’s suppressed.
- **Double Thresholding**: Assume $\(T_{\text{low}} = 40\), \(T_{\text{high}} = 90\)$. Pixels with values 100 (strong) and 80 (weak) are kept, while others are discarded.
- **Hysteresis**: If the weak edge (80) is connected to a strong edge (100), it’s retained; otherwise, it’s discarded.

---

### 4. Comparison of Sobel and Canny
| Feature                  | Sobel                              | Canny                              |
|--------------------------|------------------------------------|------------------------------------|
| **Method**               | Gradient-based (first derivative) | Multi-stage (gradient + refinement)|
| **Noise Sensitivity**    | High (no preprocessing)           | Low (Gaussian smoothing)          |
| **Edge Quality**         | Thick edges                      | Thin, well-defined edges          |
| **Computational Complexity** | Low                             | High                              |
| **Parameters**           | Kernel size, optional threshold   | Gaussian $\(\sigma\), \(T_{\text{low}}\), \(T_{\text{high}}\)$ |
| **Use Case**             | Quick edge detection, preprocessing | Precise edge detection            |

---

### 5. Code Implementation in Python

Below is a Python code example using OpenCV to perform **Sobel** and **Canny** edge detection on a grayscale image. The code includes manual Sobel gradient computation and visualization of results.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (grayscale)
image_path = 'image.jpg'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# 1. Manual Sobel Edge Detection
def sobel_edge_detection(image):
    # Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float32)
    
    # Apply Sobel filters
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    
    # Compute gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = np.clip(grad_magnitude, 0, 255).astype(np.uint8)
    
    # Compute gradient direction (optional, for visualization)
    grad_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    
    return grad_x, grad_y, grad_magnitude, grad_direction

# 2. Apply Sobel Edge Detection
sobel_x, sobel_y, sobel_magnitude, sobel_direction = sobel_edge_detection(img)

# 3. Apply OpenCV Sobel (for comparison)
sobel_x_cv = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_cv = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude_cv = np.sqrt(sobel_x_cv**2 + sobel_y_cv**2)
sobel_magnitude_cv = np.clip(sobel_magnitude_cv, 0, 255).astype(np.uint8)

# 4. Apply Canny Edge Detection
canny_edges = cv2.Canny(img, threshold1=100, threshold2=200)

# 5. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Sobel X (Horizontal Edges)
plt.subplot(2, 3, 2)
plt.title('Sobel X (Horizontal)')
plt.imshow(np.abs(sobel_x), cmap='gray')
plt.axis('off')

# Sobel Y (Vertical Edges)
plt.subplot(2, 3, 3)
plt.title('Sobel Y (Vertical)')
plt.imshow(np.abs(sobel_y), cmap='gray')
plt.axis('off')

# Sobel Magnitude
plt.subplot(2, 3, 4)
plt.title('Sobel Magnitude')
plt.imshow(sobel_magnitude, cmap='gray')
plt.axis('off')

# OpenCV Sobel Magnitude
plt.subplot(2, 3, 5)
plt.title('OpenCV Sobel Magnitude')
plt.imshow(sobel_magnitude_cv, cmap='gray')
plt.axis('off')

# Canny Edges
plt.subplot(2, 3, 6)
plt.title('Canny Edges')
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original: {img[y, x]}")
print(f"Sobel X: {sobel_x[y, x]:.2f}")
print(f"Sobel Y: {sobel_y[y, x]:.2f}")
print(f"Sobel Magnitude: {sobel_magnitude[y, x]}")
print(f"Canny: {canny_edges[y, x]}")
```

---

### Code Explanation
1. **Manual Sobel Edge Detection**:
   - The `sobel_edge_detection` function applies 3x3 Sobel kernels to compute horizontal $(\(G_x\))$ and vertical $(\(G_y\))$ gradients using `cv2.filter2D`.
   - Gradient magnitude is calculated as $\(\sqrt{G_x^2 + G_y^2}\)$, clipped to 0–255.
   - Gradient direction is computed for illustration but not displayed.

2. **OpenCV Sobel**:
   - `cv2.Sobel` is used to compute gradients, specifying the derivative order $(1 for \(x\), 0 for \(y\))$ and kernel size (3x3).
   - The magnitude is computed similarly to the manual implementation.

3. **Canny Edge Detection**:
   - `cv2.Canny` applies the full Canny algorithm with low and high thresholds (100 and 200, adjustable).
   - It includes Gaussian smoothing, gradient computation, non-maximum suppression, double thresholding, and hysteresis.

4. **Visualization**:
   - Displays the original image, Sobel gradients $(\(G_x\), \(G_y\))$, Sobel magnitude (manual and OpenCV), and Canny edges in a 2x3 grid.
   - All images are shown in grayscale.

5. **Pixel Value Output**:
   - Prints pixel values at (100, 100) for the original image, Sobel gradients, Sobel magnitude, and Canny output to illustrate the effect of each method.

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide an image file (`image.jpg`) in the same directory or update the `image_path` variable.

---

### Expected Output
- A figure with six subplots showing:
  - Original grayscale image.
  - Sobel X (horizontal edges, bright where intensity changes horizontally).
  - Sobel Y (vertical edges, bright where intensity changes vertically).
  - Sobel magnitude (combined edges).
  - OpenCV Sobel magnitude (similar to manual).
  - Canny edges (thin, well-defined edges).
- Printed pixel values, e.g.:
  ```
  Pixel at (100, 100):
  Original: 120
  Sobel X: 50.00
  Sobel Y: -30.00
  Sobel Magnitude: 58
  Canny: 255
  ```

---

### Practical Notes
- **Sobel**:
  - Produces continuous gradient values, useful for further processing (e.g., thresholding).
  - Sensitive to noise; apply Gaussian blur beforehand for better results.
- **Canny**:
  - Thresholds $(\(T_{\text{low}}\), \(T_{\text{high}}\))$ significantly affect results. Adjust them based on the image (e.g., 50–150 for low-contrast images, 100–200 for high-contrast).
  - Automatically includes Gaussian smoothing $(default \(\sigma\))$.
- **Performance**: Sobel is faster but less refined. Canny is slower but produces cleaner edges.
- **Color Images**: For RGB images, apply edge detection to each channel or convert to grayscale first.

---

### Extensions
- **Thresholding Sobel**: Apply a threshold to Sobel magnitude for binary edges.
- **Canny Parameter Tuning**: Experiment with different thresholds or Gaussian $\(\sigma\)$.
- **Color Edge Detection**: Extend to RGB images by processing each channel.
- **Edge Direction Analysis**: Visualize Sobel gradient directions using a color map.

