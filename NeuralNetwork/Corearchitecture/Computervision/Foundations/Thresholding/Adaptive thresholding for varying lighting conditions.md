### Adaptive Thresholding for Varying Lighting Conditions

**Adaptive thresholding** is an image processing technique used to segment an image into a binary image by applying locally varying thresholds, making it particularly effective for images with non-uniform lighting or shading. Unlike global thresholding (e.g., Otsu’s method), which uses a single threshold for the entire image, adaptive thresholding computes thresholds based on local pixel neighborhoods, adapting to changes in illumination. This explanation covers the mathematical concepts, practical applications, and a detailed Python code implementation using OpenCV and NumPy to demonstrate adaptive thresholding under varying lighting conditions.

---

### 1. What is Adaptive Thresholding?

Adaptive thresholding determines a threshold for each pixel based on the intensity values in its local neighborhood, making it suitable for images where lighting varies across regions (e.g., shadows, uneven illumination).

- **Definition**: For a grayscale image $\( I(x, y) \)$, adaptive thresholding computes a threshold $\( T(x, y) \)$ for each pixel based on a local region (e.g., a $\( k \times k \)$ window). The output binary image $\( B(x, y) \)$  is:
  <img width="419" height="94" alt="image" src="https://github.com/user-attachments/assets/c6d34ab1-1aa6-40e2-859a-fa08da232f63" />

- **Purpose**: Segments images with varying lighting, ensuring robust separation of foreground and background in each region.
- **Applications**:
  - Document scanning (e.g., text extraction from papers with shadows or uneven lighting).
  - Medical imaging (e.g., segmenting cells in microscopy images with gradients).
  - Object detection in real-world scenes with inconsistent illumination.
  - Preprocessing for OCR (Optical Character Recognition) or feature extraction.

---

### 2. Adaptive Thresholding Techniques

Adaptive thresholding calculates a local threshold for each pixel based on its surrounding neighborhood. The two common methods implemented in OpenCV are:

1. **Mean Adaptive Thresholding**:
   - Computes the threshold as the mean of the pixel intensities in a local neighborhood, minus an optional constant $\( C \)$:
    <img width="615" height="64" alt="image" src="https://github.com/user-attachments/assets/c43d5935-3007-4cd2-8213-f30cc7682bd8" />

   - The neighborhood is typically a square window of size $\( k \times k \)$.
   - $\( C \)$ adjusts the threshold to fine-tune segmentation (e.g., making it stricter or more lenient).

2. **Gaussian Adaptive Thresholding**:
   - Computes the threshold as a weighted mean of the neighborhood, using a Gaussian window to give more weight to pixels closer to the center:
    <img width="578" height="90" alt="image" src="https://github.com/user-attachments/assets/e15781a4-da30-4090-9c94-77323b025585" />

     Where $\( w(x', y') \)$ is a Gaussian weight based on distance from $\((x, y)\)$.
   - Gaussian weighting reduces sensitivity to noise and provides smoother thresholds.

#### Key Parameters
- **Neighborhood Size $(\( k \))$ **: The size of the window (e.g., 11x11 pixels). Larger windows capture broader lighting variations but may blur fine details.
- **Constant $(\( C \))$ **: Adjusts the threshold to control sensitivity. Positive $\( C \)$ makes the threshold lower (more pixels classified as foreground); negative $\( C \)$ makes it higher.
- **Threshold Type**: Binary (0 or 255) or inverse binary (255 or 0).

#### Comparison with Global Thresholding
- **Global Thresholding (e.g., Otsu’s)**:
  - Uses a single threshold for the entire image.
  - Fails in images with varying lighting, as a single $\( T \)$ cannot separate foreground and background effectively.
- **Adaptive Thresholding**:
  - Adapts to local lighting conditions, making it robust for non-uniform illumination.
  - More computationally intensive due to per-pixel threshold calculations.

#### Example
Consider a 3x3 grayscale image with varying lighting (8-bit):
<img width="785" height="329" alt="image" src="https://github.com/user-attachments/assets/2fbd82d4-6de6-4d9c-b4f9-c0f7916b1c48" />

  - Fails to capture low-intensity foreground (e.g., 50–65) due to high threshold.
- **Mean Adaptive Thresholding (3x3 window, $\( C = 5 \)$ )**:
 <img width="929" height="385" alt="image" src="https://github.com/user-attachments/assets/2e69e102-da41-4179-88b6-e8374a12c15a" />

  - Better captures local variations compared to global thresholding.

---

### 3. Practical Considerations
- **Neighborhood Size**:
  - Small windows (e.g., 5x5) capture fine details but are sensitive to noise.
  - Large windows (e.g., 21x21) handle broad lighting variations but may miss small features.
  - Must be odd (e.g., 11) to center the window on each pixel.
- **Constant $\( C \)$ **:
  - Adjusts sensitivity. Typical values: 2–10. Experiment to balance foreground/background separation.
- **Preprocessing**:
  - Apply Gaussian blur to reduce noise before thresholding:
    ```python
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ```
  - Enhances robustness, as noise can cause erratic local thresholds.
- **Color Images**:
  - Convert to grayscale or process a specific channel (e.g., V in HSV) before thresholding.
- **Bit Depth**:
  - Typically applied to 8-bit images (0–255), but can extend to 16-bit with scaling.
- **Performance**:
  - More computationally intensive than global thresholding due to local calculations.
  - OpenCV optimizes adaptive thresholding for efficiency.

---

### 4. Code Implementation in Python

Below is a Python code example using OpenCV and NumPy to demonstrate adaptive thresholding with mean and Gaussian methods, compared to global thresholding (Otsu’s method). The code visualizes the original image, thresholded results, and their histograms.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image_path = 'image.jpg'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# Preprocess: Apply Gaussian blur to reduce noise
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# 1. Global Thresholding (Otsu's Method)
T_otsu, img_otsu = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 2. Adaptive Thresholding (Mean)
img_adaptive_mean = cv2.adaptiveThreshold(
    img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5
)

# 3. Adaptive Thresholding (Gaussian)
img_adaptive_gaussian = cv2.adaptiveThreshold(
    img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
)

# 4. Compute Histograms
def compute_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

hist_original = compute_histogram(img_blur)
hist_otsu = compute_histogram(img_otsu)
hist_adaptive_mean = compute_histogram(img_adaptive_mean)
hist_adaptive_gaussian = compute_histogram(img_adaptive_gaussian)

# 5. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 4, 1)
plt.title('Original Grayscale (Blurred)')
plt.imshow(img_blur, cmap='gray')
plt.axis('off')

# Otsu Thresholded Image
plt.subplot(2, 4, 2)
plt.title(f'Otsu Threshold (T={T_otsu})')
plt.imshow(img_otsu, cmap='gray')
plt.axis('off')

# Adaptive Mean Thresholded Image
plt.subplot(2, 4, 3)
plt.title('Adaptive Mean (11x11, C=5)')
plt.imshow(img_adaptive_mean, cmap='gray')
plt.axis('off')

# Adaptive Gaussian Thresholded Image
plt.subplot(2, 4, 4)
plt.title('Adaptive Gaussian (11x11, C=5)')
plt.imshow(img_adaptive_gaussian, cmap='gray')
plt.axis('off')

# Histograms
plt.subplot(2, 4, 5)
plt.title('Original Histogram')
plt.plot(hist_original, color='gray')
plt.axvline(x=T_otsu, color='b', linestyle='--', label=f'Otsu T={T_otsu}')
plt.xlim([0, 256])
plt.legend()

plt.subplot(2, 4, 6)
plt.title('Otsu Histogram')
plt.plot(hist_otsu, color='gray')
plt.xlim([0, 256])

plt.subplot(2, 4, 7)
plt.title('Adaptive Mean Histogram')
plt.plot(hist_adaptive_mean, color='gray')
plt.xlim([0, 256])

plt.subplot(2, 4, 8)
plt.title('Adaptive Gaussian Histogram')
plt.plot(hist_adaptive_gaussian, color='gray')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original Grayscale (Blurred): {img_blur[y, x]}")
print(f"Otsu Threshold (T={T_otsu}): {img_otsu[y, x]}")
print(f"Adaptive Mean Threshold: {img_adaptive_mean[y, x]}")
print(f"Adaptive Gaussian Threshold: {img_adaptive_gaussian[y, x]}")
```

---

### Code Explanation
1. **Loading Image**:
   - Loads a grayscale image using `cv2.imread` with `cv2.IMREAD_GRAYSCALE`.

2. **Preprocessing**:
   - Applies a 5x5 Gaussian blur to reduce noise, improving thresholding robustness.

3. **Global Thresholding (Otsu’s)**:
   - Uses `cv2.threshold` with `cv2.THRESH_OTSU` to compute the optimal global threshold.

4. **Adaptive Thresholding**:
   - Uses `cv2.adaptiveThreshold` with:
     - `cv2.ADAPTIVE_THRESH_MEAN_C`: Mean-based thresholding.
     - `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`: Gaussian-weighted thresholding.
     - Block size = 11 (11x11 neighborhood).
     - $\( C = 5 \)$ (constant subtracted from the mean).

5. **Histogram Computation**:
   - The `compute_histogram` function calculates the histogram of intensity values using `cv2.calcHist`.

6. **Visualization**:
   - Displays the original (blurred) image, Otsu-thresholded image, and adaptive thresholded images (mean and Gaussian).
   - Plots histograms, with the Otsu threshold marked on the original histogram.

7. **Pixel Value Output**:
   - Prints pixel values at (100, 100) to show the effect of different thresholding methods.

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide an image file (`image.jpg`) in the same directory or update the `image_path` variable. Use an image with varying lighting (e.g., shadows, gradients) for best results.

---

### Expected Output
- A figure with eight subplots:
  - Original grayscale image (after blur).
  - Otsu-thresholded image (global threshold).
  - Adaptive mean-thresholded image.
  - Adaptive Gaussian-thresholded image.
  - Histogram of the original image with the Otsu threshold marked.
  - Histograms of the thresholded images (showing peaks at 0 and 255).
- Printed pixel values, e.g.:
  ```
  Pixel at (100, 100):
  Original Grayscale (Blurred): 113
  Otsu Threshold (T=150): 0
  Adaptive Mean Threshold: 255
  Adaptive Gaussian Threshold: 255
  ```

---

### Practical Notes
- **Neighborhood Size**: Adjust the block size (e.g., 11, 21) based on the scale of lighting variations. Larger blocks handle broader gradients but may miss fine details.
- **Constant $\( C \)$ **: Tune $\( C \)$ (e.g., 2–10) to balance foreground/background separation. Higher $\( C \)$ makes more pixels background.
- **Preprocessing**: Gaussian blur is critical for noisy images to prevent erratic thresholds.
- **Color Images**: Convert to grayscale or process a specific channel (e.g., V in HSV) before thresholding.
- **Applications**:
  - Document scanning under uneven lighting.
  - Segmenting objects in outdoor scenes with shadows.
  - Preprocessing for OCR or edge detection in complex lighting conditions.

---

### Mathematical Insights
- **Local Threshold**: $\( T(x, y) = \text{mean}(I_{\text{neighborhood}}) - C \)$ (mean) or a Gaussian-weighted mean.
- **Gaussian Weighting**: Uses a Gaussian kernel (e.g., $\( \sigma \)$ based on block size) to prioritize central pixels:
 <img width="407" height="75" alt="image" src="https://github.com/user-attachments/assets/9cbab447-3f67-4660-b200-2d819f358fe9" />

- **Comparison with Otsu**: Adaptive thresholding excels in non-uniform lighting, while Otsu assumes a global bimodal histogram.

---

### Extensions
- **Custom Neighborhood Sizes**: Test different block sizes (e.g., 5x5, 21x21) to evaluate their impact.
- **Color Thresholding**: Apply adaptive thresholding in HSV or LAB spaces for color images.
- **CLAHE Preprocessing**: Combine with Contrast Limited Adaptive Histogram Equalization to enhance contrast before thresholding.
- **Performance Optimization**: Explore faster implementations for large images (e.g., integral images for mean calculation).

