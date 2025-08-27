### Thresholding: Global Thresholding and Otsu’s Method

**Thresholding** is a fundamental image processing technique used to segment an image by converting it into a binary image, where pixels are classified into two categories (e.g., foreground and background) based on their intensity values. **Global thresholding** applies a single threshold value across the entire image, while **Otsu’s method** is an automatic global thresholding technique that optimally selects the threshold to maximize class separability. This explanation covers the mathematical concepts, practical applications, and a detailed Python code implementation using OpenCV and NumPy to demonstrate global thresholding and Otsu’s method.

---

### 1. What is Thresholding?

Thresholding transforms an image (typically grayscale) into a binary image by comparing each pixel’s intensity to a threshold value. Pixels above the threshold are assigned one value (e.g., 255 for white), and those below are assigned another (e.g., 0 for black).

- **Definition**: For a grayscale image $\( I(x, y) \)$ and a threshold $\( T \)$, the output binary image $\( B(x, y) \)$ is:
  <img width="403" height="97" alt="image" src="https://github.com/user-attachments/assets/df5a44fb-2003-44e6-be52-4a9fba0fe699" />

- **Purpose**: Simplifies images for segmentation, object detection, or feature extraction by separating regions of interest (e.g., objects) from the background.
- **Applications**:
  - Object segmentation (e.g., separating text from background in document scanning).
  - Preprocessing for computer vision tasks (e.g., edge detection, shape analysis).
  - Medical imaging (e.g., isolating tumors in MRI scans).

#### Types of Thresholding
- **Global Thresholding**: Uses a single threshold $\( T \)$ for the entire image.
- **Adaptive Thresholding**: Uses varying thresholds based on local image regions (not covered here).
- **Otsu’s Method**: A global thresholding technique that automatically determines the optimal threshold.

---

### 2. Global Thresholding

Global thresholding applies a fixed threshold value to all pixels, manually chosen or automatically determined (e.g., by Otsu’s method).

#### Mathematical Concept
- **Manual Global Thresholding**:
  - Select a threshold $\( T \)$ (e.g., 128 for an 8-bit image).
  - Assign pixels to binary values based on whether $\( I(x, y) \geq T \)$.
  - Example: For $\( T = 128 \)$, pixels with intensity 150 become 255 (white), and those with 100 become 0 (black).
- **Challenges**:
  - Choosing an appropriate $\( T \)$ requires knowledge of the image’s intensity distribution.
  - Poorly chosen thresholds can lead to loss of important details or inclusion of noise.

#### Otsu’s Method
Otsu’s method is an automatic global thresholding technique that selects the optimal threshold by maximizing the **between-class variance** of the foreground and background classes, assuming a bimodal histogram (two distinct intensity clusters).

##### Mathematical Formulation
1. **Histogram**:
   - Compute the histogram $\( h(v) \)$ of the grayscale image, where $\( v \)$ is the intensity (0 to 255 for 8-bit), and $\( h(v) \)$ is the number of pixels with intensity \( v \).
   - Normalize the histogram: $\( p(v) = h(v) / N \)$, where $\( N \)$ is the total number of pixels.

2. **Class Probabilities and Means**:
   - For a threshold $\( T \)$, pixels are divided into two classes:
     - **Background** $(\( C_0 \))$ : Pixels with intensity $\( v < T \)$.
     - **Foreground** $(\( C_1 \))$ : Pixels with intensity $\( v \geq T \)$ .
   - Class probabilities:
     <img width="489" height="201" alt="image" src="https://github.com/user-attachments/assets/4883d333-bdf7-47a6-8279-3cb678d663b4" />

   - Class means:
     <img width="472" height="179" alt="image" src="https://github.com/user-attachments/assets/23431a7e-0015-4119-8834-e7092b6711c0" />

   - Total mean: $\( \mu_T = w_0 \mu_0 + w_1 \mu_1 \)$ .

3. **Between-Class Variance**:
   - Otsu’s method maximizes the between-class variance:
    <img width="284" height="62" alt="image" src="https://github.com/user-attachments/assets/70228404-b17a-4d37-a430-c9969a6d4163" />

   - Alternatively, it minimizes the within-class variance, but maximizing $\( \sigma_B^2 \)$ is equivalent and simpler.

4. **Optimal Threshold**:
   - Iterate over all possible thresholds $\( T \)$ (0 to 255) and select the $\( T \)$ that maximizes $\( \sigma_B^2(T) \)$.
   - The optimal $\( T \)$ separates the histogram into two classes with minimal overlap.

##### Example
Consider a 3x3 grayscale image:
<img width="253" height="117" alt="image" src="https://github.com/user-attachments/assets/c506bf1d-4f1a-4bcc-8cfd-8b54a59b4668" />

- **Histogram**:
  <img width="607" height="84" alt="image" src="https://github.com/user-attachments/assets/021aa10a-8de9-402f-826a-a516d6f7f9fd" />

<img width="988" height="453" alt="image" src="https://github.com/user-attachments/assets/de39d7c5-c319-42bc-a2d9-5e813e41bb7c" />

- **Output Image**:
 <img width="242" height="146" alt="image" src="https://github.com/user-attachments/assets/ae73ef76-f714-4249-9d73-556f38e2d942" />

- **Result**: Pixels ≥ 125 (150) become white (255); others become black (0).

#### Properties
- **Automatic**: Otsu’s method requires no manual threshold selection.
- **Optimal for Bimodal Histograms**: Works best when the histogram has two distinct peaks (e.g., foreground and background).
- **Applications**:
  - Document scanning (separating text from background).
  - Object detection (isolating objects from uniform backgrounds).
  - Medical imaging (segmenting regions of interest).
- **Limitations**:
  - Assumes a bimodal histogram; performs poorly with unimodal or complex histograms.
  - Sensitive to noise, which can distort the histogram.
  - Global approach may fail in images with varying lighting (adaptive thresholding is better).

---

### 3. Practical Considerations
- **Preprocessing**: Apply Gaussian blur to reduce noise before thresholding, as noise can create false peaks in the histogram.
- **Color Images**: Convert to grayscale or use a specific channel (e.g., V in HSV) before thresholding.
- **Bit Depth**: Typically applied to 8-bit images (0–255), but can be extended to 16-bit with appropriate scaling.
- **Adaptive Thresholding**: For non-uniform lighting, consider adaptive methods or CLAHE (Contrast Limited Adaptive Histogram Equalization) before thresholding.

---

### 4. Code Implementation in Python

Below is a Python code example using OpenCV and NumPy to demonstrate global thresholding and Otsu’s method. The code:
- Applies manual global thresholding with a fixed threshold.
- Applies Otsu’s method to automatically determine the threshold.
- Visualizes the original image, histograms, and thresholded results.

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

# 1. Manual Global Thresholding
T_manual = 128  # Fixed threshold
_, img_thresh_manual = cv2.threshold(img, T_manual, 255, cv2.THRESH_BINARY)

# 2. Otsu's Method
T_otsu, img_thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Compute Histograms
def compute_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

hist_original = compute_histogram(img)
hist_thresh_manual = compute_histogram(img_thresh_manual)
hist_thresh_otsu = compute_histogram(img_thresh_otsu)

# 4. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Grayscale')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Manual Thresholded Image
plt.subplot(2, 3, 2)
plt.title(f'Manual Threshold (T={T_manual})')
plt.imshow(img_thresh_manual, cmap='gray')
plt.axis('off')

# Otsu Thresholded Image
plt.subplot(2, 3, 3)
plt.title(f'Otsu Threshold (T={T_otsu})')
plt.imshow(img_thresh_otsu, cmap='gray')
plt.axis('off')

# Histograms
plt.subplot(2, 3, 4)
plt.title('Original Histogram')
plt.plot(hist_original, color='gray')
plt.axvline(x=T_manual, color='r', linestyle='--', label=f'Manual T={T_manual}')
plt.axvline(x=T_otsu, color='b', linestyle='--', label=f'Otsu T={T_otsu}')
plt.xlim([0, 256])
plt.legend()

plt.subplot(2, 3, 5)
plt.title('Manual Threshold Histogram')
plt.plot(hist_thresh_manual, color='gray')
plt.xlim([0, 256])

plt.subplot(2, 3, 6)
plt.title('Otsu Threshold Histogram')
plt.plot(hist_thresh_otsu, color='gray')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

# 5. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original Grayscale: {img[y, x]}")
print(f"Manual Threshold (T={T_manual}): {img_thresh_manual[y, x]}")
print(f"Otsu Threshold (T={T_otsu}): {img_thresh_otsu[y, x]}")
```

---

### Code Explanation
1. **Loading Image**:
   - Loads a grayscale image using `cv2.imread` with `cv2.IMREAD_GRAYSCALE`.

2. **Manual Global Thresholding**:
   - Uses `cv2.threshold` with a fixed threshold \( T = 128 \), applying binary thresholding (`cv2.THRESH_BINARY`).

3. **Otsu’s Method**:
   - Uses `cv2.threshold` with `cv2.THRESH_OTSU`, which automatically computes the optimal threshold \( T_otsu \).
   - Combines with `cv2.THRESH_BINARY` for binary output.

4. **Histogram Computation**:
   - The `compute_histogram` function calculates the histogram of intensity values using `cv2.calcHist`.

5. **Visualization**:
   - Displays the original grayscale image, manually thresholded image, and Otsu-thresholded image.
   - Plots histograms for each, with vertical lines indicating the manual and Otsu thresholds on the original histogram.

6. **Pixel Value Output**:
   - Prints pixel values at (100, 100) to show the effect of thresholding.

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide an image file (`image.jpg`) in the same directory or update the `image_path` variable.

---

### Expected Output
- A figure with six subplots:
  - Original grayscale image.
  - Manually thresholded image (using $\( T = 128 \)$).
  - Otsu-thresholded image (using automatically determined $\( T_otsu \))$.
  - Original histogram with manual and Otsu thresholds marked.
  - Histogram of the manually thresholded image (two peaks at 0 and 255).
  - Histogram of the Otsu-thresholded image (two peaks at 0 and 255).
- Printed pixel values, e.g.:
  ```
  Pixel at (100, 100):
  Original Grayscale: 113
  Manual Threshold (T=128): 0
  Otsu Threshold (T=150): 0
  ```

---

### Practical Notes
- **Manual Thresholding**: Requires trial and error to choose $\( T \)$; unsuitable for images with varying lighting.
- **Otsu’s Method**: Effective for images with bimodal histograms (e.g., clear foreground/background separation). Fails if the histogram is unimodal or noisy.
- **Preprocessing**: Apply Gaussian blur to reduce noise before thresholding:
  ```python
  img = cv2.GaussianBlur(img, (5, 5), 0)
  ```
- **Color Images**: Convert to grayscale or use a specific channel (e.g., V in HSV) before thresholding.
- **Threshold Types**: OpenCV supports other types (e.g., `cv2.THRESH_BINARY_INV`, `cv2.THRESH_TRUNC`), which can be explored for different effects.

---

### Mathematical Insights
- **Otsu’s Criterion**: Maximizes between-class variance:
  <img width="259" height="56" alt="image" src="https://github.com/user-attachments/assets/e38876e5-13e1-40c5-86b5-bdf3fda0884c" />

  This ensures the best separation between foreground and background.
- **Histogram**: The normalized histogram $\( p(v) \)$ approximates the probability density of intensities.
- **Bimodal Assumption**: Otsu assumes two classes; performance degrades with complex histograms.

---

### Extensions
- **Adaptive Thresholding**: Implement adaptive thresholding (`cv2.adaptiveThreshold`) for non-uniform lighting.
- **Multi-level Thresholding**: Extend Otsu’s method to multiple thresholds for segmenting multiple regions.
- **Color Thresholding**: Apply thresholding in HSV space for color-based segmentation.
- **Quantitative Evaluation**: Measure segmentation accuracy (e.g., using ground truth masks).

