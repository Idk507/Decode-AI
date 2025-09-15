### Image Enhancement: Sharpening, Smoothing, Contrast Stretching, and Gamma Correction

**Image enhancement** techniques improve the visual quality of an image or make it more suitable for specific applications by enhancing features like sharpness, smoothness, or contrast. This response covers **sharpening** (e.g., unsharp masking), **smoothing** (e.g., Gaussian blur, median filtering), **contrast stretching**, and **gamma correction**, including their mathematical foundations, practical applications, and a Python code implementation using OpenCV and NumPy. This builds on previous topics like color processing and image compression.

---

### 1. Overview of Image Enhancement Techniques

#### Sharpening (Unsharp Masking)
- **Definition**: Enhances edges and fine details by amplifying high-frequency components.
- **Unsharp Masking**:
  - Create a blurred (low-pass) version of the image.
  - Subtract the blurred image from the original to isolate high-frequency details (edges).
  - Add the amplified high-frequency component back to the original image.
- **Purpose**: Make edges crisper, improving perceived sharpness.
- **Applications**: Photography, medical imaging, text enhancement.

#### Smoothing
- **Definition**: Reduces noise and small variations by averaging pixel intensities.
- **Gaussian Blur**:
  - Convolves the image with a Gaussian kernel to smooth while preserving structure:
   <img width="331" height="95" alt="image" src="https://github.com/user-attachments/assets/f1c27a05-792e-4c1d-8e3e-54271679db16" />

  - $\( \sigma \)$ : Controls blur strength.
- **Median Filtering**:
  - Replaces each pixel with the median value in a local neighborhood (e.g., 3x3).
  - Effective for salt-and-pepper noise.
- **Purpose**: Remove noise, prepare images for further processing.
- **Applications**: Noise reduction, preprocessing for segmentation or feature detection.

#### Contrast Stretching
- **Definition**: Expands the range of pixel intensities to enhance contrast.
- **Technique**:
  - Linearly map pixel values to a new range (e.g., [0, 255]):
    <img width="455" height="88" alt="image" src="https://github.com/user-attachments/assets/95e66126-38af-44a4-afd6-44be31896d77" />

    - $\( a, b \)$: Desired output range (e.g., 0, 255).
- **Purpose**: Improve visibility of details in low-contrast images.
- **Applications**: Medical imaging, remote sensing, photography.

#### Gamma Correction
- **Definition**: Adjusts image brightness non-linearly to account for human perception or display characteristics.
- **Formula**:
 <img width="162" height="59" alt="image" src="https://github.com/user-attachments/assets/41806099-53d4-4db6-bf23-5fd8953aed78" />

  - $\( \gamma < 1 \)$: Brightens (expands dark regions).
  - $\( \gamma > 1 \)$: Darkens (expands bright regions).
  - $\( c \)$: Scaling factor (often 1).
- **Purpose**: Correct for non-linear display response or enhance perceptual contrast.
- **Applications**: Display calibration, image preprocessing.

---

### 2. Mathematical Concepts

#### Unsharp Masking
<img width="700" height="597" alt="image" src="https://github.com/user-attachments/assets/6e9d4072-7512-4645-bd6b-a0f3e3441ff0" />

#### Gaussian Blur
- **Kernel**:
  <img width="343" height="126" alt="image" src="https://github.com/user-attachments/assets/61c92baf-1225-4701-ac67-d562967be9f7" />

- **Convolution**: $\( I_{\text{blur}} = I * G \)$.
- **Effect**: Smooths by weighting nearby pixels, reducing noise but blurring edges.

#### Median Filtering
- For a pixel in a $\( k \times k \)$ window:
  - Collect intensities in the window.
  - Replace the center pixel with the median value.
- **Effect**: Robust to outliers (e.g., salt-and-pepper noise), preserves edges better than mean filtering.

#### Contrast Stretching
- **Formula**:
<img width="360" height="89" alt="image" src="https://github.com/user-attachments/assets/53cfad84-1b7f-4520-976a-ea5bda5a6c21" />

- **Effect**: Maps input intensities to full range, enhancing contrast.
- <img width="673" height="111" alt="image" src="https://github.com/user-attachments/assets/1c944dd6-4cfb-4a4e-b2fe-56a82e402819" />


#### Gamma Correction
- **Formula**:
 <img width="277" height="89" alt="image" src="https://github.com/user-attachments/assets/535c4e99-5373-4448-b19f-05f530cc5131" />

- <img width="681" height="169" alt="image" src="https://github.com/user-attachments/assets/99d6142b-5c52-4ecd-aec8-bcab70689df9" />

  - Brightens mid-tones.

---

### 3. Practical Considerations
- **Sharpening**:
  - Adjust $\( \alpha \)$ (e.g., 0.5–2) to control sharpening strength.
  - Over-sharpening introduces artifacts (e.g., halos around edges).
- **Smoothing**:
  - **Gaussian Blur**: Choose $\( \sigma \)$ (e.g., 1–3) and kernel size (e.g., 5x5).
  - **Median Filtering**: Use small windows (e.g., 3x3) for noise; larger windows blur more.
  - Preprocess noisy images before feature detection or segmentation.
- **Contrast Stretching**:
  - Use percentiles (e.g., 1st and 99th) instead of min/max to avoid outliers.
  - Apply per channel for color images or in LAB space for perceptual uniformity.
- **Gamma Correction**:
  - Choose $\( \gamma \)$ based on display or desired effect (e.g., 0.5 for brightening, 2.0 for darkening).
  - Normalize input to [0, 1] to avoid overflow.
- **Applications**:
  - **Sharpening**: Enhance details in photography, medical imaging.
  - **Smoothing**: Noise reduction, preprocessing for segmentation.
  - **Contrast Stretching**: Improve low-contrast images (e.g., X-rays).
  - **Gamma Correction**: Display calibration, perceptual enhancement.
- **Limitations**:
  - Sharpening amplifies noise if not pre-smoothed.
  - Smoothing blurs details; balance noise reduction and edge preservation.
  - Contrast stretching may exaggerate noise in flat regions.
  - Gamma correction may alter color balance in RGB images.
- **Color Images**:
  - Apply techniques to each RGB channel or use LAB/HSV for perceptually accurate results.

---

### 4. Code Implementation in Python

Below is a Python code example using OpenCV and NumPy to demonstrate:
- **Sharpening**: Unsharp masking with Gaussian blur.
- **Smoothing**: Gaussian blur and median filtering.
- **Contrast Stretching**: Linear stretch to [0, 255].
- **Gamma Correction**: Apply non-linear transformation with \( \gamma = 0.5 \).

```
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

# 1. Sharpening (Unsharp Masking)
alpha = 1.5  # Sharpening strength
img_blur = cv2.GaussianBlur(img_rgb, (5, 5), 1)
img_sharp = cv2.addWeighted(img_rgb, 1 + alpha, img_blur, -alpha, 0)

# 2. Smoothing: Gaussian Blur
img_gaussian = cv2.GaussianBlur(img_rgb, (5, 5), 1)

# 3. Smoothing: Median Filtering
img_median = cv2.medianBlur(img_rgb, 5)

# 4. Contrast Stretching
img_float = img_rgb.astype(np.float32)
min_val = np.percentile(img_float, 1, axis=(0, 1))
max_val = np.percentile(img_float, 99, axis=(0, 1))
img_stretched = (img_float - min_val) / (max_val - min_val + 1e-10) * 255
img_stretched = np.clip(img_stretched, 0, 255).astype(np.uint8)

# 5. Gamma Correction
gamma = 0.5
img_float = img_rgb.astype(np.float32) / 255.0
img_gamma = 255 * (img_float ** gamma)
img_gamma = np.clip(img_gamma, 0, 255).astype(np.uint8)

# 6. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

# Sharpened Image
plt.subplot(2, 3, 2)
plt.title(f'Sharpened (α={alpha})')
plt.imshow(img_sharp)
plt.axis('off')

# Gaussian Blur
plt.subplot(2, 3, 3)
plt.title('Gaussian Blur')
plt.imshow(img_gaussian)
plt.axis('off')

# Median Filtering
plt.subplot(2, 3, 4)
plt.title('Median Filtering')
plt.imshow(img_median)
plt.axis('off')

# Contrast Stretching
plt.subplot(2, 3, 5)
plt.title('Contrast Stretching')
plt.imshow(img_stretched)
plt.axis('off')

# Gamma Correction
plt.subplot(2, 3, 6)
plt.title(f'Gamma Correction (γ={gamma})')
plt.imshow(img_gamma)
plt.axis('off')

plt.tight_layout()
plt.show()

# 7. Print Sample Pixel Values
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original RGB: {img_rgb[y, x]}")
print(f"Sharpened RGB: {img_sharp[y, x]}")
print(f"Gaussian Blur RGB: {img_gaussian[y, x]}")
print(f"Median Filtering RGB: {img_median[y, x]}")
print(f"Contrast Stretched RGB: {img_stretched[y, x]}")
print(f"Gamma Corrected RGB: {img_gamma[y, x]}")
</xaiArtifact>

---

### Code Explanation
1. **Loading Image**:
   - Loads a color image using `cv2.imread` and converts BGR to RGB.

2. **Sharpening (Unsharp Masking)**:
   - Applies Gaussian blur (5x5, $\( \sigma = 1 \)$).
   - Computes sharpened image using `cv2.addWeighted` with $\( \alpha = 1.5 \)$.

3. **Smoothing**:
   - **Gaussian Blur**: Applies 5x5 kernel with $\( \sigma = 1 \)$.
   - **Median Filtering**: Uses 5x5 window to compute median per channel.

4. **Contrast Stretching**:
   - Uses 1st and 99th percentiles to avoid outliers.
   - Scales each channel to [0, 255].

5. **Gamma Correction**:
   - Normalizes image to [0, 1], applies $\( \gamma = 0.5 \)$, and scales back to [0, 255].

6. **Visualization**:
   - Displays original, sharpened, Gaussian blurred, median filtered, contrast-stretched, and gamma-corrected images.

7. **Output**:
   - Prints RGB values at (100, 100) for all images.

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide an image file (`image.jpg`) in the same directory or update the `image_path` variable. Use a color image with varied details for meaningful results.

---

### Expected Output
- A figure with six subplots:
  - Original RGB image.
  - Sharpened image (crisper edges).
  - Gaussian blurred image (smoothed, less noise).
  - Median filtered image (noise reduced, edges preserved).
  - Contrast-stretched image (enhanced contrast).
  - Gamma-corrected image (brighter mid-tones with \( \gamma = 0.5 \)).
- Printed output, e.g.:
  ```
  Pixel at (100, 100):
  Original RGB: [150 100  50]
  Sharpened RGB: [160 110  60]
  Gaussian Blur RGB: [145  95  45]
  Median Filtering RGB: [148  98  48]
  Contrast Stretched RGB: [170 120  70]
  Gamma Corrected RGB: [180 140  90]
  ```

---

### Practical Notes
- **Sharpening**:
  - Adjust $\( \alpha \)$ (e.g., 0.5–2) for desired sharpness.
  - Pre-smooth noisy images to avoid amplifying noise.
- **Smoothing**:
  - **Gaussian**: Increase $\( \sigma \)$ or kernel size for stronger blur; balance detail loss.
  - **Median**: Use larger windows (e.g., 7x7) for heavy noise, but may blur details.
- **Contrast Stretching**:
  - Use percentiles to handle outliers (e.g., 1% and 99%).
  - Apply in LAB space for perceptual uniformity.
- **Gamma Correction**:
  - Choose $\( \gamma \)$ based on display or effect (e.g., 0.5 brightens, 2.0 darkens).
  - Apply per channel or in luminance space (e.g., LAB’s L channel).
- **Applications**:
  - **Sharpening**: Enhance edges in photography, medical imaging.
  - **Smoothing**: Preprocess for feature detection or segmentation.
  - **Contrast Stretching**: Improve low-contrast images (e.g., X-rays).
  - **Gamma Correction**: Calibrate for monitors or enhance visibility.
- **Limitations**:
  - Sharpening can amplify noise or introduce halos.
  - Smoothing blurs edges; median filtering preserves edges better.
  - Contrast stretching may exaggerate noise in flat regions.
  - Gamma correction may alter color balance if applied per channel.
- **Color Images**:
  - Process RGB channels separately or use LAB/HSV for perceptually accurate results.

---

### Mathematical Insights
<img width="739" height="523" alt="image" src="https://github.com/user-attachments/assets/bcfcc2b1-a141-488a-ada0-cd869efc45ed" />


---

### Extensions
- **Advanced Sharpening**: Use Laplacian or high-pass filters.
- **Bilateral Filtering**: Smooth while preserving edges.
- **Adaptive Contrast**: Apply local contrast stretching (e.g., CLAHE).
- **Gamma in LAB**: Apply gamma correction to L channel for perceptual accuracy.
- **Integration**: Combine with color balancing, SIFT, or compression.

