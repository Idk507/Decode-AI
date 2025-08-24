### Histogram Matching for Adjusting Image Intensity Distributions

**Histogram matching** (also known as histogram specification) is a histogram processing technique in image processing that adjusts the intensity distribution of a source image to match the histogram of a reference image or a specified distribution. This method is used to standardize the appearance of images, enhance contrast, or achieve consistent intensity distributions across multiple images. Unlike histogram equalization, which aims for a uniform histogram, histogram matching allows precise control over the output histogram by aligning it with a target histogram. This explanation covers the mathematical concepts, practical applications, and a detailed Python code implementation using OpenCV and NumPy to demonstrate histogram matching.

---

### 1. What is Histogram Matching?

Histogram matching transforms the pixel intensities of a source image so that its histogram matches that of a reference image or a specified histogram. This is particularly useful when you want images to have similar intensity distributions for consistent visual appearance or analysis.

- **Definition**: Given a source image with histogram $\( h_s(v) \)$ and a reference image with histogram $\( h_r(v) \)$, histogram matching finds a transformation that maps the source image’s intensities to new values such that the resulting histogram approximates $\( h_r(v) \)$.
- **Purpose**: Adjusts the intensity distribution to achieve a desired visual effect or to normalize images for tasks like image comparison, stitching, or classification.
- **Applications**:
  - Standardizing medical images (e.g., matching MRI scans across patients).
  - Image enhancement for consistent lighting in photography.
  - Preprocessing for computer vision tasks requiring uniform intensity distributions.
  - Color correction in multi-image datasets (e.g., video frames, satellite imagery).

---

### 2. Mathematical Concept of Histogram Matching

Histogram matching involves mapping the intensity levels of the source image to those of the reference image using their **cumulative distribution functions (CDFs)**. The process ensures that the source image’s intensity distribution matches the reference image’s distribution.

#### Steps in Histogram Matching
1. **Compute Histograms**:
 <img width="1034" height="189" alt="image" src="https://github.com/user-attachments/assets/7fe48ca9-b04d-4ad9-8629-103f3d8fb261" />


2. **Compute Cumulative Distribution Functions (CDFs)**:
   <img width="586" height="143" alt="image" src="https://github.com/user-attachments/assets/55b43b21-13bc-4e34-8864-467bce28dec4" />


3. **Map Intensities**:
   - For each intensity \( v \) in the source image, find the intensity \( v' \) in the reference image such that:
     <img width="290" height="57" alt="image" src="https://github.com/user-attachments/assets/7d5f550d-453a-4187-9d9d-e6c929506995" />

   - This is done by finding the closest \( v' \) where:
    <img width="310" height="44" alt="image" src="https://github.com/user-attachments/assets/d597eb9d-2024-4ca9-8666-b2978b9041eb" />

     Here, $\( \text{CDF}_r^{-1} \)$ is the inverse CDF (or lookup table) of the reference histogram.
   - In practice, for each $\( v \)$, compute $\( \text{CDF}_s(v) \)$, find the closest $\( \text{CDF}_r(v') \)$, and assign $\( v' \)$ to the output pixel.

4. **Apply Transformation**:
   - Create a lookup table (LUT) mapping each source intensity $\( v \)$ to the corresponding $\( v' \)$.
   - Transform each pixel in the source image using the LUT.

#### Example
Consider a 3x3 grayscale source image:
<img width="258" height="112" alt="image" src="https://github.com/user-attachments/assets/275fcc6e-b341-483a-a626-ae137c74556a" />

And a reference image with intensities clustered around higher values (e.g., 150–200).

- **Source Histogram**:
 <img width="849" height="194" alt="image" src="https://github.com/user-attachments/assets/3d0d2aaa-b63e-4727-9303-3448bb278314" />


- **Reference Histogram** (simplified example):
<img width="902" height="215" alt="image" src="https://github.com/user-attachments/assets/160ecd7e-3910-4aee-967c-365012d9a48c" />


- **Mapping**:
<img width="926" height="171" alt="image" src="https://github.com/user-attachments/assets/808237cc-7bd0-42db-aa0b-d40017f1da4a" />

- **Output Image**:
 <img width="264" height="130" alt="image" src="https://github.com/user-attachments/assets/717ea667-5f8b-4622-a939-51d99bc3bd69" />

- **Result**: The source image’s intensities are remapped to match the reference histogram, shifting values to higher intensities.

#### Color Images
- For RGB images, histogram matching can be applied to each channel independently, but this may alter color balance.
- A better approach is to match the luminance channel (e.g., V in HSV or L in LAB) to preserve color relationships, similar to histogram equalization.

#### Properties
- **Precise Control**: Matches the exact histogram of a reference image, unlike equalization’s uniform distribution.
- **Applications**:
  - Normalizing images for consistent appearance (e.g., in medical imaging or video processing).
  - Artistic effects (e.g., mimicking the style of a reference image).
- **Limitations**:
  - Requires a reference image or histogram, unlike equalization.
  - May amplify noise if the reference histogram has sharp peaks.
  - Computationally more intensive than equalization due to CDF matching.

---

### 3. Practical Considerations
- **Reference Selection**: The reference image should have a desirable intensity distribution (e.g., high contrast, balanced lighting).
- **Noise Sensitivity**: Matching to a noisy reference can degrade the source image; preprocessing (e.g., smoothing) may be needed.
- **Color Images**: Matching in HSV or LAB spaces avoids color distortions compared to RGB.
- **Bit Depth**: Works with 8-bit or 16-bit images, but higher bit depths provide more precise matching.

---

### 4. Code Implementation in Python

Below is a Python code example using OpenCV and NumPy to perform histogram matching on grayscale and color images. The code:
- Loads a source and reference image.
- Applies histogram matching to the grayscale source image using the reference image’s histogram.
- Applies histogram matching to the Value channel of a color image in HSV space.
- Visualizes the source, reference, and matched images along with their histograms.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform histogram matching (grayscale)
def histogram_matching(source, reference):
    # Compute histograms and CDFs
    src_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, bins = np.histogram(reference.flatten(), 256, [0, 256])
    
    src_cdf = src_hist.cumsum()
    src_cdf = src_cdf / src_cdf[-1]  # Normalize to [0, 1]
    
    ref_cdf = ref_hist.cumsum()
    ref_cdf = ref_cdf / ref_cdf[-1]  # Normalize to [0, 1]
    
    # Create mapping from source to reference intensities
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Find the closest reference CDF value to source CDF
        j = np.argmin(np.abs(ref_cdf - src_cdf[i]))
        mapping[i] = j
    
    # Apply mapping to source image
    matched = cv2.LUT(source, mapping)
    return matched

# Load source and reference images
source_path = 'source.jpg'  # Replace with your source image path
reference_path = 'reference.jpg'  # Replace with your reference image path
img_source = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
img_reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

# Check if images are loaded successfully
if img_source is None or img_reference is None:
    print("Error: Could not load one or both images.")
    exit()

# 1. Grayscale Histogram Matching
img_matched = histogram_matching(img_source, img_reference)

# 2. Color Histogram Matching (HSV)
img_source_color = cv2.imread(source_path)
img_reference_color = cv2.imread(reference_path)

# Convert to HSV
img_source_hsv = cv2.cvtColor(img_source_color, cv2.COLOR_BGR2HSV)
img_reference_hsv = cv2.cvtColor(img_reference_color, cv2.COLOR_BGR2HSV)

# Split HSV channels
h_s, s_s, v_s = cv2.split(img_source_hsv)
h_r, s_r, v_r = cv2.split(img_reference_hsv)

# Match the Value channel
v_matched = histogram_matching(v_s, v_r)

# Merge back and convert to RGB
img_matched_hsv = cv2.merge([h_s, s_s, v_matched])
img_matched_color = cv2.cvtColor(img_matched_hsv, cv2.COLOR_HSV2RGB)
img_source_rgb = cv2.cvtColor(img_source_color, cv2.COLOR_BGR2RGB)
img_reference_rgb = cv2.cvtColor(img_reference_color, cv2.COLOR_BGR2RGB)

# 3. Compute Histograms
def compute_histogram(image, title):
    if len(image.shape) == 3:  # Color image
        colors = ('r', 'g', 'b')
        hist = []
        for i, color in enumerate(colors):
            hist_i = cv2.calcHist([image], [i], None, [256], [0, 256])
            hist.append((hist_i, color))
        return hist, f"{title} (RGB)"
    else:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return [(hist, 'gray')], title

hist_source, title_source = compute_histogram(img_source, 'Source Grayscale Histogram')
hist_reference, title_reference = compute_histogram(img_reference, 'Reference Grayscale Histogram')
hist_matched, title_matched = compute_histogram(img_matched, 'Matched Grayscale Histogram')

# 4. Display Results
plt.figure(figsize=(15, 10))

# Source Grayscale Image
plt.subplot(3, 3, 1)
plt.title('Source Grayscale')
plt.imshow(img_source, cmap='gray')
plt.axis('off')

# Reference Grayscale Image
plt.subplot(3, 3, 2)
plt.title('Reference Grayscale')
plt.imshow(img_reference, cmap='gray')
plt.axis('off')

# Matched Grayscale Image
plt.subplot(3, 3, 3)
plt.title('Matched Grayscale')
plt.imshow(img_matched, cmap='gray')
plt.axis('off')

# Source Color Image
plt.subplot(3, 3, 4)
plt.title('Source Color (RGB)')
plt.imshow(img_source_rgb)
plt.axis('off')

# Reference Color Image
plt.subplot(3, 3, 5)
plt.title('Reference Color (RGB)')
plt.imshow(img_reference_rgb)
plt.axis('off')

# Matched Color Image
plt.subplot(3, 3, 6)
plt.title('Matched Color (HSV)')
plt.imshow(img_matched_color)
plt.axis('off')

# Plot Histograms
plt.subplot(3, 3, 7)
plt.title(title_source)
for hist, color in hist_source:
    plt.plot(hist, color=color)
plt.xlim([0, 256])

plt.subplot(3, 3, 8)
plt.title(title_reference)
for hist, color in hist_reference:
    plt.plot(hist, color=color)
plt.xlim([0, 256])

plt.subplot(3, 3, 9)
plt.title(title_matched)
for hist, color in hist_matched:
    plt.plot(hist, color=color)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

# 5. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Source Grayscale: {img_source[y, x]}")
print(f"Reference Grayscale: {img_reference[y, x]}")
print(f"Matched Grayscale: {img_matched[y, x]}")
print(f"Source Color (RGB): {img_source_rgb[y, x]}")
print(f"Reference Color (RGB): {img_reference_rgb[y, x]}")
print(f"Matched Color (RGB): {img_matched_color[y, x]}")
</xaiArtifact>

---

### Code Explanation
1. **Histogram Matching Function**:
   - The `histogram_matching` function computes histograms and CDFs for the source and reference grayscale images.
   - Creates a lookup table (LUT) by matching source CDF values to the closest reference CDF values.
   - Applies the LUT to the source image using `cv2.LUT`.

2. **Grayscale Matching**:
   - Loads source and reference images in grayscale.
   - Applies histogram matching to align the source image’s histogram with the reference.

3. **Color Matching (HSV)**:
   - Loads source and reference images in color, converts to HSV.
   - Matches the Value (V) channel histogram to preserve color balance.
   - Merges channels and converts back to RGB for display.

4. **Histogram Computation**:
   - The `compute_histogram` function calculates histograms for grayscale or RGB images.
   - Used to visualize the intensity distributions before and after matching.

5. **Visualization**:
   - Displays source, reference, and matched images (grayscale and color) in a 3x3 grid.
   - Plots histograms for grayscale images to show the source, reference, and matched distributions.

6. **Pixel Value Output**:
   - Prints pixel values at (100, 100) to illustrate intensity changes.

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide two image files:
  - `source.jpg`: The image to be modified.
  - `reference.jpg`: The image whose histogram will be matched.
  - Update `source_path` and `reference_path` in the code.

---

### Expected Output
- A figure with nine subplots:
  - Source grayscale image.
  - Reference grayscale image.
  - Matched grayscale image (source adjusted to match reference histogram).
  - Source color image (RGB).
  - Reference color image (RGB).
  - Matched color image (HSV-based matching).
  - Histograms for source, reference, and matched grayscale images.
- The matched grayscale histogram should closely resemble the reference histogram.
- Printed pixel values, e.g.:
  ```
  Pixel at (100, 100):
  Source Grayscale: 113
  Reference Grayscale: 180
  Matched Grayscale: 178
  Source Color (RGB): [150 100  50]
  Reference Color (RGB): [180 120  80]
  Matched Color (RGB): [170 110  60]
  ```

---

### Practical Notes
- **Reference Image**: Choose a reference with a desirable histogram (e.g., high contrast, balanced intensities).
- **Color Matching**: HSV-based matching (on the V channel) preserves color relationships better than RGB channel-wise matching.
- **Noise**: Matching to a noisy reference can degrade quality; preprocess with smoothing if needed.
- **Bit Depth**: Works with 8-bit or 16-bit images, but the code assumes 8-bit for simplicity.
- **Applications**:
  - Medical imaging: Standardize intensity distributions across scans.
  - Photography: Match lighting conditions between images.
  - Video processing: Ensure consistent appearance across frames.

---

### Mathematical Insights
- **CDF Matching**: The key idea is to align the source CDF with the reference CDF:
  <img width="283" height="60" alt="image" src="https://github.com/user-attachments/assets/76bb7a8b-3148-4aa7-8d04-dacc8b6b5779" />

  This ensures the output histogram matches the reference.
- **Lookup Table**: The LUT maps each source intensity to the reference intensity with the closest CDF value, ensuring a monotonic transformation.
- **Effect**: Unlike histogram equalization (which targets a uniform histogram), histogram matching allows arbitrary histogram shapes, providing flexibility for specific visual goals.

---

### Extensions
- **Custom Histogram**: Match to a specified histogram (e.g., Gaussian) instead of a reference image.
- **CLAHE Integration**: Combine with Contrast Limited Adaptive Histogram Equalization for local matching.
- **Color Channels**: Apply matching to all RGB channels or LAB’s L channel.
- **Quantitative Analysis**: Compute histogram similarity (e.g., Bhattacharyya distance) to evaluate matching quality.

