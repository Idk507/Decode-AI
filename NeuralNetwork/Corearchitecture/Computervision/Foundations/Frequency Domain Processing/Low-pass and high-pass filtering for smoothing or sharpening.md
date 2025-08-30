### Low-Pass and High-Pass Filtering for Smoothing or Sharpening

**Low-pass** and **high-pass filtering** are frequency domain techniques used to manipulate an image by selectively retaining or suppressing specific frequency components. These methods leverage the **Fourier Transform** to process images in the frequency domain, where low frequencies correspond to smooth regions and high frequencies correspond to sharp changes (e.g., edges, noise). Low-pass filtering smooths images by removing high-frequency components, while high-pass filtering sharpens images by emphasizing high-frequency components. This explanation covers the mathematical concepts, practical applications, and a detailed Python code implementation using OpenCV and NumPy to demonstrate low-pass and high-pass filtering.

---

### 1. What are Low-Pass and High-Pass Filtering?

- **Low-Pass Filtering**:
  - Definition: Retains low-frequency components (smooth regions, gradual intensity changes) while suppressing high-frequency components (edges, noise).
  - Effect: Smooths or blurs the image, reducing noise and fine details.
  - Applications: Noise reduction, image preprocessing, smoothing textures.
- **High-Pass Filtering**:
  - Definition: Retains high-frequency components (edges, sharp transitions) while suppressing low-frequency components (smooth areas).
  - Effect: Sharpens the image, enhancing edges and details.
  - Applications: Edge enhancement, feature detection, image sharpening.
- **Purpose**: Manipulate image frequency content to achieve desired visual or analytical effects.
- **Context**: These filters are applied in the frequency domain using the Fourier Transform, building on the frequency domain processing concepts previously discussed.

---

### 2. Mathematical Concepts

The **Fourier Transform** converts an image from the spatial domain to the frequency domain, where filtering is performed by modifying the frequency spectrum. Low-pass and high-pass filters are implemented using masks that selectively retain or suppress frequency components.

#### Discrete Fourier Transform (DFT)
For a 2D grayscale image $\( I(x, y) \) of size \( M \times N \)$, the DFT is:
<img width="446" height="107" alt="image" src="https://github.com/user-attachments/assets/843eda0c-06c7-4f6a-90b1-d5363780edbe" />

- $\( (u, v) \)$: Frequency coordinates.
- $\( F(u, v) \)$: Complex-valued spectrum (magnitude and phase).
- Centered using `np.fft.fftshift` so low frequencies are at the center.

#### Magnitude Spectrum
The magnitude $\( |F(u, v)| = \sqrt{\text{Re}(F(u, v))^2 + \text{Im}(F(u, v))^2} \)$ represents frequency strength, visualized as $\( \log(1 + |F(u, v)|) \)$.

#### Filtering Process
1. **Compute DFT**: Transform the image to the frequency domain.
2. **Apply Filter**:
   - **Low-Pass Filter**: Multiply $\( F(u, v) \)$ by a mask $\( H(u, v) \)$ that is 1 near the center (low frequencies) and 0 elsewhere.
   - **High-Pass Filter**: Multiply $\( F(u, v) \)$ by a mask $\( H(u, v) \)$ that is 1 at the edges (high frequencies) and 0 near the center.
3. **Inverse DFT**: Transform the filtered spectrum back to the spatial domain:
   <img width="628" height="107" alt="image" src="https://github.com/user-attachments/assets/cc266ede-fa79-4bcb-896d-4e762b4b89cd" />


#### Filter Masks
<img width="902" height="393" alt="image" src="https://github.com/user-attachments/assets/19bee6d4-c1c0-4901-be77-a388bd81d866" />

- **Practical Filters**:
  - Ideal filters cause ringing artifacts (Gibbs phenomenon) due to sharp cutoffs.
  - Gaussian filters (smooth transitions) are preferred:
    - Low-pass: $\( H(u, v) = e^{-\frac{(u - u_c)^2 + (v - v_c)^2}{2\sigma^2}} \)$.
    - High-pass: $\( H(u, v) = 1 - e^{-\frac{(u - u_c)^2 + (v - v_c)^2}{2\sigma^2}} \)$.

#### Example
Consider a 4x4 grayscale image:
<img width="379" height="163" alt="image" src="https://github.com/user-attachments/assets/d394d3f3-dba1-4687-933f-326ca71ada54" />

- **DFT**: Compute $\( F(u, v) \)$, shift to center low frequencies.
- **Low-Pass Filter**:
  - Apply a circular mask with radius $\( D_0 \)$, retaining low frequencies (e.g., average intensity).
  - Result: Smoothed image, reducing the sharp transition between 100 and 150.
- **High-Pass Filter**:
  - Apply the inverse mask, retaining high frequencies (e.g., edges between 100 and 150).
  - Result: Sharpened image, emphasizing edges.
- **Output**:
  - Low-pass: Blurred image with reduced edge contrast.
  - High-pass: Edge-enhanced image with suppressed smooth regions.

---

### 3. Practical Considerations
- **Preprocessing**:
  - Convert color images to grayscale or process individual channels (e.g., RGB, LAB).
  - Pad images to powers of 2 (e.g., 512x512) for faster FFT computation:
    ```python
    img = cv2.resize(img, (512, 512))
    ```
- **Filter Design**:
  - Use Gaussian filters to avoid ringing artifacts.
  - Adjust cutoff frequency $(\( D_0 \) or \( \sigma \))$ to control the degree of smoothing or sharpening.
- **Visualization**:
  - Display the log-magnitude spectrum to analyze frequency content.
  - Center the spectrum using `np.fft.fftshift` for intuitive interpretation.
- **Applications**:
  - **Low-Pass**: Noise reduction in medical imaging, smoothing textures in photography.
  - **High-Pass**: Edge enhancement for object detection, sharpening blurry images.
- **Limitations**:
  - Assumes image periodicity, which may cause boundary artifacts.
  - Computationally intensive for large images; use FFT for efficiency.
- **Color Images**:
  - Apply filters to each channel or convert to a single channel (e.g., luminance in LAB).

---

### 4. Code Implementation in Python

Below is a Python code example using OpenCV and NumPy to demonstrate low-pass and high-pass filtering. The code:
- Loads a grayscale image and computes its DFT.
- Applies Gaussian low-pass and high-pass filters.
- Visualizes the original image, magnitude spectrum, filter masks, and filtered images.

<xaiArtifact artifact_id="e556212d-00d7-4ed6-89a0-3f6511170cb8" artifact_version_id="42d78b2d-7e83-4317-9f9f-6b8e266a2029" title="frequency_filtering.py" contentType="text/python">
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

# 1. Compute Fourier Transform
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)  # Shift zero frequency to center

# 2. Compute Magnitude Spectrum for Visualization
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

# 3. Create Gaussian Low-Pass and High-Pass Filters
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
sigma = 30  # Controls filter smoothness

# Gaussian Low-Pass Filter
u, v = np.meshgrid(np.arange(-ccol, ccol), np.arange(-crow, crow))
d = np.sqrt(u**2 + v**2)
low_pass = np.exp(-(d**2) / (2 * sigma**2))

# Gaussian High-Pass Filter
high_pass = 1 - low_pass

# Apply Filters
dft_shift_low = dft_shift * low_pass
dft_shift_high = dft_shift * high_pass

# 4. Inverse Fourier Transform
# Low-Pass Filtered Image
dft_ishift_low = np.fft.ifftshift(dft_shift_low)
img_low = np.fft.ifft2(dft_ishift_low)
img_low = np.abs(img_low).astype(np.uint8)

# High-Pass Filtered Image
dft_ishift_high = np.fft.ifftshift(dft_shift_high)
img_high = np.fft.ifft2(dft_ishift_high)
img_high = np.abs(img_high).astype(np.uint8)

# 5. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Grayscale')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Magnitude Spectrum
plt.subplot(2, 3, 2)
plt.title('Magnitude Spectrum (Log)')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

# Low-Pass Filter Mask
plt.subplot(2, 3, 3)
plt.title('Gaussian Low-Pass Filter')
plt.imshow(low_pass, cmap='gray')
plt.axis('off')

# Low-Pass Filtered Image
plt.subplot(2, 3, 4)
plt.title('Low-Pass Filtered (Smoothed)')
plt.imshow(img_low, cmap='gray')
plt.axis('off')

# High-Pass Filter Mask
plt.subplot(2, 3, 5)
plt.title('Gaussian High-Pass Filter')
plt.imshow(high_pass, cmap='gray')
plt.axis('off')

# High-Pass Filtered Image
plt.subplot(2, 3, 6)
plt.title('High-Pass Filtered (Sharpened)')
plt.imshow(img_high, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original Grayscale: {img[y, x]}")
print(f"Low-Pass Filtered: {img_low[y, x]}")
print(f"High-Pass Filtered: {img_high[y, x]}")
print(f"Magnitude Spectrum (Log): {magnitude_spectrum[y, x]:.2f}")
</xaiArtifact>

---

### Code Explanation
1. **Loading Image**:
   - Loads a grayscale image using `cv2.imread` with `cv2.IMREAD_GRAYSCALE`.

2. **Fourier Transform**:
   - Computes the 2D DFT using `np.fft.fft2`.
   - Shifts the zero frequency to the center with `np.fft.fftshift`.

3. **Magnitude Spectrum**:
   - Computes the log-magnitude spectrum: $\( 20 \cdot \log(1 + |F(u, v)|) \)$.

4. **Filter Design**:
   - Creates a Gaussian low-pass filter: $\( H(u, v) = e^{-\frac{d^2}{2\sigma^2}} \)$, where $\( d \)$ is the distance from the center.
   - Creates a high-pass filter: $\( H(u, v) = 1 - \text{low-pass} \)$.
   - $\( \sigma = 30 \)$ controls the filter’s smoothness.

5. **Filtering**:
   - Applies the low-pass and high-pass filters by multiplying the shifted DFT.

6. **Inverse Fourier Transform**:
   - Shifts back the filtered DFTs using `np.fft.ifftshift`.
   - Computes the inverse DFT with `np.fft.ifft2` to reconstruct the filtered images.

7. **Visualization**:
   - Displays the original image, magnitude spectrum, low-pass filter, high-pass filter, and filtered images.

8. **Pixel Value Output**:
   - Prints pixel values at (100, 100) for the original, low-pass, high-pass, and magnitude spectrum.

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide an image file (`image.jpg`) in the same directory or update the `image_path` variable. Use a grayscale image with clear edges or noise for meaningful results.

---

### Expected Output
- A figure with six subplots:
  - Original grayscale image.
  - Log-magnitude spectrum (low frequencies at center, high at edges).
  - Gaussian low-pass filter (bright center).
  - Low-pass filtered image (smoothed, reduced edges/noise).
  - Gaussian high-pass filter (dark center).
  - High-pass filtered image (sharpened, enhanced edges).
- Printed pixel values, e.g.:
  ```
  Pixel at (100, 100):
  Original Grayscale: 113
  Low-Pass Filtered: 110
  High-Pass Filtered: 5
  Magnitude Spectrum (Log): 5.67
  ```

---

### Practical Notes
- **Filter Parameter $(\( \sigma \))$ **:
  - Smaller $\( \sigma \)$ : Stronger filtering (more smoothing or sharpening).
  - Larger $\( \sigma \)$: Weaker filtering, retaining more frequencies.
  - Adjust $\( \sigma \)$ (e.g., 10–50) based on image content.
- **Preprocessing**:
  - Gaussian blur in the spatial domain can complement low-pass filtering.
  - Convert color images to grayscale or process channels separately.
- **Applications**:
  - **Low-Pass**: Remove noise in medical imaging, smooth textures in photography.
  - **High-Pass**: Enhance edges for object detection, sharpen blurry images.
- **Limitations**:
  - Ideal filters (sharp cutoffs) cause ringing artifacts; Gaussian filters are preferred.
  - High-pass filtering may amplify noise; preprocess with noise reduction.
- **Color Images**:
  - Apply filters to each RGB channel or use a single channel (e.g., V in HSV).

---

### Mathematical Insights
- **Low-Pass Filter**: Retains low frequencies:
 <img width="742" height="334" alt="image" src="https://github.com/user-attachments/assets/0c82cd38-1732-41f1-9acf-d482cf707a24" />


---

### Extensions
- **Band-Pass/Band-Stop Filters**: Select specific frequency ranges for targeted effects.
- **Butterworth Filters**: Use for smoother transitions than ideal filters but sharper than Gaussian.
- **Color Image Processing**: Apply filters to LAB or HSV channels for better results.
- **Sharpening Enhancement**: Combine high-pass filtered image with the original:
  ```python
  img_sharpened = cv2.addWeighted(img, 1.0, img_high, 0.5, 0)
  ```
- **Frequency Analysis**: Extract dominant frequencies for texture or pattern recognition.

