### Image Restoration: Deblurring (Wiener Filter, Inverse Filtering)

**Deblurring** is an image restoration technique aimed at recovering a sharp image from a blurred version, often caused by motion, defocus, or optical imperfections. The **Wiener filter** and **inverse filtering** are frequency domain methods that attempt to reverse the blurring process by modeling the blur as a convolution. This explanation covers the mathematical concepts, practical applications, and a detailed Python code implementation using OpenCV and NumPy to demonstrate deblurring with Wiener and inverse filtering, building on previous frequency domain processing concepts.

---

### 1. What is Deblurring?

Deblurring seeks to restore an image degraded by a blurring process, typically modeled as a convolution of the original image with a **point spread function (PSF)** plus additive noise.

- **Blur Model**:
 <img width="740" height="314" alt="image" src="https://github.com/user-attachments/assets/ca88e88f-6f71-4f08-a27f-87989108c10d" />

- **Inverse Filtering**:
  - Directly inverts the blur by dividing the Fourier Transform of the blurred image by the PSF’s Fourier Transform.
  - Sensitive to noise, leading to amplification of high-frequency noise.
- **Wiener Filter**:
  - A more robust method that accounts for noise, minimizing the mean squared error between the restored and original images.
  - Balances deblurring and noise suppression.
- **Purpose**:
  - Recover sharp details for improved visual quality or analysis.
  - Enhance performance of tasks like object detection or recognition.
- **Applications**:
  - Photography (e.g., correcting motion blur in handheld shots).
  - Medical imaging (e.g., sharpening MRI or ultrasound images).
  - Astronomy (e.g., correcting atmospheric blur in telescope images).
  - Forensics (e.g., enhancing license plate images).

---

### 2. Mathematical Concepts

Deblurring is typically performed in the **frequency domain** using the Fourier Transform, as convolution in the spatial domain becomes multiplication in the frequency domain.

#### Convolution Theorem
<img width="757" height="196" alt="image" src="https://github.com/user-attachments/assets/3a9e089f-b4a6-4123-9df8-22276387c28c" />

#### Inverse Filtering
<img width="780" height="369" alt="image" src="https://github.com/user-attachments/assets/7fc68dca-5fbb-471b-adc9-89a92da74500" />


#### Wiener Filter
-<img width="1021" height="605" alt="image" src="https://github.com/user-attachments/assets/abe3f1f6-6223-4584-bea1-675e2dff9b11" />


#### Example
Consider a 3x3 image blurred with a 3x3 Gaussian PSF and additive noise:
<img width="531" height="109" alt="image" src="https://github.com/user-attachments/assets/5cab0f8e-0b23-4b8c-8940-12060c22a0de" />

- Blurred image: $\( g = I * h + n \)$.
- **Inverse Filtering**:
  - Compute $\( G(u, v) \)$ and $\( H(u, v) \)$ (Fourier Transforms).
  - Estimate: $\( \hat{F}(u, v) = \frac{G(u, v)}{H(u, v)} \)$.
  - Noise amplification occurs where \( H(u, v) \approx 0 \).
- **Wiener Filter**:
  - Compute: $\( \hat{F}(u, v) = \frac{H^*(u, v) \cdot G(u, v)}{|H(u, v)|^2 + K} \)$.
  - $\( K = 0.01 \) prevents division by small \( H(u, v) \)$, reducing noise.
- Output: Restored image $\( \hat{f}(x, y) \)$ approximates the original $\( I \)$, with Wiener filter preserving more details in noisy conditions.

---

### 3. Practical Considerations
- **PSF Estimation**:
  - The PSF $\( h(x, y) \)$ must be known or estimated (e.g., Gaussian for defocus, linear for motion blur).
  - Incorrect PSF leads to poor deblurring results.
- **Noise Handling**:
  - Inverse filtering fails with significant noise; Wiener filter is preferred.
  - Tune $\( K \)$ in Wiener filter (e.g., 0.001–0.1) based on noise level.
- **Preprocessing**:
  - Convert color images to grayscale or process channels separately.
  - Apply noise reduction (e.g., Gaussian blur) before deblurring if noise is high.
- **Applications**:
  - **Inverse Filtering**: Best for low-noise images with known PSF (e.g., controlled lab settings).
  - **Wiener Filter**: General-purpose for noisy images (e.g., real-world photography, medical imaging).
- **Limitations**:
  - Requires accurate PSF; blind deblurring (unknown PSF) is more complex.
  - High noise levels degrade performance, especially for inverse filtering.
  - Computationally intensive due to Fourier Transforms.
- **Performance**:
  - Use Fast Fourier Transform (FFT) for efficiency.
  - Pad images to powers of 2 (e.g., 512x512) for faster computation.

---

### 4. Code Implementation in Python

Below is a Python code example using OpenCV and NumPy to demonstrate deblurring with inverse and Wiener filtering. The code:
- Loads a grayscale image and applies a synthetic Gaussian blur with noise.
- Implements inverse filtering and Wiener filtering in the frequency domain.
- Visualizes the original, blurred, and restored images, along with the magnitude spectrum.

<xaiArtifact artifact_id="d3bcd790-58d2-459c-aa41-f178839d0b55" artifact_version_id="ed1a5ca3-edd4-4dfb-9f3c-a7ac3408c92f" title="deblurring.py" contentType="text/python">
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to add Gaussian blur and noise
def add_blur_and_noise(img, kernel_size=5, sigma=1.5, noise_sigma=10):
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    noisy_blurred = blurred + noise
    noisy_blurred = np.clip(noisy_blurred, 0, 255).astype(np.uint8)
    return noisy_blurred, cv2.getGaussianKernel(kernel_size, sigma) * cv2.getGaussianKernel(kernel_size, sigma).T

# Load an image
image_path = 'image.jpg'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# 1. Add Gaussian Blur and Noise
img_blurred, psf = add_blur_and_noise(img, kernel_size=5, sigma=1.5, noise_sigma=10)

# 2. Compute Fourier Transforms
# Pad PSF to match image size
rows, cols = img.shape
psf_padded = np.zeros((rows, cols), dtype=np.float32)
k_size = psf.shape[0]
k_half = k_size // 2
psf_padded[rows//2 - k_half:rows//2 + k_half + 1, cols//2 - k_half:cols//2 + k_half + 1] = psf
psf_padded /= psf_padded.sum()  # Normalize PSF

# Compute DFT
dft_img = np.fft.fft2(img_blurred)
dft_psf = np.fft.fft2(psf_padded)
dft_shift = np.fft.fftshift(dft_img)
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

# 3. Inverse Filtering
H = dft_psf
epsilon = 1e-8  # Avoid division by zero
inverse_filter = np.divide(1, H, where=np.abs(H) > epsilon, out=np.zeros_like(H, dtype=np.complex64))
dft_inverse = dft_img * inverse_filter
img_inverse = np.abs(np.fft.ifft2(dft_inverse)).astype(np.uint8)

# 4. Wiener Filtering
K = 0.01  # Noise-to-signal ratio
H_conj = np.conj(H)
wiener_filter = H_conj / (np.abs(H)**2 + K)
dft_wiener = dft_img * wiener_filter
img_wiener = np.abs(np.fft.ifft2(dft_wiener)).astype(np.uint8)

# 5. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Grayscale')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Blurred + Noisy Image
plt.subplot(2, 3, 2)
plt.title('Blurred + Noisy')
plt.imshow(img_blurred, cmap='gray')
plt.axis('off')

# Magnitude Spectrum
plt.subplot(2, 3, 3)
plt.title('Magnitude Spectrum (Log)')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

# PSF
plt.subplot(2, 3, 4)
plt.title('Point Spread Function')
plt.imshow(psf_padded, cmap='gray')
plt.axis('off')

# Inverse Filtered
plt.subplot(2, 3, 5)
plt.title('Inverse Filtering')
plt.imshow(img_inverse, cmap='gray')
plt.axis('off')

# Wiener Filtered
plt.subplot(2, 3, 6)
plt.title('Wiener Filtering (K=0.01)')
plt.imshow(img_wiener, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original Grayscale: {img[y, x]}")
print(f"Blurred + Noisy: {img_blurred[y, x]}")
print(f"Inverse Filtered: {img_inverse[y, x]}")
print(f"Wiener Filtered: {img_wiener[y, x]}")
</xaiArtifact>

---

### Code Explanation
1. **Loading Image**:
   - Loads a grayscale image using `cv2.imread` with `cv2.IMREAD_GRAYSCALE`.

2. **Adding Blur and Noise**:
   - Applies a 5x5 Gaussian blur $(\( \sigma = 1.5 \))$ and Gaussian noise $(\( \sigma = 10 \))$.
   - Generates the PSF (Gaussian kernel) for deblurring.

3. **Fourier Transforms**:
   - Pads the PSF to match the image size and normalizes it.
   - Computes the DFT of the blurred image and PSF using `np.fft.fft2`.

4. **Inverse Filtering**:
   - Computes $\( \hat{F} = \frac{G}{H} \)$, with a small $\( \epsilon \)$ to avoid division by zero.
   - Applies the inverse DFT to reconstruct the image.

5. **Wiener Filtering**:
   - Computes the Wiener filter: $\( \frac{H^*}{|H|^2 + K} \)$, with $\( K = 0.01 \)$.
   - Applies the filter to the DFT and reconstructs the image.

6. **Visualization**:
   - Displays the original image, blurred+noisy image, magnitude spectrum, PSF, inverse-filtered image, and Wiener-filtered image.

7. **Pixel Value Output**:
   - Prints pixel values at (100, 100) for all images.

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide an image file (`image.jpg`) in the same directory or update the `image_path` variable. Use a grayscale image with clear details for meaningful results.

---

### Expected Output
- A figure with six subplots:
  - Original grayscale image.
  - Blurred and noisy image (Gaussian blur + noise).
  - Log-magnitude spectrum of the blurred image.
  - PSF (padded Gaussian kernel).
  - Inverse-filtered image (may show noise amplification).
  - Wiener-filtered image (sharper, less noisy).
- Printed pixel values, e.g.:
  ```
  Pixel at (100, 100):
  Original Grayscale: 113
  Blurred + Noisy: 110
  Inverse Filtered: 150
  Wiener Filtered: 112
  ```

---

### Practical Notes
- **PSF**:
  - Must be known or estimated (e.g., Gaussian for defocus, linear for motion blur).
  - Incorrect PSF leads to poor restoration.
- **Wiener Filter Parameter $(\( K \))$**:
  - Small $\( K \)$ (e.g., 0.001): More aggressive deblurring, but noise-sensitive.
  - Large $\( K \)$ (e.g., 0.1): More noise suppression, but less sharpening.
  - Tune empirically based on noise level.
- **Inverse Filtering**:
  - Works well for low-noise images but fails with significant noise.
  - Avoid division by zero using a small $\( \epsilon \)$.
- **Preprocessing**:
  - Apply noise reduction (e.g., Gaussian blur) before deblurring if noise is high.
  - Use grayscale or process color channels separately.
- **Applications**:
  - **Inverse Filtering**: Controlled settings with known PSF and low noise.
  - **Wiener Filter**: Real-world images with noise (e.g., photography, medical imaging).
- **Limitations**:
  - Requires accurate PSF; blind deblurring is more complex.
  - Inverse filtering amplifies noise; Wiener filter is preferred.
  - Computationally intensive due to FFT.

---

### Mathematical Insights
<img width="778" height="466" alt="image" src="https://github.com/user-attachments/assets/1c7934d4-92e2-48aa-8475-1ff9df8cc37f" />

---

### Extensions
- **Blind Deblurring**: Estimate PSF using methods like Richardson-Lucy or deep learning.
- **Motion Blur**: Use a linear PSF for motion deblurring.
- **Color Images**: Apply deblurring to each RGB channel or use LAB/HSV.
- **Advanced Methods**: Explore total variation or non-local means deblurring.
- **Evaluation**: Use PSNR or SSIM to quantify restoration quality.

