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
