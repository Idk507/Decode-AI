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
