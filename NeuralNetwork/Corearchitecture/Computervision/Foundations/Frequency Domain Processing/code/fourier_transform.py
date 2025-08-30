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
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)  # Log scale for visibility

# 3. Low-Pass Filter: Create a circular mask
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
radius = 30  # Radius of low-pass filter
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), radius, 1, -1)  # Circular mask
dft_shift_filtered = dft_shift * mask

# 4. Inverse Fourier Transform
dft_ishift = np.fft.ifftshift(dft_shift_filtered)
img_filtered = np.fft.ifft2(dft_ishift)
img_filtered = np.abs(img_filtered).astype(np.uint8)

# 5. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.title('Original Grayscale')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Magnitude Spectrum
plt.subplot(2, 2, 2)
plt.title('Magnitude Spectrum (Log)')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

# Low-Pass Filter Mask
plt.subplot(2, 2, 3)
plt.title('Low-Pass Filter Mask')
plt.imshow(mask, cmap='gray')
plt.axis('off')

# Filtered Image
plt.subplot(2, 2, 4)
plt.title('Low-Pass Filtered Image')
plt.imshow(img_filtered, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original Grayscale: {img[y, x]}")
print(f"Filtered Grayscale: {img_filtered[y, x]}")
print(f"Magnitude Spectrum (Log): {magnitude_spectrum[y, x]:.2f}")
