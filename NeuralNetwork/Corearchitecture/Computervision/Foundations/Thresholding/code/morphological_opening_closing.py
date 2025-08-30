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

# 1. Preprocessing: Apply Gaussian blur and adaptive thresholding
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
img_binary = cv2.adaptiveThreshold(
    img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
)

# 2. Define Structuring Element (3x3 cross)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# Alternative: Rectangular kernel
# kernel = np.ones((3, 3), np.uint8)

# 3. Opening (Erosion followed by Dilation)
img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)

# 4. Closing (Dilation followed by Erosion)
img_close = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

# 5. Display Results
plt.figure(figsize=(15, 10))

# Original Grayscale Image
plt.subplot(2, 2, 1)
plt.title('Original Grayscale')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Binary Image (Adaptive Thresholding)
plt.subplot(2, 2, 2)
plt.title('Binary (Adaptive Threshold)')
plt.imshow(img_binary, cmap='gray')
plt.axis('off')

# Opened Image
plt.subplot(2, 2, 3)
plt.title('Opening (3x3 Cross)')
plt.imshow(img_open, cmap='gray')
plt.axis('off')

# Closed Image
plt.subplot(2, 2, 4)
plt.title('Closing (3x3 Cross)')
plt.imshow(img_close, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original Grayscale: {img[y, x]}")
print(f"Binary (Adaptive): {img_binary[y, x]}")
print(f"Opening: {img_open[y, x]}")
print(f"Closing: {img_close[y, x]}")
