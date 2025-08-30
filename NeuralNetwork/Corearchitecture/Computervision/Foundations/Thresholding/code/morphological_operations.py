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

# 1. Preprocessing: Apply Otsu's thresholding to create a binary image
_, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 2. Define Structuring Element (3x3 cross)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# Alternative: Rectangular kernel
# kernel = np.ones((3, 3), np.uint8)

# 3. Dilation
img_dilated = cv2.dilate(img_binary, kernel, iterations=1)

# 4. Erosion
img_eroded = cv2.erode(img_binary, kernel, iterations=1)

# 5. Display Results
plt.figure(figsize=(15, 10))

# Original Grayscale Image
plt.subplot(2, 2, 1)
plt.title('Original Grayscale')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Binary Image (Otsu's)
plt.subplot(2, 2, 2)
plt.title('Binary (Otsu)')
plt.imshow(img_binary, cmap='gray')
plt.axis('off')

# Dilated Image
plt.subplot(2, 2, 3)
plt.title('Dilated (3x3 Cross)')
plt.imshow(img_dilated, cmap='gray')
plt.axis('off')

# Eroded Image
plt.subplot(2, 2, 4)
plt.title('Eroded (3x3 Cross)')
plt.imshow(img_eroded, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original Grayscale: {img[y, x]}")
print(f"Binary (Otsu): {img_binary[y, x]}")
print(f"Dilated: {img_dilated[y, x]}")
print(f"Eroded: {img_eroded[y, x]}")
