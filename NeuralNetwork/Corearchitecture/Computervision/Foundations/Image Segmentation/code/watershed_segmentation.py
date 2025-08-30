import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image_path = 'image.jpg'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(image_path)

# Check if image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# 1. Preprocessing: Apply Gaussian blur and thresholding
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
_, img_binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 2. Morphological Operations to Clean Noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=2)

# 3. Distance Transform to Find Object Centers
dist_transform = cv2.distanceTransform(img_open, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# 4. Identify Background and Unknown Regions
sure_bg = cv2.dilate(img_open, kernel, iterations=3)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 5. Marker Labeling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1  # Add 1 to distinguish from background (0)
markers[unknown == 255] = 0  # Mark unknown regions as 0

# 6. Apply Watershed Algorithm
markers = cv2.watershed(img_color, markers)
img_color[markers == -1] = [0, 0, 255]  # Mark watershed lines in red

# 7. Display Results
plt.figure(figsize=(15, 10))

# Original Grayscale Image
plt.subplot(2, 3, 1)
plt.title('Original Grayscale')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Binary Image (Otsu's)
plt.subplot(2, 3, 2)
plt.title('Binary (Otsu)')
plt.imshow(img_binary, cmap='gray')
plt.axis('off')

# Distance Transform
plt.subplot(2, 3, 3)
plt.title('Distance Transform')
plt.imshow(dist_transform, cmap='gray')
plt.axis('off')

# Sure Foreground
plt.subplot(2, 3, 4)
plt.title('Sure Foreground')
plt.imshow(sure_fg, cmap='gray')
plt.axis('off')

# Sure Background and Unknown
plt.subplot(2, 3, 5)
plt.title('Unknown Regions')
plt.imshow(unknown, cmap='gray')
plt.axis('off')

# Watershed Result
plt.subplot(2, 3, 6)
plt.title('Watershed Segmentation')
plt.imshow(img_color)
plt.axis('off')

plt.tight_layout()
plt.show()

# 8. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original Grayscale: {img[y, x]}")
print(f"Binary (Otsu): {img_binary[y, x]}")
print(f"Distance Transform: {dist_transform[y, x]:.2f}")
print(f"Markers: {markers[y, x]}")
print(f"Watershed Result (RGB): {img_color[y, x]}")
