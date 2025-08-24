import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image_path = 'image.jpg'  # Replace with your image path
img_color = cv2.imread(image_path)

# Check if image is loaded successfully
if img_color is None:
    print("Error: Could not load image.")
    exit()

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# Convert to grayscale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 1. Histogram Equalization (Grayscale)
img_gray_eq = cv2.equalizeHist(img_gray)

# 2. Histogram Equalization (Color - HSV)
# Convert to HSV
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
# Split HSV channels
h, s, v = cv2.split(img_hsv)
# Equalize the Value channel
v_eq = cv2.equalizeHist(v)
# Merge back and convert to RGB
img_hsv_eq = cv2.merge([h, s, v_eq])
img_rgb_eq = cv2.cvtColor(img_hsv_eq, cv2.COLOR_HSV2RGB)

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

# Compute histograms for original and equalized images
hist_gray, title_gray = compute_histogram(img_gray, 'Grayscale Histogram')
hist_gray_eq, title_gray_eq = compute_histogram(img_gray_eq, 'Equalized Grayscale Histogram')
hist_rgb, title_rgb = compute_histogram(img_rgb, 'Color Histogram')
hist_rgb_eq, title_rgb_eq = compute_histogram(img_rgb_eq, 'Equalized Color Histogram (HSV)')

# 4. Display Results
plt.figure(figsize=(15, 10))

# Original Grayscale Image
plt.subplot(2, 3, 1)
plt.title('Original Grayscale')
plt.imshow(img_gray, cmap='gray')
plt.axis('off')

# Equalized Grayscale Image
plt.subplot(2, 3, 2)
plt.title('Equalized Grayscale')
plt.imshow(img_gray_eq, cmap='gray')
plt.axis('off')

# Original Color Image
plt.subplot(2, 3, 4)
plt.title('Original Color (RGB)')
plt.imshow(img_rgb)
plt.axis('off')

# Equalized Color Image
plt.subplot(2, 3, 5)
plt.title('Equalized Color (HSV)')
plt.imshow(img_rgb_eq)
plt.axis('off')

# Plot Histograms
plt.subplot(2, 3, 3)
plt.title(title_gray)
for hist, color in hist_gray:
    plt.plot(hist, color=color)
plt.xlim([0, 256])

plt.subplot(2, 3, 6)
plt.title(title_gray_eq)
for hist, color in hist_gray_eq:
    plt.plot(hist, color=color)
plt.xlim([0, 256])

# Note: Color histograms are complex to display in a single plot, so we skip them for simplicity
# Uncomment the following to plot RGB histograms if desired
"""
plt.subplot(2, 3, 3)
plt.title(title_rgb)
for hist, color in hist_rgb:
    plt.plot(hist, color=color)
plt.xlim([0, 256])

plt.subplot(2, 3, 6)
plt.title(title_rgb_eq)
for hist, color in hist_rgb_eq:
    plt.plot(hist, color=color)
plt.xlim([0, 256])
"""

plt.tight_layout()
plt.show()

# 5. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original Grayscale: {img_gray[y, x]}")
print(f"Equalized Grayscale: {img_gray_eq[y, x]}")
print(f"Original Color (RGB): {img_rgb[y, x]}")
print(f"Equalized Color (RGB): {img_rgb_eq[y, x]}")
