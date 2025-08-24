import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (grayscale for simplicity)
image_path = 'image.jpg'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Check if image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# 1. Manual Mean Filter (3x3)
def apply_mean_filter(image):
    height, width = image.shape
    output = np.zeros_like(image, dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.float32) / 9  # 3x3 mean kernel
    
    # Pad the image to handle borders
    padded_img = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    
    # Apply convolution
    for i in range(height):
        for j in range(width):
            # Extract 3x3 neighborhood
            neighborhood = padded_img[i:i+3, j:j+3]
            # Compute weighted sum
            output[i, j] = np.sum(neighborhood * kernel)
    
    return output.astype(np.uint8)

# 2. Apply Filters
# Manual Mean Filter
mean_filtered = apply_mean_filter(img)

# OpenCV Mean (Box) Filter
box_filtered = cv2.boxFilter(img, -1, (3, 3))

# Gaussian Filter (3x3, sigma=1)
gaussian_filtered = cv2.GaussianBlur(img, (3, 3), sigmaX=1)

# Median Filter (3x3)
median_filtered = cv2.medianBlur(img, 3)

# Sharpening Filter
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=np.float32)
sharpened = cv2.filter2D(img, -1, sharpening_kernel)

# 3. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Manual Mean Filter
plt.subplot(2, 3, 2)
plt.title('Manual Mean Filter (3x3)')
plt.imshow(mean_filtered, cmap='gray')
plt.axis('off')

# OpenCV Box Filter
plt.subplot(2, 3, 3)
plt.title('OpenCV Box Filter (3x3)')
plt.imshow(box_filtered, cmap='gray')
plt.axis('off')

# Gaussian Filter
plt.subplot(2, 3, 4)
plt.title('Gaussian Filter (3x3)')
plt.imshow(gaussian_filtered, cmap='gray')
plt.axis('off')

# Median Filter
plt.subplot(2, 3, 5)
plt.title('Median Filter (3x3)')
plt.imshow(median_filtered, cmap='gray')
plt.axis('off')

# Sharpening Filter
plt.subplot(2, 3, 6)
plt.title('Sharpening Filter')
plt.imshow(sharpened, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 4. Example: Print pixel values before and after filtering (at (100, 100))
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original: {img[y, x]}")
print(f"Manual Mean Filter: {mean_filtered[y, x]}")
print(f"OpenCV Box Filter: {box_filtered[y, x]}")
print(f"Gaussian Filter: {gaussian_filtered[y, x]}")
print(f"Median Filter: {median_filtered[y, x]}")
print(f"Sharpened: {sharpened[y, x]}")
