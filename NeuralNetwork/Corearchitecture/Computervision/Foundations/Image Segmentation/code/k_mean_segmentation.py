import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image_path = 'image.jpg'  # Replace with your image path
img = cv2.imread(image_path)

# Check if image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Optional: Convert to LAB color space for better perceptual uniformity
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Reshape image to a 2D array of pixels (feature vectors)
pixels = img_lab.reshape(-1, 3).astype(np.float32)

# Define K-means parameters
K = 3  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
attempts = 10
flags = cv2.KMEANS_PP_CENTERS

# Apply K-means clustering
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, attempts, flags)

# Convert centers to uint8 and back to original color space
centers = np.uint8(centers)
centers_rgb = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)

# Create segmented image by assigning each pixel its cluster centroid color
segmented = centers[labels.flatten()]
segmented = segmented.reshape(img.shape)

# 5. Display Results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

# Segmented Image
plt.subplot(1, 3, 2)
plt.title(f'K-Means Segmentation (K={K})')
plt.imshow(segmented)
plt.axis('off')

# Cluster Centroids (Color Bar)
plt.subplot(1, 3, 3)
plt.title('Cluster Centroids')
color_bar = np.zeros((100, 300, 3), dtype=np.uint8)
width = 300 // K
for i in range(K):
    color_bar[:, i*width:(i+1)*width] = centers_rgb[i]
plt.imshow(color_bar)
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values and labels at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original RGB: {img_rgb[y, x]}")
print(f"Segmented RGB: {segmented[y, x]}")
print(f"Cluster Label: {labels[y * img.shape[1] + x]}")
print(f"Centroid Colors (RGB): {centers_rgb}")
