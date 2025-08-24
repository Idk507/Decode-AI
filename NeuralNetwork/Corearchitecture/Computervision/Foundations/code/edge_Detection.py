import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (grayscale)
image_path = 'image.jpg'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# 1. Manual Sobel Edge Detection
def sobel_edge_detection(image):
    # Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float32)
    
    # Apply Sobel filters
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    
    # Compute gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = np.clip(grad_magnitude, 0, 255).astype(np.uint8)
    
    # Compute gradient direction (optional, for visualization)
    grad_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    
    return grad_x, grad_y, grad_magnitude, grad_direction

# 2. Apply Sobel Edge Detection
sobel_x, sobel_y, sobel_magnitude, sobel_direction = sobel_edge_detection(img)

# 3. Apply OpenCV Sobel (for comparison)
sobel_x_cv = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_cv = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude_cv = np.sqrt(sobel_x_cv**2 + sobel_y_cv**2)
sobel_magnitude_cv = np.clip(sobel_magnitude_cv, 0, 255).astype(np.uint8)

# 4. Apply Canny Edge Detection
canny_edges = cv2.Canny(img, threshold1=100, threshold2=200)

# 5. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Sobel X (Horizontal Edges)
plt.subplot(2, 3, 2)
plt.title('Sobel X (Horizontal)')
plt.imshow(np.abs(sobel_x), cmap='gray')
plt.axis('off')

# Sobel Y (Vertical Edges)
plt.subplot(2, 3, 3)
plt.title('Sobel Y (Vertical)')
plt.imshow(np.abs(sobel_y), cmap='gray')
plt.axis('off')

# Sobel Magnitude
plt.subplot(2, 3, 4)
plt.title('Sobel Magnitude')
plt.imshow(sobel_magnitude, cmap='gray')
plt.axis('off')

# OpenCV Sobel Magnitude
plt.subplot(2, 3, 5)
plt.title('OpenCV Sobel Magnitude')
plt.imshow(sobel_magnitude_cv, cmap='gray')
plt.axis('off')

# Canny Edges
plt.subplot(2, 3, 6)
plt.title('Canny Edges')
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original: {img[y, x]}")
print(f"Sobel X: {sobel_x[y, x]:.2f}")
print(f"Sobel Y: {sobel_y[y, x]:.2f}")
print(f"Sobel Magnitude: {sobel_magnitude[y, x]}")
print(f"Canny: {canny_edges[y, x]}")
