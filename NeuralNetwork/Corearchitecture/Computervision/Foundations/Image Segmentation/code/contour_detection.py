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

# 1. Preprocessing: Apply Gaussian blur and adaptive thresholding
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
img_binary = cv2.adaptiveThreshold(
    img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
)

# 2. Morphological Operation: Opening to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)

# 3. Contour Detection
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Draw Contours on Color Image
img_contours = img_color.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)  # Green contours

# 5. Analyze a Selected Contour (e.g., largest contour by area)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    img_largest_contour = img_color.copy()
    cv2.drawContours(img_largest_contour, [largest_contour], -1, (0, 0, 255), 2)  # Red contour

    # Compute properties
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    moments = cv2.moments(largest_contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = 0, 0

    # Draw centroid
    cv2.circle(img_largest_contour, (cx, cy), 5, (255, 0, 0), -1)  # Blue dot

# 6. Display Results
plt.figure(figsize=(15, 10))

# Original Grayscale Image
plt.subplot(2, 2, 1)
plt.title('Original Grayscale')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Binary Image
plt.subplot(2, 2, 2)
plt.title('Binary (Adaptive Threshold)')
plt.imshow(img_binary, cmap='gray')
plt.axis('off')

# All Contours
plt.subplot(2, 2, 3)
plt.title('All Contours')
plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Largest Contour with Centroid
if contours:
    plt.subplot(2, 2, 4)
    plt.title(f'Largest Contour\nArea: {area:.0f}, Perimeter: {perimeter:.0f}, Centroid: ({cx}, {cy})')
    plt.imshow(cv2.cvtColor(img_largest_contour, cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.tight_layout()
plt.show()

# 7. Example: Print contour properties
if contours:
    print(f"Largest Contour Properties:")
    print(f"Area: {area:.0f} pixels")
    print(f"Perimeter: {perimeter:.0f} pixels")
    print(f"Centroid: ({cx}, {cy})")
    print(f"Number of Contours: {len(contours)}")
