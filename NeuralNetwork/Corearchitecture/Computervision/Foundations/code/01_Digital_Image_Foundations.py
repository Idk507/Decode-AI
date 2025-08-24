import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image_path = 'image.jpg'  # Replace with your image path
img = cv2.imread(image_path)  # Reads image in BGR format

# Check if image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# 1. Pixel Access and Grayscale Conversion
def convert_to_grayscale(image):
    height, width, _ = image.shape
    grayscale = np.zeros((height, width), dtype=np.uint8)
    
    # Manual grayscale conversion: Luminance = 0.299R + 0.587G + 0.114B
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]  # BGR format in OpenCV
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            grayscale[i, j] = gray_value
    return grayscale

# Convert image to grayscale
gray_img = convert_to_grayscale(img)

# 2. RGB to HSV Conversion
# OpenCV uses BGR, so convert BGR to RGB first for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert RGB to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split HSV channels
h, s, v = cv2.split(img_hsv)

# 3. Display Results
plt.figure(figsize=(15, 10))

# Original Image (RGB)
plt.subplot(2, 3, 1)
plt.title('Original Image (RGB)')
plt.imshow(img_rgb)
plt.axis('off')

# Grayscale Image
plt.subplot(2, 3, 2)
plt.title('Grayscale Image')
plt.imshow(gray_img, cmap='gray')
plt.axis('off')

# HSV Channels
plt.subplot(2, 3, 3)
plt.title('Hue Channel')
plt.imshow(h, cmap='hsv')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Saturation Channel')
plt.imshow(s, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Value Channel')
plt.imshow(v, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 4. Example: Print pixel values at a specific coordinate (e.g., (100, 100))
x, y = 100, 100
bgr_pixel = img[y, x]  # BGR values
rgb_pixel = img_rgb[y, x]  # RGB values
hsv_pixel = img_hsv[y, x]  # HSV values
gray_pixel = gray_img[y, x]  # Grayscale value

print(f"Pixel at ({x}, {y}):")
print(f"BGR: {bgr_pixel} (B={bgr_pixel[0]}, G={bgr_pixel[1]}, R={bgr_pixel[2]})")
print(f"RGB: {rgb_pixel} (R={rgb_pixel[0]}, G={rgb_pixel[1]}, B={rgb_pixel[2]})")
print(f"HSV: {hsv_pixel} (H={hsv_pixel[0]}, S={hsv_pixel[1]}, V={hsv_pixel[2]})")
print(f"Grayscale: {gray_pixel}")
