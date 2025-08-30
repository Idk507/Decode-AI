import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to add Gaussian noise
def add_gaussian_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy_img = img + gaussian_noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

# Function to add salt-and-pepper noise
def add_salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
    noisy_img = img.copy()
    total_pixels = img.size
    # Salt noise (white pixels)
    num_salt = int(total_pixels * salt_prob)
    coords = [np.random.randint(0, i, num_salt) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 255
    # Pepper noise (black pixels)
    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 0
    return noisy_img

# Load an image
image_path = 'image.jpg'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# 1. Add Noise
img_gaussian_noise = add_gaussian_noise(img, mean=0, sigma=25)
img_salt_pepper_noise = add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02)

# 2. Noise Reduction
# Gaussian Blur for Gaussian Noise
img_gaussian_filtered = cv2.GaussianBlur(img_gaussian_noise, (5, 5), 1.5)

# Median Blur for Salt-and-Pepper Noise
img_sp_filtered = cv2.medianBlur(img_salt_pepper_noise, 5)

# 3. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Grayscale')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Gaussian Noise
plt.subplot(2, 3, 2)
plt.title('Gaussian Noise (σ=25)')
plt.imshow(img_gaussian_noise, cmap='gray')
plt.axis('off')

# Gaussian Filtered
plt.subplot(2, 3, 3)
plt.title('Gaussian Blur (5x5, σ=1.5)')
plt.imshow(img_gaussian_filtered, cmap='gray')
plt.axis('off')

# Salt-and-Pepper Noise
plt.subplot(2, 3, 4)
plt.title('Salt-and-Pepper Noise (2%)')
plt.imshow(img_salt_pepper_noise, cmap='gray')
plt.axis('off')

# Median Filtered
plt.subplot(2, 3, 5)
plt.title('Median Blur (5x5)')
plt.imshow(img_sp_filtered, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 4. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original Grayscale: {img[y, x]}")
print(f"Gaussian Noise: {img_gaussian_noise[y, x]}")
print(f"Gaussian Filtered: {img_gaussian_filtered[y, x]}")
print(f"Salt-and-Pepper Noise: {img_salt_pepper_noise[y, x]}")
print(f"Median Filtered: {img_sp_filtered[y, x]}")
