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
height, width = img.shape[:2]

# 1. Translation
def apply_translation(image, tx, ty):
    translation_matrix = np.float32([[1, 0, tx],
                                    [0, 1, ty]])
    return cv2.warpAffine(image, translation_matrix, (width, height))

# 2. Rotation
def apply_rotation(image, angle, center=None, scale=1.0):
    if center is None:
        center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

# 3. Scaling
def apply_scaling(image, sx, sy):
    scaling_matrix = np.float32([[sx, 0, 0],
                                [0, sy, 0]])
    return cv2.warpAffine(image, scaling_matrix, (int(width * sx), int(height * sy)))

# 4. Shearing
def apply_shearing(image, kx, ky):
    shear_matrix = np.float32([[1, kx, 0],
                               [ky, 1, 0]])
    return cv2.warpAffine(image, shear_matrix, (width, height))

# 5. Affine Transformation
def apply_affine(image, src_points, dst_points):
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    return cv2.warpAffine(image, affine_matrix, (width, height))

# 6. Perspective Transformation
def apply_perspective(image, src_points, dst_points):
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, perspective_matrix, (width, height))

# Apply Transformations
# Translation: Move 50 pixels right, 30 pixels down
translated = apply_translation(img_rgb, tx=50, ty=30)

# Rotation: Rotate 45 degrees around center
rotated = apply_rotation(img_rgb, angle=45)

# Scaling: Double the size
scaled = apply_scaling(img_rgb, sx=2, sy=2)

# Shearing: Horizontal shear with kx=0.5
sheared = apply_shearing(img_rgb, kx=0.5, ky=0)

# Affine: Define three points for transformation
src_points = np.float32([[50, 50], [200, 50], [50, 200]])
dst_points = np.float32([[10, 100], [200, 50], [100, 250]])
affined = apply_affine(img_rgb, src_points, dst_points)

# Perspective: Define four points for transformation
src_points_p = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
dst_points_p = np.float32([[0, 0], [width-1, 100], [0, height-1], [width-1, height-100]])
perspectived = apply_perspective(img_rgb, src_points_p, dst_points_p)

# Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

# Translated Image
plt.subplot(2, 3, 2)
plt.title('Translated (50, 30)')
plt.imshow(translated)
plt.axis('off')

# Rotated Image
plt.subplot(2, 3, 3)
plt.title('Rotated 45°')
plt.imshow(rotated)
plt.axis('off')

# Scaled Image
plt.subplot(2, 3, 4)
plt.title('Scaled 2x')
plt.imshow(scaled)
plt.axis('off')

# Sheared Image
plt.subplot(2, 3, 5)
plt.title('Sheared (kx=0.5)')
plt.imshow(sheared)
plt.axis('off')

# Affine Transformed Image
plt.subplot(2, 3, 6)
plt.title('Affine Transformation')
plt.imshow(affined)
plt.axis('off')

# Perspective Transformed Image (in a separate figure due to space)
plt.figure(figsize=(5, 5))
plt.title('Perspective Transformation')
plt.imshow(perspectived)
plt.axis('off')

plt.tight_layout()
plt.show()

# Example: Print transformation matrix for rotation
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
print("Rotation Matrix (45°):")
print(rotation_matrix)
