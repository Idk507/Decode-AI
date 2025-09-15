### Geometric Transformations: Scaling, Rotation, Translation, and Shearing

**Geometric transformations** modify the spatial arrangement of pixels in an image, altering its size, orientation, or shape without changing the pixel intensities (beyond interpolation effects). These transformations include **scaling** (resizing), **rotation** (changing orientation), **translation** (shifting position), and **shearing** (slanting). They are fundamental in image processing for tasks like image alignment, augmentation, or correction. This explanation covers the mathematical concepts, practical applications, and a detailed Python code implementation using OpenCV and NumPy to demonstrate these transformations.

---

### 1. What are Geometric Transformations?

Geometric transformations map each pixel in the input image to a new position in the output image using a transformation matrix. They are typically applied in the **spatial domain** and involve interpolation to handle non-integer pixel coordinates.

- **Scaling**: Changes the size of the image (enlarging or shrinking).
- **Rotation**: Rotates the image around a point (usually the center) by an angle.
- **Translation**: Shifts the image along the x or y axis.
- **Shearing**: Slants the image along one axis, creating a parallelogram effect.
- **Purpose**:
  - Align images (e.g., registration in medical imaging).
  - Correct distortions (e.g., perspective correction in photography).
  - Augment data (e.g., for machine learning).
- **Applications**:
  - Image registration (e.g., aligning satellite images).
  - Computer vision (e.g., correcting camera distortions).
  - Graphics and animation (e.g., transforming objects).
  - Data augmentation for deep learning.

---

### 2. Mathematical Concepts

Geometric transformations are represented using **homogeneous coordinates** to combine operations (e.g., rotation, translation) into a single matrix multiplication. For a 2D image, a pixel \( (x, y) \) is represented as \( (x, y, 1) \), and transformations are applied via a 3x3 matrix.

<img width="614" height="354" alt="image" src="https://github.com/user-attachments/assets/94e5f49e-6f6e-4ee1-8715-76db48dfe31c" />


#### Scaling
- **Definition**: Resizes the image by scaling factors \( s_x \) (x-axis) and \( s_y \) (y-axis).
<img width="845" height="422" alt="image" src="https://github.com/user-attachments/assets/93288b5f-59da-4117-9d20-08d8cc8b31b0" />


#### Rotation
- **Definition**: Rotates the image by angle $\( \theta \)$ (counterclockwise) around the origin or a specified center.
- **Matrix** (rotation around origin):
 <img width="462" height="133" alt="image" src="https://github.com/user-attachments/assets/9d738e50-2777-455e-92e9-e7f62e8c0db7" />

- **Rotation Around Center**:
  - Translate the center to the origin, rotate, then translate back:
  <img width="533" height="61" alt="image" src="https://github.com/user-attachments/assets/c83caab3-f722-40e9-ace5-71d84d495291" />

- **Example**: For $\( \theta = 45^\circ \)$, $\( \cos 45^\circ = \sin 45^\circ = \frac{\sqrt{2}}{2} \)$.

#### Translation
- **Definition**: Shifts the image by $\( t_x \)$ (x-axis) and $\( t_y \)$ (y-axis).
- **Matrix**:
 <img width="342" height="136" alt="image" src="https://github.com/user-attachments/assets/6160a078-a50f-487d-a650-79912d604e0c" />

- **Effect**: $\( (x, y) \to (x + t_x, y + t_y) \)$.
- **Example**: For $\( t_x = 50, t_y = 30 \)$, shifts the image 50 pixels right and 30 pixels down.

#### Shearing
- **Definition**: Slants the image along one axis by factors $\( a \)$ (x-axis shear) or $\( b \)$ (y-axis shear).
<img width="739" height="451" alt="image" src="https://github.com/user-attachments/assets/96821cc6-e435-4e69-ac9d-6b49f72ec16d" />

- **Example**: For $\( a = 0.5 \)$, a vertical line slants right as $\( y \)$ increases.

#### Interpolation
- Transformations produce non-integer coordinates, requiring interpolation (e.g., nearest-neighbor, bilinear) to compute pixel values.
- Bilinear interpolation is common for smooth results.

#### Example
<img width="787" height="427" alt="image" src="https://github.com/user-attachments/assets/0df1566f-d575-482f-8461-0b41632d07a1" />


---

### 3. Practical Considerations
- **Transformation Matrix**:
  - Combine multiple transformations by multiplying matrices: \( T = T_1 \cdot T_2 \cdot \ldots \).
  - Use OpenCV’s `cv2.getAffineTransform` or `cv2.getRotationMatrix2D` for common transformations.
- **Interpolation**:
  - Use `cv2.INTER_LINEAR` (bilinear) for smooth results or `cv2.INTER_NEAREST` for speed.
  - Higher-order interpolation (e.g., `cv2.INTER_CUBIC`) improves quality but is slower.
- **Image Size**:
  - Scaling or rotation may require adjusting the output image size to avoid cropping.
  - Translation and shearing may shift content out of bounds; use padding or adjust the transformation matrix.
- **Applications**:
  - **Scaling**: Resize images for display or processing.
  - **Rotation**: Correct tilted images (e.g., scanned documents).
  - **Translation**: Align images in registration tasks.
  - **Shearing**: Correct perspective distortions or create artistic effects.
- **Limitations**:
  - Interpolation introduces artifacts, especially in scaling or rotation.
  - Repeated transformations may degrade quality due to cumulative interpolation errors.
- **Color Images**:
  - Apply transformations to each channel (RGB) or convert to grayscale.

---

### 4. Code Implementation in Python

Below is a Python code example using OpenCV and NumPy to demonstrate scaling, rotation, translation, and shearing. The code:
- Loads a color image.
- Applies each transformation (scaling, rotation, translation, shearing).
- Visualizes the original and transformed images.

<xaiArtifact artifact_id="01ec69ba-bda6-4c24-b717-12b0dd691253" artifact_version_id="8308f29f-8c45-4ba0-a8e9-b6b4a29d9f72" title="geometric_transformations.py" contentType="text/python">
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

# Image dimensions
rows, cols = img.shape[:2]

# 1. Scaling
scale_x, scale_y = 1.5, 1.5
img_scaled = cv2.resize(img_rgb, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# 2. Rotation
center = (cols // 2, rows // 2)
angle = 45  # Degrees
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
img_rotated = cv2.warpAffine(img_rgb, rotation_matrix, (cols, rows), flags=cv2.INTER_LINEAR)

# 3. Translation
t_x, t_y = 100, 50
translation_matrix = np.float32([[1, 0, t_x], [0, 1, t_y]])
img_translated = cv2.warpAffine(img_rgb, translation_matrix, (cols, rows), flags=cv2.INTER_LINEAR)

# 4. Shearing (x-axis shear)
shear_x = 0.3
shear_matrix = np.float32([[1, shear_x, 0], [0, 1, 0]])
img_sheared = cv2.warpAffine(img_rgb, shear_matrix, (cols, rows), flags=cv2.INTER_LINEAR)

# 5. Display Results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

# Scaled Image
plt.subplot(2, 3, 2)
plt.title(f'Scaled (sx={scale_x}, sy={scale_y})')
plt.imshow(img_scaled)
plt.axis('off')

# Rotated Image
plt.subplot(2, 3, 3)
plt.title(f'Rotated (θ={angle}°)')
plt.imshow(img_rotated)
plt.axis('off')

# Translated Image
plt.subplot(2, 3, 4)
plt.title(f'Translated (tx={t_x}, ty={t_y})')
plt.imshow(img_translated)
plt.axis('off')

# Sheared Image
plt.subplot(2, 3, 5)
plt.title(f'Sheared (a={shear_x})')
plt.imshow(img_sheared)
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original RGB: {img_rgb[y, x]}")
if y < img_scaled.shape[0] and x < img_scaled.shape[1]:
    print(f"Scaled RGB: {img_scaled[y, x]}")
print(f"Rotated RGB: {img_rotated[y, x]}")
print(f"Translated RGB: {img_translated[y, x]}")
print(f"Sheared RGB: {img_sheared[y, x]}")
</xaiArtifact>

---

### Code Explanation
1. **Loading Image**:
   - Loads a color image using `cv2.imread` and converts BGR to RGB for display.

2. **Scaling**:
   - Uses `cv2.resize` with scaling factors $\( s_x = 1.5, s_y = 1.5 \)$ and bilinear interpolation.

3. **Rotation**:
   - Uses `cv2.getRotationMatrix2D` to create a 2x3 rotation matrix for $\( \theta = 45^\circ \)$ around the image center.
   - Applies `cv2.warpAffine` to transform the image.

4. **Translation**:
   - Defines a 2x3 translation matrix for $\( t_x = 100, t_y = 50 \)$.
   - Applies `cv2.warpAffine` to shift the image.

5. **Shearing**:
   - Defines a 2x3 shear matrix for x-axis shear $(\( a = 0.3 \))$.
   - Applies `cv2.warpAffine` to slant the image.

6. **Visualization**:
   - Displays the original, scaled, rotated, translated, and sheared images.

7. **Pixel Value Output**:
   - Prints RGB values at (100, 100) for all images (if within bounds).

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide an image file (`image.jpg`) in the same directory or update the `image_path` variable. Use a color image with clear objects for meaningful results.

---

### Expected Output
- A figure with five subplots:
  - Original RGB image.
  - Scaled image (enlarged by 1.5x).
  - Rotated image (45° counterclockwise).
  - Translated image (shifted 100 pixels right, 50 pixels down).
  - Sheared image (slanted along x-axis).
- Printed pixel values, e.g.:
  ```
  Pixel at (100, 100):
  Original RGB: [150 100  50]
  Scaled RGB: [145  95  45]
  Rotated RGB: [0 0 0]
  Translated RGB: [0 0 0]
  Sheared RGB: [0 0 0]
  ```
  (Note: Some pixels may be [0, 0, 0] if they map outside the image bounds.)

---

### Practical Notes
- **Scaling**:
  - Adjust $\( s_x, s_y \)$ for desired size; ensure output dimensions fit the display.
  - Use `cv2.INTER_CUBIC` for high-quality upscaling.
- **Rotation**:
  - Adjust output size to avoid cropping:
    ```python
    new_cols = int(abs(cols * np.cos(np.radians(angle))) + abs(rows * np.sin(np.radians(angle))))
    new_rows = int(abs(cols * np.sin(np.radians(angle))) + abs(rows * np.cos(np.radians(angle))))
    ```
- **Translation**:
  - Ensure $\( t_x, t_y \)$ do not shift content out of bounds; use padding if needed.
- **Shearing**:
  - Shear factors $\( a, b \)$ (e.g., 0.1–0.5) control slant; large values may distort significantly.
- **Applications**:
  - **Scaling**: Resize images for machine learning or display.
  - **Rotation**: Correct tilted scanned documents or align images.
  - **Translation**: Register images in medical or satellite imaging.
  - **Shearing**: Correct lens distortions or create artistic effects.
- **Limitations**:
  - Interpolation artifacts may occur, especially with nearest-neighbor.
  - Transformations may clip content; adjust output size or use padding.

---

### Mathematical Insights
- **Affine Transformation**: General form:
 <img width="395" height="133" alt="image" src="https://github.com/user-attachments/assets/20627c5e-c382-48d2-a8e7-d9e440cf67be" />

- **Composition**: Combine transformations by matrix multiplication:
 <img width="435" height="52" alt="image" src="https://github.com/user-attachments/assets/abcdfd0e-6c06-4428-a1e4-28c14275f466" />

- **Inverse Transformation**: Compute $\( T^{-1} \)$ to reverse the transformation.

---

### Extensions
- **Perspective Transformations**: Use 3x3 homography matrices for non-affine transformations (e.g., perspective correction).
- **Combined Transformations**: Apply multiple transformations in a single matrix.
- **Image Registration**: Align images using feature matching (e.g., SIFT) and transformations.
- **Data Augmentation**: Apply random transformations for machine learning.

