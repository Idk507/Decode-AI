### Transformations in Image Processing

Transformations in image processing involve modifying the spatial arrangement or pixel values of an image to achieve specific effects, such as resizing, rotating, or warping. These operations are essential for tasks like image alignment, correction of distortions, augmentation for machine learning, and creating visual effects. This explanation covers the key types of transformations, their mathematical foundations, practical applications, and a detailed Python code implementation using OpenCV and NumPy.

---

### 1. Overview of Transformations

Transformations in image processing can be broadly classified into two categories:

- **Geometric Transformations**: Alter the spatial arrangement of pixels (e.g., translation, rotation, scaling, shearing, perspective warping). These are typically applied using a transformation matrix.
- **Intensity Transformations**: Modify pixel values without changing their positions (e.g., brightness adjustment, contrast stretching). Since the user’s query focuses on transformations in the context of image processing basics (alongside filtering and edge detection), this response emphasizes **geometric transformations**, as intensity transformations are more akin to histogram processing or enhancement.

#### Key Geometric Transformations:
- **Translation**: Shifts the image along the x or y axis.
- **Rotation**: Rotates the image around a specified point (usually the center).
- **Scaling**: Resizes the image by stretching or shrinking it.
- **Shearing**: Slants the image along one axis.
- **Affine Transformation**: Combines linear transformations (translation, rotation, scaling, shearing) while preserving parallelism.
- **Perspective Transformation**: Warps the image to simulate a change in viewpoint, preserving collinearity but not parallelism.

#### Applications:
- **Image Alignment**: Align images for stitching (e.g., panoramas) or registration (e.g., medical imaging).
- **Augmentation**: Generate variations of images for machine learning training.
- **Distortion Correction**: Correct lens distortions in cameras.
- **Visual Effects**: Create artistic effects or animations.

---

### 2. Mathematical Concepts of Geometric Transformations

Geometric transformations map each pixel’s coordinates $\((x, y)\)$ in the input image to new coordinates $\((x', y')\)$ in the output image. This mapping is typically represented by a transformation matrix applied to homogeneous coordinates.

#### Homogeneous Coordinates
To handle translations and other transformations uniformly, pixels are represented in homogeneous coordinates as $\((x, y, 1)\)$. A 2D transformation is then applied using a 3x3 matrix:

<img width="302" height="115" alt="image" src="https://github.com/user-attachments/assets/429cff06-0173-4d03-adae-98c67fbd2060" />


Where $\((x', y')\)$ are the transformed coordinates, and the matrix defines the transformation.

#### Common Geometric Transformations

1. **Translation**:
   - Shifts the image by $\(t_x\) (horizontal) and \(t_y\) (vertical)$.
   - Transformation Matrix:
    <img width="240" height="123" alt="image" src="https://github.com/user-attachments/assets/aa32ec38-db8b-44fd-bc01-5869f1ceefdf" />

   - Effect: $\((x', y') = (x + t_x, y + t_y)\)$.
   - Example: Moving an image 50 pixels right and 30 pixels down: $\(t_x = 50\), \(t_y = 30\)$.

2. **Rotation**:
   - Rotates the image by an angle $\(\theta\)$ around a point (usually the image center).
   - Transformation Matrix (rotation around origin):
     <img width="309" height="137" alt="image" src="https://github.com/user-attachments/assets/a2aa4027-7d18-47c6-b888-f66148578fd2" />

   - For rotation around a point $\((c_x, c_y)\)$, translate to the origin, rotate, then translate back.
   - Example: Rotating 45° around the center adjusts pixel positions based on trigonometric functions.

3. **Scaling**:
   - Resizes the image by scaling factors $\(s_x\)$ (horizontal) and $\(s_y\)$ (vertical).
   - Transformation Matrix:
     <img width="208" height="131" alt="image" src="https://github.com/user-attachments/assets/c22dc5d2-5fca-42d0-990f-2a7de3105f2c" />

   - Effect: $\((x', y') = (s_x \cdot x, s_y \cdot y)\)$.
   - Example: Doubling the size: $\(s_x = 2\), \(s_y = 2\)$ .

4. **Shearing**:
   - Slants the image along one axis.
   - Transformation Matrix (horizontal shear):
   <img width="213" height="126" alt="image" src="https://github.com/user-attachments/assets/b43e3022-d517-455e-acd9-dcf60b1c7a32" />

   - Effect: $\((x', y') = (x + k \cdot y, y)\)$ .
   - Example: $\(k = 0.5\)$  slants the image horizontally.

5. **Affine Transformation**:
   - Combines translation, rotation, scaling, and shearing.
   - General Form:
<img width="240" height="131" alt="image" src="https://github.com/user-attachments/assets/a2bd127d-0402-4a85-a596-cdb679c501ac" />

   - Preserves straight lines and parallelism but not distances or angles.
   - Example: Combining rotation and translation to rotate an image around its center.

6. **Perspective Transformation**:
   - Warps the image to simulate a 3D viewpoint change.
   - Transformation Matrix (3x3, non-zero bottom row):
    <img width="218" height="113" alt="image" src="https://github.com/user-attachments/assets/47a13a14-9ef6-46fd-b4df-f1259891b279" />

   - Effect: $\((x', y', w') = P \cdot (x, y, 1)\)$ , where final coordinates are \((x'/w', y'/w')\).
   - Requires four corresponding points between input and output images to compute the matrix.
   - Example: Correcting a photo of a tilted document to appear flat.

#### Interpolation
Since transformed coordinates $\((x', y')\)$  may not align with integer pixel positions, interpolation is used to estimate pixel values:
- **Nearest Neighbor**: Assigns the value of the closest pixel (fast but blocky).
- **Bilinear**: Interpolates using a weighted average of the four nearest pixels (smoother).
- **Bicubic**: Uses 16 neighboring pixels for higher-quality results (slower).

#### Inverse Mapping
To avoid gaps in the output image, transformations are often applied using **inverse mapping**:
- For each pixel in the output image, compute its corresponding position in the input image using the inverse transformation matrix.
- Interpolate the input image’s pixel values at that position.

---

### 3. Example: Applying a Rotation Transformation

Consider rotating an image by 45° around its center. For an image of size $\(W \times H\)$ , the center is $\((c_x, c_y) = (W/2, H/2)\)$ .

1. Translate the image so the center is at the origin: $\((x - c_x, y - c_y)\)$.
2. Apply rotation:
 <img width="484" height="135" alt="image" src="https://github.com/user-attachments/assets/6e82a24d-e070-4127-81f7-d8bcbb21aa9d" />

3. Translate back: $\((x_r + c_x, y_r + c_y)\)$ .

For a pixel at $\((100, 100)\)$ in a 200x200 image $(\(c_x = 100\), \(c_y = 100\))$ :
- Translate to origin: $\((0, 0)\)$ .
- Rotate by 45° $(\(\cos 45^\circ = \sin 45^\circ \approx 0.707\))$ :
  <img width="340" height="94" alt="image" src="https://github.com/user-attachments/assets/b5b0155b-f1e1-4519-9fd9-f170b993df00" />

- Translate back: $\((0 + 100, 0 + 100) = (100, 100)\)$.
- For non-center pixels, the rotation shifts their positions, and interpolation estimates the new pixel value.

---

### 4. Code Implementation in Python

Below is a Python code example using OpenCV and NumPy to demonstrate common geometric transformations: translation, rotation, scaling, shearing, affine, and perspective transformations. The code applies these transformations to an image and visualizes the results.

```python
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
```

---

### Code Explanation
1. **Transformation Functions**:
   - **Translation**: Uses a 2x3 affine matrix to shift the image by $\(t_x, t_y\)$.
   - **Rotation**: Uses `cv2.getRotationMatrix2D` to compute a 2x3 matrix for rotation around the center.
   - **Scaling**: Applies a scaling matrix, adjusting the output image size.
   - **Shearing**: Uses a shear matrix with parameters $\(k_x, k_y\)$.
   - **Affine**: Computes a 2x3 matrix from three corresponding points using `cv2.getAffineTransform`.
   - **Perspective**: Computes a 3x3 matrix from four corresponding points using `cv2.getPerspectiveTransform`.

2. **Applying Transformations**:
   - `cv2.warpAffine` is used for translation, rotation, scaling, shearing, and affine transformations (2x3 matrices).
   - `cv2.warpPerspective` is used for perspective transformations (3x3 matrix).
   - OpenCV uses inverse mapping and bilinear interpolation by default.

3. **Visualization**:
   - Displays the original image and transformed images (translation, rotation, scaling, shearing, affine) in a 2x3 grid.
   - Perspective transformation is shown separately due to space constraints.

4. **Output**:
   - Prints the rotation matrix as an example of the transformation matrix.

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide an image file (`image.jpg`) in the same directory or update the `image_path` variable.

---

### Expected Output
- A figure with six subplots showing:
  - Original image.
  - Translated image (shifted 50 pixels right, 30 pixels down).
  - Rotated image (45° around the center).
  - Scaled image (2x size).
  - Sheared image (horizontal slant).
  - Affine transformed image (based on three-point mapping).
- A separate figure showing the perspective transformed image (simulating a tilted view).
- Printed rotation matrix, e.g.:
  ```
  Rotation Matrix (45°):
  [[ 0.70710678  0.70710678 -0.41421356]
   [-0.70710678  0.70710678  0.41421356]]
  ```

---

### Practical Notes
- **Interpolation**: OpenCV’s `warpAffine` and `warpPerspective` use bilinear interpolation by default. For higher quality, specify `cv2.INTER_CUBIC`.
- **Boundary Handling**: Pixels outside the image are set to black by default. Use `borderMode` (e.g., `cv2.BORDER_REPLICATE`) to handle borders differently.
- **Affine vs. Perspective**:
  - Affine transformations require three points and preserve parallelism.
  - Perspective transformations require four points and can model 3D effects like vanishing points.
- **Performance**: Transformations are fast for small images but can be slow for large images or complex perspective warps.
- **Applications**: Perspective transformations are used in augmented reality, while affine transformations are common in image registration.

---

### Mathematical Insights
- **Affine Transformations**: The 2x3 matrix has six degrees of freedom (rotation, translation, scaling, shearing), computed from three corresponding points.
- **Perspective Transformations**: The 3x3 matrix has eight degrees of freedom (one fixed), computed from four corresponding points, allowing for non-parallel mappings.
- **Inverse Mapping**: Ensures no gaps in the output image by mapping output pixels back to input coordinates.
- **Rotation Center**: Rotating around the image center requires combining translation and rotation matrices.

---

### Extensions
- **Custom Transformation Matrices**: Manually define transformation matrices for specific effects.
- **Image Registration**: Use feature matching (e.g., SIFT) to compute transformation matrices automatically.
- **Animation**: Apply a sequence of transformations for visual effects.
- **Inverse Transformations**: Compute the inverse matrix to undo a transformation.

