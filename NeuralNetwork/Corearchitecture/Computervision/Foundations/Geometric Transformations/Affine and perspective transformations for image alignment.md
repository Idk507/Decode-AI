### Affine and Perspective Transformations for Image Alignment

**Affine** and **perspective transformations** are geometric transformations used to align images by mapping points from one image to corresponding points in another. These transformations are essential for tasks like image registration, stitching, or correcting distortions. **Affine transformations** preserve parallel lines and include scaling, rotation, translation, and shearing, while **perspective transformations** handle more complex distortions, such as those caused by camera angles. This explanation covers the mathematical concepts, practical applications, and a detailed Python code implementation using OpenCV and NumPy to demonstrate affine and perspective transformations for image alignment.

---

### 1. What are Affine and Perspective Transformations?

- **Affine Transformation**:
  - Definition: A linear mapping that preserves points, straight lines, and parallelism, but not necessarily distances or angles.
  - Operations: Scaling, rotation, translation, shearing.
  - Matrix: 2x3 matrix (6 parameters) in homogeneous coordinates.
  - Use Case: Align images with simple distortions (e.g., rotated or scaled views).
- **Perspective Transformation**:
  - Definition: A more general mapping that preserves straight lines but not parallelism, accounting for perspective effects (e.g., vanishing points).
  - Matrix: 3x3 homography matrix (8 parameters) in homogeneous coordinates.
  - Use Case: Correct perspective distortions (e.g., images taken from different angles).
- **Purpose**:
  - Align images for registration, stitching, or correction.
  - Correct distortions caused by camera angles or lens effects.
- **Applications**:
  - Image registration (e.g., aligning medical or satellite images).
  - Panorama stitching (e.g., combining photos into a single wide image).
  - Augmented reality (e.g., overlaying objects in real-world scenes).
  - Document scanning (e.g., correcting skewed pages).

---

### 2. Mathematical Concepts

Both transformations use **homogeneous coordinates** to represent pixel mappings as matrix operations. Image alignment involves finding the transformation that maps points in one image to corresponding points in another.

#### Affine Transformation
<img width="834" height="319" alt="image" src="https://github.com/user-attachments/assets/7fb48ef1-2604-446a-9aa5-8968bbef82d7" />

- **Parameters**: 6 degrees of freedom $(\( a, b, c, d, e, f \))$.
- **Estimation**: Requires at least 3 pairs of corresponding points, solved using:
  - OpenCV’s `cv2.getAffineTransform` for three points.
  - Least-squares for more points (e.g., RANSAC for robustness).
- **Properties**: Preserves parallelism, ratios of distances along lines.

#### Perspective Transformation
-<img width="805" height="303" alt="image" src="https://github.com/user-attachments/assets/1a211a67-9d12-48da-a168-d9a395242362" />

- **Parameters**: 8 degrees of freedom (normalized by $\( h_{33} = 1 \)$).
- **Estimation**: Requires at least 4 pairs of corresponding points, solved using:
  - OpenCV’s `cv2.getPerspectiveTransform` for four points.
  - Direct Linear Transformation (DLT) or RANSAC for more points.
- **Properties**: Preserves straight lines but not parallelism, handles perspective effects.

#### Image Alignment
- **Process**:
  1. Identify corresponding points (e.g., manually or via feature matching like SIFT).
  2. Estimate the transformation matrix $(\( T \)$ or $\( H \))$.
  3. Apply the transformation to warp one image to align with another.
- **Feature Matching**:
  - Use algorithms like SIFT, ORB, or manual point selection to find correspondences.
  - RANSAC improves robustness by filtering outliers.

#### Example
Consider two images with corresponding points:
- Source points: $\( [(0, 0), (100, 0), (0, 100), (100, 100)] \)$.
- Destination points: $\( [(10, 20), (110, 20), (0, 120), (100, 120)] \)$.
- **Affine Transformation**:
  - Use 3 points to compute $\( T \)$ (e.g., via `cv2.getAffineTransform`).
  - Maps source to destination, preserving parallelism.
- **Perspective Transformation**:
  - Use 4 points to compute $\( H \)$ (e.g., via `cv2.getPerspectiveTransform`).
  - Handles non-parallel mappings (e.g., correcting a trapezoid to a rectangle).

---

### 3. Practical Considerations
- **Point Correspondences**:
  - Manual selection for controlled settings (e.g., document correction).
  - Feature matching (e.g., SIFT, ORB) for automatic alignment.
- **Interpolation**:
  - Use `cv2.INTER_LINEAR` (bilinear) for smooth results or `cv2.INTER_CUBIC` for higher quality.
- **Output Size**:
  - Perspective transformations may require adjusting the output canvas to avoid cropping.
  - Compute the transformed corners to determine the new image size.
- **Applications**:
  - **Affine**: Align images with rotation, scaling, or translation (e.g., registering medical images).
  - **Perspective**: Correct perspective distortions (e.g., straightening a photographed document).
- **Limitations**:
  - Affine transformations cannot handle perspective effects (e.g., vanishing points).
  - Perspective transformations require accurate point correspondences; errors lead to misalignment.
  - Interpolation introduces artifacts, especially for large transformations.
- **Robustness**:
  - Use RANSAC (via `cv2.findHomography`) to handle outliers in feature matching.
- **Color Images**:
  - Apply transformations to each RGB channel or convert to grayscale for processing.

---

### 4. Code Implementation in Python

Below is a Python code example using OpenCV and NumPy to demonstrate affine and perspective transformations for image alignment. The code:
- Loads a source image and defines corresponding points manually (simulating a reference image).
- Computes and applies affine and perspective transformations.
- Visualizes the original and transformed images.

<xaiArtifact artifact_id="c48d924f-6c19-4aa5-a63d-76a5b4e23659" artifact_version_id="8bf08784-5e1b-49b9-b510-a2248c55238f" title="image_alignment.py" contentType="text/python">
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

# 1. Define corresponding points (source and destination)
# Example: Simulate a distorted image aligning to a reference
src_points = np.float32([
    [0, 0],       # Top-left
    [cols-1, 0],  # Top-right
    [0, rows-1],  # Bottom-left
    [cols-1, rows-1]  # Bottom-right
])
dst_points = np.float32([
    [50, 50],           # Top-left (shifted)
    [cols-50, 50],     # Top-right
    [0, rows-50],      # Bottom-left
    [cols-100, rows-50]  # Bottom-right (perspective distortion)
])

# 2. Affine Transformation (using 3 points)
affine_matrix = cv2.getAffineTransform(src_points[:3], dst_points[:3])
img_affine = cv2.warpAffine(img_rgb, affine_matrix, (cols, rows), flags=cv2.INTER_LINEAR)

# 3. Perspective Transformation (using 4 points)
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_perspective = cv2.warpPerspective(img_rgb, perspective_matrix, (cols, rows), flags=cv2.INTER_LINEAR)

# 4. Display Results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

# Affine Transformed Image
plt.subplot(1, 3, 2)
plt.title('Affine Transformation')
plt.imshow(img_affine)
plt.axis('off')

# Perspective Transformed Image
plt.subplot(1, 3, 3)
plt.title('Perspective Transformation')
plt.imshow(img_perspective)
plt.axis('off')

plt.tight_layout()
plt.show()

# 5. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Original RGB: {img_rgb[y, x]}")
print(f"Affine Transformed RGB: {img_affine[y, x]}")
print(f"Perspective Transformed RGB: {img_perspective[y, x]}")

# 6. Print transformation matrices
print("\nAffine Matrix (2x3):")
print(affine_matrix)
print("\nPerspective Matrix (3x3):")
print(perspective_matrix)
</xaiArtifact>

---

### Code Explanation
1. **Loading Image**:
   - Loads a color image using `cv2.imread` and converts BGR to RGB for display.

2. **Corresponding Points**:
   - Defines source points (e.g., image corners) and destination points (simulating alignment to a reference).
   - Source points: Four corners of the original image.
   - Destination points: Shifted and distorted to simulate alignment needs.

3. **Affine Transformation**:
   - Uses `cv2.getAffineTransform` with 3 point pairs to compute a 2x3 matrix.
   - Applies `cv2.warpAffine` to transform the image, preserving parallelism.

4. **Perspective Transformation**:
   - Uses `cv2.getPerspectiveTransform` with 4 point pairs to compute a 3x3 homography matrix.
   - Applies `cv2.warpPerspective` to handle perspective distortions.

5. **Visualization**:
   - Displays the original, affine-transformed, and perspective-transformed images.

6. **Output**:
   - Prints RGB values at (100, 100) for all images.
   - Prints the affine and perspective transformation matrices.

---

### Prerequisites
- Install required libraries:
  ```bash
  pip install opencv-python numpy matplotlib
  ```
- Provide an image file (`image.jpg`) in the same directory or update the `image_path` variable. Use a color image with clear features for meaningful results.

---

### Expected Output
- A figure with three subplots:
  - Original RGB image.
  - Affine-transformed image (e.g., shifted and scaled).
  - Perspective-transformed image (e.g., corrected for perspective distortion).
- Printed pixel values, e.g.:
  ```
  Pixel at (100, 100):
  Original RGB: [150 100  50]
  Affine Transformed RGB: [0 0 0]
  Perspective Transformed RGB: [0 0 0]
  ```
  (Note: Some pixels may be [0, 0, 0] if they map outside the image bounds.)
- Printed transformation matrices, e.g.:
  ```
  Affine Matrix (2x3):
  [[ 1.00000000e+00  0.00000000e+00  5.00000000e+01]
   [ 0.00000000e+00  1.00000000e+00  5.00000000e+01]]

  Perspective Matrix (3x3):
  [[ 1.00000000e+00  0.00000000e+00  5.00000000e+01]
   [ 0.00000000e+00  1.00000000e+00  5.00000000e+01]
   [ 0.00000000e+00  0.00000000e+00  1.00000000e+00]]
  ```

---

### Practical Notes
- **Point Selection**:
  - Manual points are used here for simplicity; in practice, use feature matching (e.g., SIFT, ORB) for automatic alignment.
  - Ensure points are accurate to avoid misalignment.
- **Affine vs. Perspective**:
  - Use affine for simple transformations (e.g., rotation, scaling).
  - Use perspective for complex distortions (e.g., correcting a photographed document).
- **Output Size**:
  - Adjust the output canvas for perspective transformations to avoid cropping:
    ```python
    corners = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    transformed_corners = cv2.perspectiveTransform(corners[None, :, :], perspective_matrix)
    min_x, min_y = np.min(transformed_corners, axis=1)[0]
    max_x, max_y = np.max(transformed_corners, axis=1)[0]
    ```
- **Applications**:
  - **Affine**: Align rotated or scaled medical images.
  - **Perspective**: Correct skewed documents or stitch panoramas.
- **Limitations**:
  - Requires accurate point correspondences; errors lead to poor alignment.
  - Interpolation (e.g., bilinear) may introduce artifacts.
  - Perspective transformations are computationally heavier than affine.
- **Feature Matching**:
  - Use `cv2.findHomography` with RANSAC for robust perspective transformation estimation:
    ```python
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    ```

---

### Mathematical Insights
<img width="879" height="314" alt="image" src="https://github.com/user-attachments/assets/66848484-d406-49c0-bbd8-e3e75f7b10b2" />

  Handles perspective, solved with 4 point pairs.
- **Homogeneous Coordinates**: Enable linear representation of non-linear perspective mappings.

---

### Extensions
- **Feature-Based Alignment**: Use SIFT or ORB to automatically detect and match keypoints.
- **Image Stitching**: Combine multiple images using homography for panoramas.
- **Blind Alignment**: Estimate transformations without manual points using optimization.
- **Non-Linear Transformations**: Explore elastic or deformable registration.
- **Evaluation**: Measure alignment accuracy using metrics like mean squared error of point correspondences.

