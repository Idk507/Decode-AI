import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform histogram matching (grayscale)
def histogram_matching(source, reference):
    # Compute histograms and CDFs
    src_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, bins = np.histogram(reference.flatten(), 256, [0, 256])
    
    src_cdf = src_hist.cumsum()
    src_cdf = src_cdf / src_cdf[-1]  # Normalize to [0, 1]
    
    ref_cdf = ref_hist.cumsum()
    ref_cdf = ref_cdf / ref_cdf[-1]  # Normalize to [0, 1]
    
    # Create mapping from source to reference intensities
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Find the closest reference CDF value to source CDF
        j = np.argmin(np.abs(ref_cdf - src_cdf[i]))
        mapping[i] = j
    
    # Apply mapping to source image
    matched = cv2.LUT(source, mapping)
    return matched

# Load source and reference images
source_path = 'source.jpg'  # Replace with your source image path
reference_path = 'reference.jpg'  # Replace with your reference image path
img_source = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
img_reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

# Check if images are loaded successfully
if img_source is None or img_reference is None:
    print("Error: Could not load one or both images.")
    exit()

# 1. Grayscale Histogram Matching
img_matched = histogram_matching(img_source, img_reference)

# 2. Color Histogram Matching (HSV)
img_source_color = cv2.imread(source_path)
img_reference_color = cv2.imread(reference_path)

# Convert to HSV
img_source_hsv = cv2.cvtColor(img_source_color, cv2.COLOR_BGR2HSV)
img_reference_hsv = cv2.cvtColor(img_reference_color, cv2.COLOR_BGR2HSV)

# Split HSV channels
h_s, s_s, v_s = cv2.split(img_source_hsv)
h_r, s_r, v_r = cv2.split(img_reference_hsv)

# Match the Value channel
v_matched = histogram_matching(v_s, v_r)

# Merge back and convert to RGB
img_matched_hsv = cv2.merge([h_s, s_s, v_matched])
img_matched_color = cv2.cvtColor(img_matched_hsv, cv2.COLOR_HSV2RGB)
img_source_rgb = cv2.cvtColor(img_source_color, cv2.COLOR_BGR2RGB)
img_reference_rgb = cv2.cvtColor(img_reference_color, cv2.COLOR_BGR2RGB)

# 3. Compute Histograms
def compute_histogram(image, title):
    if len(image.shape) == 3:  # Color image
        colors = ('r', 'g', 'b')
        hist = []
        for i, color in enumerate(colors):
            hist_i = cv2.calcHist([image], [i], None, [256], [0, 256])
            hist.append((hist_i, color))
        return hist, f"{title} (RGB)"
    else:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return [(hist, 'gray')], title

hist_source, title_source = compute_histogram(img_source, 'Source Grayscale Histogram')
hist_reference, title_reference = compute_histogram(img_reference, 'Reference Grayscale Histogram')
hist_matched, title_matched = compute_histogram(img_matched, 'Matched Grayscale Histogram')

# 4. Display Results
plt.figure(figsize=(15, 10))

# Source Grayscale Image
plt.subplot(3, 3, 1)
plt.title('Source Grayscale')
plt.imshow(img_source, cmap='gray')
plt.axis('off')

# Reference Grayscale Image
plt.subplot(3, 3, 2)
plt.title('Reference Grayscale')
plt.imshow(img_reference, cmap='gray')
plt.axis('off')

# Matched Grayscale Image
plt.subplot(3, 3, 3)
plt.title('Matched Grayscale')
plt.imshow(img_matched, cmap='gray')
plt.axis('off')

# Source Color Image
plt.subplot(3, 3, 4)
plt.title('Source Color (RGB)')
plt.imshow(img_source_rgb)
plt.axis('off')

# Reference Color Image
plt.subplot(3, 3, 5)
plt.title('Reference Color (RGB)')
plt.imshow(img_reference_rgb)
plt.axis('off')

# Matched Color Image
plt.subplot(3, 3, 6)
plt.title('Matched Color (HSV)')
plt.imshow(img_matched_color)
plt.axis('off')

# Plot Histograms
plt.subplot(3, 3, 7)
plt.title(title_source)
for hist, color in hist_source:
    plt.plot(hist, color=color)
plt.xlim([0, 256])

plt.subplot(3, 3, 8)
plt.title(title_reference)
for hist, color in hist_reference:
    plt.plot(hist, color=color)
plt.xlim([0, 256])

plt.subplot(3, 3, 9)
plt.title(title_matched)
for hist, color in hist_matched:
    plt.plot(hist, color=color)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

# 5. Example: Print pixel values at (100, 100)
x, y = 100, 100
print(f"Pixel at ({x}, {y}):")
print(f"Source Grayscale: {img_source[y, x]}")
print(f"Reference Grayscale: {img_reference[y, x]}")
print(f"Matched Grayscale: {img_matched[y, x]}")
print(f"Source Color (RGB): {img_source_rgb[y, x]}")
print(f"Reference Color (RGB): {img_reference_rgb[y, x]}")
print(f"Matched Color (RGB): {img_matched_color[y, x]}")
