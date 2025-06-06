import cv2
import numpy as np 
import matplotlib.pyplot as plt

# Attempt to read in the image
image_path = "11.jpg"  # Change this to the full path if necessary
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Image not found or could not be opened: {image_path}")
    exit()

image = cv2.resize(image, (1280, 720))  # Resize the image
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur

# Define Prewitt kernels
prewitt_x = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]])

prewitt_y = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])

# Apply Prewitt operator
gradient_x = cv2.filter2D(blurred, cv2.CV_64F, prewitt_x)  # Gradient in x direction
gradient_y = cv2.filter2D(blurred, cv2.CV_64F, prewitt_y)  # Gradient in y direction
prewitt_combined = cv2.magnitude(gradient_x, gradient_y)  # Combine the gradients

# Normalize the combined gradient image to uint8
prewitt_combined = cv2.normalize(prewitt_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display the Prewitt edge detection result
plt.figure()
plt.imshow(prewitt_combined, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis("off")
plt.show()

# Find contours from the Prewitt edge detection result
contours, hierarchy = cv2.findContours(prewitt_combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Identify the target contour (if needed for further processing)
target = None
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break

if target is not None:
    print("Found a valid contour with 4 points.")
else:
    print("No valid contour found.")
