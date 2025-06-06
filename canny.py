import cv2
import numpy as np 
import matplotlib.pyplot as plt

def canny_edge(image):
    # Attempt to read in the image
    #image_path = "11.jpg"  # Change this to the full path if necessary
    #image = cv2.imread(image)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Image not found or could not be opened: {image}")
        exit()
    
    image = cv2.resize(image, (1300, 800))  # Resize the image
    orig = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    
    # Apply Canny edge detection
    edged = cv2.Canny(blurred, 30, 50)

      
    
    # Display the Canny edge detection result
    #plt.figure()
    #plt.imshow(edged, cmap='gray')
    #plt.title("Canny Edge Detection")
    #plt.axis("off")
    #plt.show()
    
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
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

    return plt.show()

# Call the function with the image path
#canny_edge("11.jpg")  # Ensure the image path is correct