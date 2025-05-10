import streamlit as st
from PIL import Image
import numpy as np
#from canny import canny_edge
import cv2

def compare_algorithms(images):
    comparisons = {}
    # Example comparison (mean pixel value for simplicity)
    for name, image in images.items():
        comparisons[name] = np.mean(image)
    return comparisons

def canny_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def sobel_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)  
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)  
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)  
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return sobel_combined

def prewitt_edge(image):
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

    return prewitt_combined



def yolo_edge(image):
     # Load YOLO model configuration and weights
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    # Load the class labels
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Get the names of the output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input for the network
    net.setInput(blob)

    # Forward pass through the network
    outputs = net.forward(output_layers)

    # Process the outputs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and detect edges
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            # Extract the region of interest (ROI) for edge detection
            roi = image[y:y + h, x:x + w]

            # Convert the ROI to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply Sobel edge detection
            sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=5)
            edges = cv2.sqrt(sobelx**2 + sobely**2)

            # Normalize the edges to [0, 255]
            edges = cv2.convertScaleAbs(edges)

            # Overlay the edges on the original image
            image[y:y + h, x:x + w] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return image


st.title("Edge Detection Algorithms with COCO Dataset")
st.write("Upload an image from the COCO dataset and choose an edge detection algorithm.")

# Image upload
uploaded_file = st.file_uploader("choose an image!!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
        # Open and convert the image
        image = Image.open(uploaded_file)
        image = np.array(image)

        
        # Display original image
        st.image(image, caption="Original Image", use_column_width=True)

        algorithm = st.selectbox(
            "Select Edge Detection Algorithm",
            ("Canny", "Sobel", "Prewitt", "YOLO")
        )

        # Edge detection
        if st.button("Apply Edge Detection"):
            if algorithm == "Canny":
                output_image = canny_edge(image)
            elif algorithm == "Sobel":
                output_image = sobel_edge(image)
            elif algorithm == "YOLO":
                output_image = yolo_edge(image)
            elif algorithm == "Prewitt":
                output_image = prewitt_edge(image)

            # Display output image
            st.image(output_image, caption=f"{algorithm} Edge Detection", use_column_width=True)
        
        # Compare algorithms
        if st.button("Compare Algorithms"):
            results = {
                "Canny": canny_edge(image),
                "Sobel": sobel_edge(image),
                "Prewitt": prewitt_edge(image),
                "YOLO": yolo_edge(image)
            }

            comparisons = compare_algorithms(results)

            # Display comparison results
            st.subheader("Comparison Results")
            for name, score in comparisons.items():
                st.write(f"{name}: Mean Pixel Value = {score:.2f}")
