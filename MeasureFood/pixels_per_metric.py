import cv2
import numpy as np
import yaml

from FoodAreaSegmentation.utils import format_bbox
from detect_coin import detect_coin

def load_yaml_to_dict(file_path):
    with open(file_path, "r") as yaml_file:
        loaded_dict = yaml.safe_load(yaml_file)
    return loaded_dict

def get_circle_or_rectangle(image, show=False):

    #Detections
    classes,boxes,scores = detect_coin(image_path=image,model_path=r'best.pt')

    #Detect if it's a coin or a bill
    if int(classes[0]) in [0,1,2]:
      is_coin=True
    else:
      is_coin=False
    
    #Enlarge the bounding box
    boxes[0][2] = boxes[0][2]*1.1
    boxes[0][3] = boxes[0][3]*1.1

    #format bbox
    roi = format_bbox(boxes[0])

    # Extract bounding box coordinates
    x0, y0, x1, y1 = map(int,roi)

    # Crop the region containing the coin or bill
    roi_patch = image[y0:y1, x0:x1]

    # Convert to grayscale
    gray = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    kernel_open = np.ones((2,2),np.uint8)
    kernel_close = np.ones((3,3),np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (assuming it's the coin or bill)
    largest_contour = max(contours, key=lambda x: cv2.arcLength(x, closed=False))

    # If it's a coin
    if is_coin:
        # Get the center and radius of the circle
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(largest_contour)
        center = (int(center_x) + x0, int(center_y) + y0)
        radius = int(radius)

        if show:
            # Draw the circle on the original image
            cv2.circle(image, center, radius, (0, 255, 0), 2)

        # Return diameter and center coordinates
        diameter = 2 * radius
        return diameter, center

    # If it's a bill
    else:
        # Get minimum area bounding rectangle around the contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        
        if show:
            # Draw the rotated rectangle on the original image
            box += np.array([[x0, y0]])
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # Calculate the width and height of the rotated rectangle
        (x, y), (w_rect, h_rect), angle = rect

        # Adjust the rectangle coordinates to the original image
        x_rect = x + x0
        y_rect = y + y0

        # Return rectangle coordinates
        return (x_rect, y_rect, w_rect, h_rect)