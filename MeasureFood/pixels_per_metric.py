import cv2
import numpy as np

from FoodAreaSegmentation.utils import format_bbox
from detect_coin import detect_coin

def get_circle_or_rectangle(image, is_coin=True):

    boxes,scores = detect_coin(image_path=image,model_path=r'best.pt')
    roi = format_bbox(boxes[0]) 
    # Extract bounding box coordinates
    x, y, w, h = map(int,roi)

    w = int(w*1.2)
    h = int(h*1.2)
    # Crop the region containing the coin or bill
    roi_patch = image[y:y+h, x:x+w]

    # Convert to grayscale
    gray = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Apply dilation to close small gaps in the contour
    kernel = np.ones((5,5),np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (assuming it's the coin or bill)
    largest_contour = max(contours, key=cv2.contourArea)

    # If it's a coin
    if is_coin:
        # Get the center and radius of the circle
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(largest_contour)
        center = (int(center_x) + x, int(center_y) + y)
        radius = int(radius)

        # Draw the circle on the original image
        cv2.circle(image, center, radius, (0, 255, 0), 2)

        # Return diameter and center coordinates
        diameter = 2 * radius
        return diameter, center

    # If it's a bill
    else:
        # Get bounding rectangle around the contour
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
        # Adjust the rectangle coordinates to the original image
        x_rect += x
        y_rect += y

        # Draw the rectangle on the original image
        cv2.rectangle(image, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), 2)

        # Return rectangle coordinates
        return (x_rect, y_rect, w_rect, h_rect)