import cv2
import numpy as np
import yaml
import math
import glob

from detect_coin import detect_coin


MEASURMENTS_YAML_FILE = r'MeasureFood\korean_coins_bills.yaml'

def format_bbox(input_bbox):
    #format the bounding box values to sam model bbox specifications
    return [input_bbox[0]-int(input_bbox[2]/2),
            input_bbox[1]-int(input_bbox[3]/2),
            input_bbox[0]+int(input_bbox[2]/2),
            input_bbox[1]+int(input_bbox[3]/2)]

def load_yaml_to_dict(file_path):
    with open(file_path, "r") as yaml_file:
        loaded_dict = yaml.safe_load(yaml_file)
    return loaded_dict

def get_circle_or_rectangle(image, show=False):

    #Detections
    classes, boxes, _ = detect_coin(image_path=image,model_path=r'MeasureFood\coin_detector.pt')

    #Detect if it's a coin or a bill
    if int(classes[0]) in [0,1,2,3]:
      is_coin=True
    else:
      is_coin=False
    
    #Enlarge the bounding box
    #boxes[0][2] = boxes[0][2]*1.1
    #boxes[0][3] = boxes[0][3]*1.1

    #format bbox
    roi = format_bbox(boxes[0])

    # Extract bounding box coordinates
    x0, y0, x1, y1 = map(int,roi)

    # Crop the region containing the coin or bill
    roi_patch = image[y0:y1, x0:x1]

    # Convert to grayscale
    gray = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)


    # If it's a coin
    if is_coin:
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3,3), 0)

        # Detect edges using Canny edge detector
        edges = cv2.Canny(blurred, 50, 225)

        # Apply dilation to close small gaps in the contour
        #kernel_open = np.ones((2,2),np.uint8)
        kernel_close = np.ones((3,3),np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)

        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contour (assuming it's the coin or bill)
        largest_contour = max(contours, key=lambda x: cv2.arcLength(x, closed=False))

        # Get the center and radius of the circle
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(largest_contour)
        center = (int(center_x) + x0, int(center_y) + y0)
        radius = int(radius)

        if show:
            # Draw the circle on the original image
            cv2.circle(image, center, radius, (0, 255, 0), 2)

        # Return diameter and center coordinates
        diameter = 2 * radius

        # Return diameter and label
        return diameter , int(classes[0])

    # If it's a bill
    else:
        # Detect edges using Canny edge detector
        edges = cv2.Canny(gray, 50,175)

        # Apply dilation to close small gaps in the contour
        #kernel_open = np.ones((2,2),np.uint8)
        kernel_close = np.ones((3,3),np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)

        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contour (assuming it's the coin or bill)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get minimum area bounding rectangle around the contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        #Adjust the box coordinates
        box += np.array([[x0, y0]])

        if show:
            # Draw the rotated rectangle on the original image
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # Calculate the width and height of the rotated rectangle
        (_, _), (w_rect, h_rect), _ = rect

        # Calculate diameter
        diameter = math.sqrt(w_rect**2 + h_rect**2)

        # Return diameter and label
        return diameter , int(classes[0])
    
def calculate_ppm(diameter,label):

    korean_currency = load_yaml_to_dict(MEASURMENTS_YAML_FILE)

    return diameter / korean_currency[label]["diameter_mm"]


if __name__ == "__main__":

    images = glob.glob(r'MeasureFood\test_images\*.jpg')

    for path in images:

        image = cv2.imread(path)

        # Get rectangle coordinates for the bill
        diameter,label = get_circle_or_rectangle(image,show=True)

        print('Pixels per mm :',calculate_ppm(diameter,label))

        # Display the result
        cv2.imshow('test',image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        