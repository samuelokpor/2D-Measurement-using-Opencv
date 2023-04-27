import cv2
import numpy as np

# Load the image
img = cv2.imread('./images/2.1.jpg')

# Define the desired width and height
width = 640
height = 480

# Resize the image to the desired size
resized_img = cv2.resize(img, (width, height))

# Convert the resized image to grayscale
gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Define the lower and upper thresholds for circular contours
circular_thresh_lower = 1
circular_thresh_upper = 255

# Define the lower and upper thresholds for rectangular contours
rect_thresh_lower = 10
rect_thresh_upper = 150

# Define a function to find contours based on the given threshold value
def find_contours(circular_thresh, rect_thresh):
    # Threshold the image to obtain a binary image
    _, thresh = cv2.threshold(gray, max(circular_thresh, rect_thresh), 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty list to store the circular and rectangular contours
    circular_contours = []
    rect_contours = []
    
    # Loop over all the contours found
    for contour in contours:
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(contour, True)
        
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
        
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # Check if the contour is circular
        if len(approx) > 7:
            (x,y), radius = cv2.minEnclosingCircle(contour)
            if radius > circular_thresh_lower and radius < circular_thresh_upper:
                circular_contours.append(contour)
        
        # Check if the contour is rectangular
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if aspect_ratio >= 0.5 and aspect_ratio <= 1.5:
                if w > rect_thresh_lower and w < rect_thresh_upper and h > rect_thresh_lower and h < rect_thresh_upper:
                    rect_contours.append(contour)
    
    return circular_contours, rect_contours

# Define the update_image function
def update_image(dummy):
    circular_thresh = cv2.getTrackbarPos("Circular Threshold", "Threshold Image")
    rect_thresh = cv2.getTrackbarPos("Rectangular Threshold", "Threshold Image")
    circular_contours, rect_contours = find_contours(circular_thresh, rect_thresh)
    
    # Draw the circular contours on the resized image
    for contour in circular_contours:

        circular_img = resized_img.copy()
        cv2.drawContours(circular_img, circular_contours, -1, (0, 255, 0), 2)
        area = cv2.contourArea(contour)
        #cv2.putText(circular_img, f"{area:.2f}", tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(circular_img, "Contour Area: {:.2f}".format(area), (10, height-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1)
    
    # Draw the rectangular contours on the resized image
    for contour in circular_contours:

        rect_img = resized_img.copy()
        cv2.drawContours(rect_img, rect_contours, -1, (0, 0, 255), 2)
        area = cv2.contourArea(contour)
        #cv2.putText(rect_img, f"{area:.2f}", tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(rect_img, "Contour Area: {:.2f}".format(area), (10, height-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1)
    
    # Convert the grayscale image to a color image
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Concatenate the grayscale,circular, rectangular, and resized images horizontally
    combined_img = np.hstack((gray_color, circular_img, rect_img, resized_img))
    # Display the combined image
    cv2.imshow("Combined Image", combined_img)

#Create a window to display the thresholded image)
cv2.namedWindow("Threshold Image", cv2.WINDOW_NORMAL)

#Create trackbars to adjust the threshold values
cv2.createTrackbar("Circular Threshold", "Threshold Image", 1, 255, update_image)
cv2.createTrackbar("Rectangular Threshold", "Threshold Image", 10, 150, update_image)

#Call the update_image function to display the initial image
update_image(0)
#Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows

