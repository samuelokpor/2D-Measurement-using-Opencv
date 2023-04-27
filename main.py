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

# Threshold the image to obtain a binary image
_, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Select the largest contour
contour = max(contours, key=cv2.contourArea)

# Calculate the area of the contour
contour_area = cv2.contourArea(contour)

# Draw the contour on the resized image
cv2.drawContours(resized_img, [contour], 0, (0, 255, 0), 2)

# Convert the grayscale and thresholded images to color images
gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# Concatenate the grayscale, thresholded, and resized images horizontally
combined_img = np.hstack((gray_color, thresh_color, resized_img))

# Display the combined image and the contour area
cv2.putText(combined_img, "Contour Area: {:.2f}".format(contour_area), (10, height-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1)
cv2.imshow("Combined Image", combined_img)
cv2.waitKey(0)

