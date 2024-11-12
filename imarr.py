import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, morphology, img_as_bool, segmentation
from scipy import ndimage as ndi
import operator
import tensorflow as tf


# Loading the trained model
model = tf.keras.models.load_model("model.keras")

# Function to predict the digit
def predict_digit(digit_img):
    """
    This function predicts the digit in the image.
    The steps involved are:
    1. Resize the image.
    2. Predict the digit.
    
    Parameters:
    digit_img:
        np.ndarray
        The digit image.
        
    Returns:
        int
        The predicted digit.
        """
    img = cv2.resize(digit_img, (28, 28))
    img = np.array(img).reshape(-1, 28, 28, 1)
    predictions = model.predict(img)
    return np.argmax(predictions)

# Function to predict the sudoku
def predict_sudoku(digits: list):
  """
  This function predicts the sudoku from the digits.
  The steps involved are:
  1. From all the digits, predict the digit and append it to the sudoku array.
  
  Parameters:
  digits:
      list
      The digits in the image.
        
    Returns:
        list
        The predicted sudoku."""
  sudoku = []
  for digit in digits:
    if np.sum(digit) > 255*30:
      sudoku.append(predict_digit(digit))
    else:
      sudoku.append(0)
  return sudoku

# Function to get the sudoku from the image


def img2arr(img):
    """
    This function predicts the sudoku from the image.
    The steps involved are:
    1. Convert the image to grayscale.
    2. Apply Gaussian Blur to the image.
    3. Apply Adaptive Thresholding to the image.
    4. Canny Edge Detection.
    5. Find the main contour.
    6. Crop the main contour.
    7. Divide the cropped image into 81 cells and subplot.
    8. For every cell, plot the cell with its largets contour, remove border and plot the digit in Grayscale. If no contour is present, keep a white image in grayscale.
    9. Predict the sudoku.
    """
    # Read the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to the image
    x, y = max(img.shape[0]//200, 3), max(img.shape[1]//200, 3)
    blurred = cv2.GaussianBlur(gray, (x+(x+1)%2, y+(y+1)%2), 0)

    # Apply Adaptive Thresholding to the image
    thresholded_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)

    # Canny Edge Detection
    edges = cv2.Canny(thresholded_img, 50, 150)

    # Find the main contour
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get the main contour
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)

    # Crop the main contour
    cropped_img = img[y:y+h, x:x+w]

    # Divide the cropped image into 81 cells and subplot
    cell_height, cell_width = h//9, w//9

    # For every cell, plot the cell with its largets contour, remove border and plot the digit in Grayscale. If no contour is present, keep a white image in grayscale
    digits = []
    for i in range(9):
        for j in range(9):
            cell = cropped_img[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            _, thresholded_cell = cv2.threshold(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            digit = thresholded_cell[y:y+h, x:x+w]
            # Crop the border
            digit = digit[3:-3, 3:-3]
            # Create a buffer of 5 pixels   
            digit = cv2.copyMakeBorder(digit, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # Invert the image
            digit = cv2.bitwise_not(digit)
            # Resize the image
            digit = cv2.resize(digit, (28, 28))
            digits.append(digit)

    # Convert the digits to numpy array
    digits = np.array(digits)

    # Predict the sudoku
    sudoku = predict_sudoku(digits)
    sudoku = np.array(sudoku).reshape(9,9)
    sudoku = sudoku.T
    sudoku = sudoku.tolist()
    return sudoku