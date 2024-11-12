import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image
import cv2
import io
from io import BytesIO
import matplotlib.patches as patches
from skimage import io, morphology, img_as_bool, segmentation
from scipy import ndimage as ndi
import operator
import tensorflow as tf

def show_animation(all_boards):
    show_final_board(all_boards[-1])

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

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False

    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

# mrv = minimum remaining values herustic
def find_mrv(board):
    min_count = 10
    mrv_position = (-1, -1)
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                count = sum(is_valid(board, row, col, num) for num in range(1, 10))
                if count < min_count:
                    min_count = count
                    mrv_position = (row, col)
    return mrv_position

def solve_sudoku(board,all_boards):
    all_boards.append([row[:] for row in board])
    empty = find_mrv(board)
    if empty == (-1, -1):
        return True, all_boards

    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board,all_boards):
                return True, all_boards
            board[row][col] = 0

    return False, all_boards


def animate_sudoku(all_boards):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.set_xticks([i + 0.5 for i in range(9)], minor=True)
        ax.set_yticks([i + 0.5 for i in range(9)], minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.imshow([[5]*9]*9, cmap="Blues", vmin=0, vmax=9)
        
        for i in range(9):
            for j in range(9):
                num = all_boards[frame][i][j]
                if num != 0:
                    ax.text(j, i, str(num), ha="center", va="center", color="black")
        
        ax.set_xticks([])
        ax.set_yticks([])

    ani = animation.FuncAnimation(fig, update, frames=len(all_boards), interval=500, repeat=False)
    plt.show()

    

def show_final_board(grid):
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(10):
        linewidth = 2 if i % 3 == 0 else 1
        ax.plot([i, i], [0, 9], color='black', linewidth=linewidth)
        ax.plot([0, 9], [i, i], color='black', linewidth=linewidth)

    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                ax.text(j + 0.5, 8.5 - i, str(grid[i][j]), ha='center', va='center', fontsize=16)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    st.image(buf, use_column_width=True)

    # Close the plot to free memory
    plt.close(fig)
    
    

st.title("Sudoku Solver using Optimized Backtracking")

# Define the layout with two columns
left_column, right_column = st.columns(2)

with right_column:
    st.header("Image Input")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # display the uploaded image
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

with left_column:
    st.header("Output")

    if uploaded_file is None:
        st.markdown("<p style='color: red;'>Please select an input image to display the Sudoku animation.</p>", unsafe_allow_html=True)
    else:
        img = Image.open(uploaded_file)
        img_arr = np.array(img)
        board = img2arr(img_arr)
        board = np.array(board).T
        list_of_boards = []
        is_solved, all_boards = solve_sudoku(board,list_of_boards)
        if is_solved:
            show_animation(all_boards)
            st.write("Sudoku solved successfully!")
        else:
            show_animation(all_boards)
            st.write("No solution exists.")


