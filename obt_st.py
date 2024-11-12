import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image
import cv2

def show_animation(all_boards):
    show_final_board(all_boards[-1])
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io
from io import BytesIO
import numpy as np
from PIL import Image
import streamlit as st

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


