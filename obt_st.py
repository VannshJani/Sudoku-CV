import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image
from obt import solve_sudoku, animate_sudoku, show_final_board
import cv2
from imarr import img2arr

def show_animation(all_boards):
    show_final_board(all_boards[-1])
    

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


