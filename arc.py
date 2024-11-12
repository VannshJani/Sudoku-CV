import streamlit as st
import numpy as np
import random
from copy import deepcopy

def calculate_domains(board):  # forwards checking
    domains = [[set() for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            if board[i, j] != 0:
                domains[i][j] = {-1}
            else:
                possible_values = set(range(1, 10))
                possible_values -= set(board[i, :])
                possible_values -= set(board[:, j])
                start_row, start_col = 3 * (i // 3), 3 * (j // 3)
                for r in range(start_row, start_row + 3):
                    for c in range(start_col, start_col + 3):
                        possible_values.discard(board[r, c])
                domains[i][j] = possible_values
    return domains

# initialize the domains
def initialize_domains(board):
    domains = [[set() for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            if board[i, j] != 0:
                domains[i][j] = {board[i, j]}
            else:
                possible_values = set(range(1, 10))
                domains[i][j] = possible_values
    return domains

def load_random_sudoku_board(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    random_line = random.choice(lines).strip()
    parts = random_line.split()
    board_values_str = parts[1]
    board_values = [int(char) for char in board_values_str]
    board = np.array(board_values).reshape((9, 9))
    domains = initialize_domains(board)
    return board, domains

def display_sudoku_board(board_values, domains, played_moves, arrow_start=None, arrow_end=None):
    cell_size = 50
    board_html = """
    <style>
        .sudoku-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 45px;
            position: relative;  /* Positioning context for absolute elements */
        }
        table {
            border-collapse: collapse;
            border: 2px solid white;
            position: relative;  /* Make table relative to be above SVG */
            z-index: 2;  /* Ensure table is above SVG */
        }
        th, td {
            border: 1px solid white;
            padding: 15px;
            text-align: center;
            width: 50px;
            height: 50px;
            font-size: 1.2em;
            font-weight: bold;
            position: relative; /* Ensure cell is above SVG */
            z-index: 3;
        }
        td {
            color: white; /* Default text color */
        }
        .new-move {
            color: green; /* New move text color */
        }
        td .domain-hover {
            visibility: hidden; 
            position: absolute; 
            top: -10px; 
            left: 50%; 
            transform: translate(-50%, -100%);
            background-color: #f9f9f9; 
            padding: 5px; 
            border-radius: 5px; 
            box-shadow: 0px 0px 5px #aaa;
            white-space: nowrap;
            color: black;
            z-index: 4;  /* Ensure hover box is above SVG */
        }
        td:hover .domain-hover {
            visibility: visible;
        }
    </style>
    <div class="sudoku-container">
    <table>
    """

    for i in range(9):
        board_html += "<tr>"
        for j in range(9):
            cell = board_values[i][j]
            cell_display = cell if cell != 0 else "&nbsp;"  # Show value or empty cell
            domain_display = ", ".join(map(str, sorted(domains[i][j]))) if cell == 0 else str(cell)
            new_move_class = "new-move" if (i, j) in played_moves else ""
            board_html += f"<td class='{new_move_class}'>{cell_display}<span class='domain-hover'>{domain_display}</span></td>"
        board_html += "</tr>"
    
    board_html += "</table>"
    # 135,0,135,-12
    # Add SVG arrows if specified
    if arrow_start[0] >=0:
        start_x = arrow_start[1] * cell_size + cell_size//2 + 120 # Adjust for padding
        start_y = arrow_start[0] * cell_size + cell_size//2 +10 # Adjust for padding
        end_x = arrow_end[1] * cell_size + cell_size//2  + 121
        end_y = arrow_end[0] * cell_size + cell_size//2 

        # Define control points for the curve
        control_x = (start_x + end_x) / 2
        control_y = start_y - 40  # Control point to create an upward curve

        # SVG for the curved arrow
        board_html += f"""
        <svg width="100%" height="100%" style="position:absolute; top: 0; left: 0; z-index: 1; pointer-events: none;">
            <path d="M {start_x} {start_y} Q {control_x} {control_y}, {end_x} {end_y}" 
                  style="stroke:red; stroke-width: 2; fill: none; marker-end: url(#arrowhead);" />
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                        refX="0" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="red" />
                </marker>
            </defs>
        </svg>
        """
    
    board_html += "</div>"

    st.markdown(board_html, unsafe_allow_html=True)





def update_domains(board, domains):
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                possible_values = set(range(1, 10))
                for k in range(9):
                    possible_values.discard(board[i, k])
                    possible_values.discard(board[k, j])
                bi, bj = 3 * (i // 3), 3 * (j // 3)
                for r in range(3):
                    for c in range(3):
                        possible_values.discard(board[bi + r, bj + c])
                domains[i][j] = possible_values

def arc_consistency_check(domains, x1, y1, x2, y2):

    domX1 = domains[x1][y1]
    domX2 = domains[x2][y2]
    if len(domX1) == 0:
        return True, []
    revised = False
    val = []
    for x in domX1:
        isconsistent = False
        for y in domX2:
            if x != y:
                isconsistent = True
                break
        if not isconsistent:
            val.append(x)
            revised = True
    for x in val:
        domX1.discard(x)
    return revised, val

def play_move_func(board, domains, played_moves, i, j, value):
    if board[i, j] == 0 and value in domains[i][j]:
        board[i, j] = value
        played_moves.add((i, j))  # Add the moved cell to the played moves
        # update_domains(board, domains)
        domains[i][j] = {value} 
        return True
    elif (i,j) in played_moves and value in domains[i][j]:
        board[i, j] = value
        return True

    return False

def get_all_arcs(domains):
    constraints = {}
    for row in range(9):
        for col in range(9):
            cell = (row, col)
            constraints[str(cell)] = []
            # Row constraints
            constraints[str(cell)].extend([(row, c) for c in range(9) if c != col])
            # Column constraints
            constraints[str(cell)].extend([(r, col) for r in range(9) if r != row])
            # Subgrid constraints
            subgrid_row = (row // 3) * 3
            subgrid_col = (col // 3) * 3
            constraints[str(cell)].extend([(sr, sc) for sr in range(subgrid_row, subgrid_row + 3)
                                       for sc in range(subgrid_col, subgrid_col + 3)
                                       if (sr, sc) != cell])

    # Create the set S containing all arcs (Xi, Xj)
    # S = [((int(Xi[0]),int(Xi[1])), Xj) for Xi in constraints for Xj in constraints[Xi] if Xi in domains]
    S = []
    for Xi in constraints:
        for Xj in constraints[Xi]:
            S.append(((int(Xi[1]),int(Xi[4])), Xj))
    
    return S,constraints

def AC3(domains, arcs,constraints):
    is_consistent = True
    updated_domains_every_step = []
    all_arcs_ac3 = []
    while arcs:
        (Xi, Xj) = arcs.pop(0)
        revised, val = arc_consistency_check(domains, Xi[0],Xi[1], Xj[0],Xj[1])
        updated_domains_every_step.append(deepcopy(domains))
        all_arcs_ac3.append(((Xi[0],Xi[1]),(Xj[0],Xj[1])))
        if revised:
            if len(domains[Xi[0]][Xi[1]]) == 0:
                is_consistent = False
                return is_consistent, updated_domains_every_step, all_arcs_ac3, Xi, Xj
            for Xk in constraints:
                if Xi in constraints[Xk]:
                    arcs.append(((int(Xk[1]),int(Xk[4])), Xi))

    return is_consistent, updated_domains_every_step, all_arcs_ac3,None,None



st.title("Arc Consistency Using Sudoku")

# Initialize the board and domains
if 'board' not in st.session_state or 'domains' not in st.session_state:
    st.session_state.board, st.session_state.domains = load_random_sudoku_board("sudokus.txt")
    st.session_state.played_moves = set()  # Track played moves
# st.write(st.session_state.domains[0][0])
if 'initial_board' not in st.session_state:
    st.session_state.initial_board = st.session_state.board.copy()
if 'initial_domains' not in st.session_state:
    st.session_state.initial_domains = deepcopy(st.session_state.domains)


if 'i' not in st.session_state:
    st.session_state.i = 0
if 'j' not in st.session_state:
    st.session_state.j = 0
if 'value' not in st.session_state:
    st.session_state.value = 1
if 'flag1' not in st.session_state:
    st.session_state.flag1 = False
if 'flag2' not in st.session_state:
    st.session_state.flag2 = False
if 'x1' not in st.session_state:
    st.session_state.x1 = 0
if 'y1' not in st.session_state:
    st.session_state.y1 = 0
if 'x2' not in st.session_state:
    st.session_state.x2 = 8
if 'y2' not in st.session_state:
    st.session_state.y2 = 0
if 'flag3' not in st.session_state:
    st.session_state.flag3 = False
if 'all_arcs_ac3' not in st.session_state:
    st.session_state.all_arcs_ac3 = []
if 'all_domains_ac3' not in st.session_state:
    st.session_state.all_domains_ac3 = []
if 'is_consistent' not in st.session_state:
    st.session_state.is_consistent = True
if "play_initial" not in st.session_state:
    st.session_state.play_initial = False
if 'index' not in st.session_state:
    st.session_state.index = 0
if 'flag4' not in st.session_state:
    st.session_state.flag4 = False



# Add reset and change board button in columns
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    reset = st.button("Reset Board")
    change_board = st.button("Change Board")
with col2:
    play_move = st.button("Play Move")
    arc_consistency = st.button("Check Arc Consistency")
with col3:
    one_step = st.button("Run AC3 Step by Step") 
    ac3 = st.button("Run AC3")
    
# st.write(st.session_state.initial_domains[0][0],1)
# st.write(st.session_state.domains[0][0],2)

if reset:
    st.session_state.flag1 = False
    st.session_state.flag2 = False
    st.session_state.flag3 = False
    st.session_state.flag4 = False
    st.session_state.index = 0
    # st.session_state.play_initial = False
    st.session_state.board = st.session_state.initial_board
    st.session_state.domains = initialize_domains(st.session_state.board)
    st.session_state.played_moves = set()  # Reset played moves
# st.write(st.session_state.domains[0][0],"after reset")

if change_board:
    st.session_state.flag1 = False
    st.session_state.flag2 = False
    st.session_state.flag3 = False
    st.session_state.flag4 = False
    st.session_state.index = 0
    st.session_state.play_initial = False
    st.session_state.board, st.session_state.domains = load_random_sudoku_board("sudokus.txt")
    st.session_state.initial_board = st.session_state.board.copy()
    st.session_state.initial_domains = deepcopy(st.session_state.domains)
    st.session_state.played_moves = set()  # Reset played moves

# Run AC3
if not st.session_state.play_initial:
    arcs,constraints = get_all_arcs(st.session_state.domains)
    st.session_state.is_consistent, st.session_state.all_domains_ac3, st.session_state.all_arcs_ac3,_,__ = AC3(deepcopy(st.session_state.domains), arcs,constraints)
    st.session_state.play_initial = True



if (play_move or st.session_state.flag1):
    st.session_state.flag2 = False
    st.session_state.flag3 = False
    st.session_state.flag4 = False
if (play_move or st.session_state.flag1) and not (st.session_state.flag2 or arc_consistency) and not (st.session_state.flag3 or one_step) and not (st.session_state.flag4 or ac3):
    st.session_state.flag2 = False
    st.session_state.flag1 = True
    st.session_state.flag3 = False
    st.session_state.flag4 = False
    st.session_state.i = st.number_input("Row (0-8)", 0, 8, 0)
    st.session_state.j = st.number_input("Column (0-8)", 0, 8, 0)
    st.session_state.value = st.number_input("Value (1-9)", 1, 9, 1)
    play = st.button("Play")
    if play:
        if play_move_func(st.session_state.board, st.session_state.domains, st.session_state.played_moves, st.session_state.i, st.session_state.j, st.session_state.value):
            st.write(f"Move played: ({st.session_state.i}, {st.session_state.j}) = {st.session_state.value}")
        else:
            st.write("Invalid move! Please try again.")

if arc_consistency or st.session_state.flag2:
    st.session_state.flag1 = False
    st.session_state.flag3 = False
    st.session_state.flag4 = False

if arc_consistency or st.session_state.flag2 and not (st.session_state.flag1 or play_move) and not (st.session_state.flag3 or one_step) and not (st.session_state.flag4 or ac3):
    st.session_state.flag1 = False
    st.session_state.flag2 = True
    st.session_state.flag3 = False
    st.session_state.flag4 = False
    st.write("Checking Arc Consistency for X1 -> X2")
    # select X1 and X2
    col1, col2 = st.columns([1, 1])
    with col1:
        st.session_state.x1 = st.number_input("Row X1 (0-8)", 0, 8, 0)
        st.session_state.y1 = st.number_input("Column X1 (0-8)", 0, 8, 0)
    with col2:
        st.session_state.x2 = st.number_input("Row X2 (0-8)", 0, 8, 1)
        st.session_state.y2 = st.number_input("Column X2 (0-8)", 0, 8, 1)
    
    # display_sudoku_board(st.session_state.board, st.session_state.domains, st.session_state.played_moves,(st.session_state.x1,st.session_state.y1),(st.session_state.x2,st.session_state.y2))
    
    check = st.button("Check")
    if check:
        revised, val = arc_consistency_check(st.session_state.domains, st.session_state.x1, st.session_state.y1, st.session_state.x2, st.session_state.y2)
        if revised:
            if len(val) == 1:
                val = val[0]
                st.write(f":red[This arc is not consistent. Removed value: {val} from domain of X1]")
            elif len(val) > 1:
                st.write(f":red[This arc is not consistent. Removed values: {val} from domain of X1]")
            else:
                st.write(f":red[This arc is not consistent. Domain of X1 is empty.]")
        else:
            st.write(":green[This arc is consistent.]")


if one_step or st.session_state.flag3:
    st.session_state.flag1 = False
    st.session_state.flag2 = False
    if st.session_state.flag4:
        st.write("AC3 already completed")
        st.session_state.flag3 = False
        st.session_state.flag4 = False

if one_step or st.session_state.flag3 and not (st.session_state.flag1 or play_move) and not (st.session_state.flag2 or arc_consistency) and not (st.session_state.flag4 or ac3):
    st.session_state.flag3 = True
    st.session_state.flag1 = False
    st.session_state.flag2 = False
    st.session_state.flag4 = False
    st.write("Running AC3 Step by Step")
    # st.session_state.domains = initialize_domains(st.session_state.board)
    if st.session_state.index < len(st.session_state.all_arcs_ac3):
        (st.session_state.x1,st.session_state.y1),(st.session_state.x2,st.session_state.y2) = st.session_state.all_arcs_ac3[st.session_state.index]
        
    col1,col2 = st.columns([1,1])
    with col1:
        st.write(f"Check arc consistency of ({st.session_state.x1},{st.session_state.y1}) -> ({st.session_state.x2},{st.session_state.y2})")
        chec2 = st.button("Check")
    with col2:
        st.write("Run next step")
        next = st.button("Next")
    
    if chec2:
        revised, val = arc_consistency_check(st.session_state.domains, st.session_state.x1, st.session_state.y1, st.session_state.x2, st.session_state.y2)
        if revised:
            if len(val) == 1:
                val = val[0]
                st.write(f":red[This arc is not consistent. Removed value: {val} from domain of X1]")
            elif len(val) > 1:
                st.write(f":red[This arc is not consistent. Removed values: {val} from domain of X1]")
            else:
                st.write(f":red[This arc is not consistent. Domain of X1 is empty.]")
        else:
            st.write(":green[This arc is consistent.]")
    if next:
        st.session_state.index += 1
        # rerun script
        if st.session_state.index >= len(st.session_state.all_arcs_ac3):
            st.session_state.flag3 = False
            st.session_state.index = 0
            st.write("AC3 completed.")
            st.session_state.all_arcs_ac3 = []
            st.session_state.all_domains_ac3 = []
            st.session_state.play_initial = False
        st.rerun()
        

if ac3 or st.session_state.flag4:
    st.session_state.flag1 = False
    st.session_state.flag2 = False
    st.session_state.flag3 = False

if ac3 or st.session_state.flag4 and not (st.session_state.flag1 or play_move) and not (st.session_state.flag2 or arc_consistency) and not (st.session_state.flag3 or one_step):
    st.session_state.flag4 = True
    st.session_state.flag1 = False
    st.session_state.flag2 = False
    st.session_state.flag3 = False
    st.write("Running AC3")
    arcs,constraints = get_all_arcs(st.session_state.domains)
    st.session_state.is_consistent, st.session_state.all_domains_ac3, st.session_state.all_arcs_ac3,Xi,Xj = AC3(st.session_state.domains, arcs,constraints)
    if st.session_state.is_consistent:
        st.write(":green[The Sudoku is Arc Consistent.]")
    else:
        st.write(":red[The Sudoku is not Arc Consistent.]")
        st.write(f"The Arc  ({Xi[0]},{Xi[1]}) -> ({Xj[0]},{Xj[1]}) is not consistent.")
        st.session_state.x1 = Xi[0]
        st.session_state.y1 = Xi[1]
        st.session_state.x2 = Xj[0]
        st.session_state.y2 = Xj[1]
    



if not (arc_consistency or st.session_state.flag2) and not (one_step or st.session_state.flag3) and not (ac3 or st.session_state.flag4):
    st.session_state.x1 = -1
if st.session_state.flag4 == True:
    if st.session_state.is_consistent:
        st.session_state.x1 = -1


# st.write(st.session_state.initial_domains[0][0],3)
# st.write(st.session_state.domains[0][0],4)
display_sudoku_board(st.session_state.board, st.session_state.domains, st.session_state.played_moves,(st.session_state.x1,st.session_state.y1),(st.session_state.x2,st.session_state.y2))

