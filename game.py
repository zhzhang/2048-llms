import random
from enum import Enum

import numpy as np


class Move(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class IllegalMove(Exception):
    pass


def init_board():
    board = np.zeros((4, 4), dtype=int)
    add_random_tile(board)
    add_random_tile(board)
    return board


def add_random_tile(board):
    value = 2 if random.random() < 0.9 else 4
    empty = np.transpose((board == 0).nonzero())
    if len(empty) == 0:
        return  # No empty cells available
    cell_index = random.choice(empty)
    board[cell_index[0], cell_index[1]] = value


def move(board, move: Move):
    # Create a copy of the board to avoid modifying the original
    working_board = board.copy()

    if move == Move.RIGHT:
        working_board = np.flip(working_board, 1)
    elif move == Move.UP:
        working_board = np.transpose(working_board)
    elif move == Move.DOWN:
        working_board = np.flip(np.transpose(working_board), 1)

    new_board = []

    for i in range(4):
        new = []
        last_val = 0
        for j in range(4):
            val = working_board[i, j]
            if val == 0:
                continue
            if len(new) == 0:
                new.append(val)
                last_val = val
            else:
                if val == last_val:
                    new[-1] *= 2
                    last_val = 0
                else:
                    new.append(val)
                    last_val = val
        while len(new) < 4:
            new.append(0)
        new_board.append(new)

    new_board = np.array(new_board)

    # Restore the original orientation
    if move == Move.RIGHT:
        new_board = np.flip(new_board, 1)
    elif move == Move.UP:
        new_board = np.transpose(new_board)
    elif move == Move.DOWN:
        new_board = np.transpose(np.flip(new_board, 1))

    # Check if the move actually changed the board
    if (board == new_board).all():
        raise IllegalMove()

    add_random_tile(new_board)
    return new_board


def print_board(board):
    out = ""
    for i in range(4):
        for j in range(4):
            num = str(board[i, j])
            while len(num) < 4:
                num = " " + num
            out += num + "  "
        out += "\n"
    print(out)
    return out


if __name__ == "__main__":
    board = init_board()
    print_board(board)
    while True:
        # Wait for user input:
        print("Next move:")
        key = input()
        if key == "w":
            board = move(board, Move.UP)
        elif key == "s":
            board = move(board, Move.DOWN)
        elif key == "a":
            board = move(board, Move.LEFT)
        elif key == "d":
            board = move(board, Move.RIGHT)
        else:
            break
        print_board(board)
