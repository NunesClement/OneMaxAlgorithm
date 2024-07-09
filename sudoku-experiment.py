import math
import time
import random
import numpy as np

size = 9
subSize = int(size / math.sqrt(size))
binarySize = math.ceil(math.log2(size))


def create_sudoku_grid():
    if size <= 0 or int(math.sqrt(size)) ** 2 != size:
        raise ValueError("The size must be a perfect square.")

    n = int(math.sqrt(size))

    grid = np.random.randint(1, size + 1, size=(size, size))

    return grid


def display_sudoku_grid(grid):
    size = len(grid)
    subSize = int(math.sqrt(size))

    for i in range(size):
        if i % subSize == 0 and i != 0:
            print("- " * (size * 2 + 1))

        for j in range(size):
            if j % subSize == 0 and j != 0:
                print("|", end=" ")

            print(f"{grid[i][j]:2}", end=" ")

        print("|")


def digitToBinary(digit):
    return bin(digit)[2:].zfill(binarySize)


def convertGridToBinary(grid):
    binaryGrid = [[digitToBinary(grid[i][j]) for j in range(size)] for i in range(size)]
    return binaryGrid


def get_sub_sudoku_grids(grid):
    subSize = int(size / math.sqrt(size))
    sub_grids = []

    for i in range(0, size, subSize):
        for j in range(0, size, subSize):
            sub_grids.append([i, j])

    return sub_grids


def sub_sudoku_grid_penalty(subgrid, grid):
    subItemX = subgrid[0]
    subItemY = subgrid[1]

    seen_numbers = set()
    redundancy_count = 0

    for k in range(subItemX, subItemX + subSize):
        for l in range(subItemY, subItemY + subSize):
            num = grid[k][l]
            if num in seen_numbers:
                redundancy_count += 1
            seen_numbers.add(num)

    # print(redundancy_count)
    return redundancy_count


def sudoku_line_penalty(grid):
    seen_numbers = set()
    redundancy_count = 0

    for i in range(grid[0], grid[0] + size):
        num = grid[i]
        if num in seen_numbers:
            redundancy_count += 1
        seen_numbers.add(num)

    return sudoku_line_penalty


sudokuGrid = create_sudoku_grid()

display_sudoku_grid(sudokuGrid)

for lines in range(0, size):
    sudoku_line_penalty(sudokuGrid)

# start_time = time.time()

# for i in range(0, 100000):
#     for subgrid in get_sub_sudoku_grids(sudokuGrid):
#         sub_sudoku_grid_penalty(subgrid, sudokuGrid)

# end_time = time.time()

# print(f"Execution time: {end_time - start_time:.6f} seconds")
