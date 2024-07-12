import math
import time
import random
import numpy as np

# global_total_case = 16
# global_sudoku_n = 4
# global_subSize = 2
# global_binarySize = 2

# URG: Should born auto
# global_total_case = 81
# global_sudoku_n = 9
# global_subSize = 3
# global_binarySize = 4

global_total_case = 256
global_sudoku_n = 16
global_subSize = 4
global_binarySize = 4


# URG: to be tested
def set_global_size(size, subSize, binarySize):
    global global_total_case, global_sudoku_n, global_binarySize

    global_total_case = size
    global_sudoku_n = subSize
    global_binarySize = binarySize


def create_sudoku_grid():
    if global_sudoku_n <= 0 or int(math.sqrt(global_sudoku_n)) ** 2 != global_sudoku_n:
        raise ValueError("The size must be a perfect square.")

    n = int(math.sqrt(global_sudoku_n))

    grid = np.random.randint(
        1, global_sudoku_n + 1, size=(global_sudoku_n, global_sudoku_n)
    )

    return grid


def display_sudoku_grid(grid):
    for i in range(global_sudoku_n):
        if i % global_subSize == 0 and i != 0:
            print("- " * (global_sudoku_n * 2 + 1))

        for j in range(global_sudoku_n):
            if j % global_subSize == 0 and j != 0:
                print("|", end=" ")
            print(f"{grid[i][j]+1:2}", end=" ")

        print("|")


def digitToBinary(digit):
    return bin(digit)[2:].zfill(global_binarySize)


def convertGridToBinary(grid):
    binaryString = "".join(
        digitToBinary(grid[i][j])
        for i in range(global_sudoku_n)
        for j in range(global_sudoku_n)
    )

    return binaryString


# URG: test performance and fix
def convertBinaryToGrid(binaryString: str):
    if len(binaryString) % global_sudoku_n != 0:
        raise ValueError("Length of binaryString is not a multiple of global_sudoku_n")

    int_values = np.zeros(global_total_case, dtype=int)
    # print(int_values)

    for idx in range(global_total_case):
        start = idx * global_binarySize
        end = start + global_binarySize

        int_values[idx] = int(
            "".join(str(bit) for bit in binaryString[start:end][:global_binarySize]), 2
        )
        # print(int_values[idx])

    grid = int_values.reshape(global_sudoku_n, global_sudoku_n)
    # print(grid)
    return grid


def get_sub_sudoku_grids(grid):
    sub_grids = []

    for i in range(0, global_sudoku_n, global_subSize):
        for j in range(0, global_sudoku_n, global_subSize):
            sub_grids.append([i, j])

    return sub_grids


def sub_sudoku_grid_penalty(subgrid, grid):
    subItemX = subgrid[0]
    subItemY = subgrid[1]

    seen_numbers = set()
    redundancy_count = 0

    # print("subItemX")
    # print(subgrid)
    # print(subItemX)
    # print(subItemX + global_sudoku_n)
    # print(subItemY + global_sudoku_n)

    for k in range(subItemX, subItemX + global_subSize):
        for l in range(subItemY, subItemY + global_subSize):
            # print(k, l)
            num = grid[k][l]
            if num in seen_numbers:
                redundancy_count += 1
            seen_numbers.add(num)
    #     print("---")
    # print("######")

    return redundancy_count


def sudoku_line_penalty(grid):
    total_redundancy_count = 0
    for i in range(0, global_sudoku_n):
        seen_numbers = set()

        for j in grid[i]:
            if j in seen_numbers:
                total_redundancy_count += 1
            seen_numbers.add(j)
    return total_redundancy_count


def sudoku_column_penalty(grid):
    total_redundancy_count = 0
    for i in range(0, global_sudoku_n):
        seen_numbers = set()

        for j in range(0, global_sudoku_n):
            num = grid[j][i]
            if num in seen_numbers:
                total_redundancy_count += 1
            seen_numbers.add(num)
    return total_redundancy_count


# URG: check a better way to calculate penalty
def calculate_fitness(binary):
    grid = convertBinaryToGrid(binary)
    sub_grids = get_sub_sudoku_grids(grid)

    total_penalty = 0

    # for sub_grid in sub_grids:
    #     total_penalty += sub_sudoku_grid_penalty(sub_grid, grid)

    total_penalty += sudoku_line_penalty(grid)
    total_penalty += sudoku_column_penalty(grid)

    return -total_penalty


# binaries = create_sudoku_grid()
# converted = convertGridToBinary(binaries)

# start_time = time.time()
# for i in range(0, 3000000):
#     convertBinaryToGrid(binaries)
# end_time = time.time()

# print(f"Execution time: {end_time - start_time:.6f} seconds")
