import math
import time
import random
import numpy as np

# URG: display the sudoku at the end


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
    binaryString = "".join(
        digitToBinary(grid[i][j]) for i in range(size) for j in range(size)
    )

    return binaryString


# URG: test performance
def convertBinaryToGrid(intList):
    num_elements = size * size
    binary_strings = [format(x, f"0{size}b") for x in intList]
    binaryString = "".join(binary_strings)

    int_values = np.zeros(num_elements, dtype=int)

    for idx in range(num_elements):
        start = idx * size
        end = start + size
        int_values[idx] = int(binaryString[start:end], 2)

    grid = int_values.reshape(size, size)
    return grid


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
    total_redundancy_count = 0
    for i in range(0, size):
        seen_numbers = set()

        for j in grid[i]:
            if j in seen_numbers:
                total_redundancy_count += 1
            seen_numbers.add(j)
    return total_redundancy_count


def sudoku_column_penalty(grid):
    total_redundancy_count = 0
    for i in range(0, size):
        seen_numbers = set()

        for j in range(0, size):
            num = grid[j][i]
            if num in seen_numbers:
                total_redundancy_count += 1
            seen_numbers.add(num)
    return total_redundancy_count


def calculate_fitness(binary):
    grid = convertBinaryToGrid(binary)
    sub_grids = get_sub_sudoku_grids(grid)

    total_penalty = 0
    for sub_grid in sub_grids:
        total_penalty += sub_sudoku_grid_penalty(sub_grid, grid)

    total_penalty += sudoku_line_penalty(grid)
    total_penalty += sudoku_column_penalty(grid)

    print(grid)

    return total_penalty


# start_time = time.time()
# for i in range(0, 3000000):
#     convertBinaryToGrid(binaries)
# end_time = time.time()

# print(f"Execution time: {end_time - start_time:.6f} seconds")
