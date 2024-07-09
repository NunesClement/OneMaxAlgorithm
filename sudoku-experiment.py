import math
import time
import random
import numpy as np

size = 9
subSize = int(size / math.sqrt(size))
binarySize = math.ceil(math.log2(size))


def create_sudoku_grid():
    # Check if the size is a perfect square
    if size <= 0 or int(math.sqrt(size)) ** 2 != size:
        raise ValueError("The size must be a perfect square.")

    # Calculate the grid dimensions
    n = int(math.sqrt(size))

    # Initialize the grid with random integers from 1 to size (inclusive)
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


# start_time = time.time()

# for i in range(0, 200):
#     create_sudoku_grid2()
#     end_time = time.time()

# print(f"Execution time2: {end_time - start_time:.6f} seconds")


def digitToBinary(digit):
    return bin(digit)[2:].zfill(binarySize)


def convertGridToBinary(grid):
    binaryGrid = [[digitToBinary(grid[i][j]) for j in range(size)] for i in range(size)]
    return binaryGrid


def sub_sudoku_grid():
    subSize = int(size / math.sqrt(size))

    for i in range(0, size, subSize):
        for j in range(0, size, subSize):
            print(str(i) + " " + str(j))


print(sub_sudoku_grid())

# start_time = time.time()

# for i in range(0, 100000):
#     convertGridToBinary(create_sudoku_grid())
#     end_time = time.time()

# print(f"Execution time2: {end_time - start_time:.6f} seconds")
