import math
import time
import random

size = 16
subSize = int(size / math.sqrt(size))
binarySize = math.ceil(math.log2(size))


import random
import math


def create_sudoku_grid():
    # Check if the size is a perfect square
    if size <= 0 or int(math.sqrt(size)) ** 2 != size:
        raise ValueError("The size must be a perfect square.")

    # Calculate the grid dimensions
    n = int(math.sqrt(size))

    # Initialize the grid with zeros
    grid = [[random.randint(1, size) for _ in range(size)] for _ in range(size)]

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

            print(
                f"{grid[i][j]:2}", end=" "
            )  # Print each number with a width of 2 characters

        print("|")


print(display_sudoku_grid(create_sudoku_grid()))


# print(create_sudoku_grid())


# def digitToBinary(digit):
#     return bin(digit)[2:].zfill(binarySize)


# def convertGridToBinary(grid):
#     binaryGrid = [[digitToBinary(grid[i][j]) for j in range(size)] for i in range(size)]
#     return binaryGrid


# # print(convertGridToBinary(create_sudoku_grid()))

# # print(binaryLengthNeeded(127))


# def sub_cube_analyzer():
#     subSize = int(size / math.sqrt(size))

#     for i in range(0, size, subSize):
#         for j in range(0, size, subSize):
#             print(str(i) + str(j))


# # start_time = time.time()

# # for i in range(0, 100000):
# #     sub_cube_analyzer(100)
# #     end_time = time.time()

# # print(f"Execution time1: {end_time - start_time:.6f} seconds")
