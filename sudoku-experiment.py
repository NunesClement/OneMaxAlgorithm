import math
import time


def create_sudoku_grid(size):
    if size <= 0 or int(size**0.5) ** 2 != size:
        raise ValueError("The size must be a perfect square.")

    # Initialize the grid with zeros
    grid = [[0 for _ in range(size)] for _ in range(size)]

    return grid


def display_sudoku_grid(grid):
    size = len(grid)
    subSize = int(size / math.sqrt(size))

    for i in range(size):
        if i % subSize == 0:
            print("- " * (size * 2 + 3))

        for j in range(size):
            if j % subSize == 0:
                print("|", end=" ")

            print(grid[i][j], end=" ")

        print("|")


# print(display_sudoku_grid(create_sudoku_grid(9)))


def binaryLengthNeeded(size):
    return math.ceil(math.log2(size))


def digitToBinary(digit):
    return bin(digit)[2:]


print(digitToBinary(15))
print(digitToBinary(31))
print(digitToBinary(63))
print(digitToBinary(127))

print(len(digitToBinary(127)))
print(binaryLengthNeeded(127))


def sub_cube_analyzer(size):
    subSize = int(size / math.sqrt(size))

    for i in range(0, size, subSize):
        for j in range(0, size, subSize):
            print(str(i) + str(j))


# start_time = time.time()

# for i in range(0, 100000):
#     sub_cube_analyzer(100)
#     end_time = time.time()

# print(f"Execution time1: {end_time - start_time:.6f} seconds")
