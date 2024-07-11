import math
import time
import random
import numpy as np

# URG: display the sudoku at the end
# URG: test sudoku with a 9x9 grid
global_size = 0
global_subSize = 0
global_binarySize = 0


# URG: to be tested
def set_global_size(size, subSize, binarySize):
    global global_size, global_subSize, global_binarySize

    global_size = size
    global_subSize = subSize
    global_binarySize = binarySize


def create_sudoku_grid():
    if global_subSize <= 0 or int(math.sqrt(global_subSize)) ** 2 != global_subSize:
        raise ValueError("The size must be a perfect square.")

    n = int(math.sqrt(global_subSize))

    grid = np.random.randint(
        1, global_subSize + 1, size=(global_subSize, global_subSize)
    )

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
    return bin(digit)[2:].zfill(global_binarySize)


def convertGridToBinary(grid):
    binaryString = "".join(
        digitToBinary(grid[i][j])
        for i in range(global_subSize)
        for j in range(global_subSize)
    )

    return binaryString


# URG: test performance and fix
def convertBinaryToGrid(binaryString: str):
    if len(binaryString) % global_subSize != 0:
        raise ValueError("Length of binaryString is not a multiple of global_subSize")

    int_values = np.zeros(global_size, dtype=int)

    for idx in range(global_size):
        start = idx * global_binarySize
        end = start + global_binarySize

        int_values[idx] = int(binaryString[start:end], 2)

    grid = int_values.reshape(global_subSize, global_subSize)
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


def calculate_fitness(binary, givenSize):
    print("#######")
    size = givenSize**2
    subSize = int(size / math.sqrt(size))
    binarySize = math.ceil(math.log2(subSize))
    print(size)
    print(subSize)
    print(binarySize)
    print(size * binarySize)

    set_global_size(size, subSize, binarySize)
    print("here")

    grid = convertBinaryToGrid(binary)
    print("grid")
    print(grid)

    sub_grids = get_sub_sudoku_grids(grid)
    print("sub_grids")
    print(sub_grids)

    total_penalty = 0
    for sub_grid in sub_grids:
        total_penalty += sub_sudoku_grid_penalty(sub_grid, grid)

    total_penalty += sudoku_line_penalty(grid)
    total_penalty += sudoku_column_penalty(grid)

    return total_penalty


set_global_size(81, 9, 4)

binaries = create_sudoku_grid()
print("binaries")
print(len(binaries))
print(binaries)

converted = convertGridToBinary(binaries)
print("converted")
print(len(converted))
print(converted)

print(convertBinaryToGrid(converted))

# start_time = time.time()
# for i in range(0, 3000000):
#     convertBinaryToGrid(binaries)
# end_time = time.time()

# print(f"Execution time: {end_time - start_time:.6f} seconds")
