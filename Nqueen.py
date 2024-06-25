import random
from itertools import islice, product, starmap

configurationBase = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
size = len(configurationBase)
max_iter = 30000
allConfigurations = []
allConfigurationsMatrix = []


# Fonctionnel pour 11 queens

# uniform_crossover
# selection pair better
# 5 flips avec bitflp à 10k itérations
# 8 pops
# 121 de largeur
# 15000 itérations
# seed de 17


def displayConfiguration(configuration):
    for a in configuration:
        print(a)


# x
def check_horizontally(configuration, num_queen_x=0, num_queen_y=0):
    for j in range(num_queen_y + 1, len(configuration[num_queen_x])):
        if configuration[num_queen_x][j] == "R":
            return False
    return True


# y
def check_vertically(configuration, num_queen_x=0, num_queen_y=0):
    for j in range(num_queen_x + 1, len(configuration[num_queen_y])):
        if configuration[j][num_queen_y] == "R":
            return False
    return True


def check_diagonally_down(configuration, num_queen_x=0, num_queen_y=0):
    for i in range(1, len(configuration[num_queen_x])):
        if num_queen_x + i < size and num_queen_y + i < size:
            if configuration[num_queen_x + i][num_queen_y + i] == "R":
                return False
    return True


def check_diagonally_up(configuration, num_queen_x=0, num_queen_y=0):
    for i in range(1, len(configuration[num_queen_x])):
        if num_queen_x - i >= 0 and num_queen_y + i < size:
            if configuration[num_queen_x - i][num_queen_y + i] == "R":
                return False
    return True


def check_nb_queens(configuration=configurationBase):
    nb_queen = 0
    for i in range(0, len(configuration)):
        for j in range(0, len(configuration[i])):
            if configuration[i][j] == "R":
                nb_queen = nb_queen + 1
    if nb_queen != size:
        return False
    return True


def fitness_function(configuration, num_queen_x, num_queen_y):
    fitness = 0
    if not check_horizontally(configuration, num_queen_x, num_queen_y):
        fitness = fitness - 300
    else:
        fitness = fitness + 10
    if not check_vertically(configuration, num_queen_x, num_queen_y):
        # print('2')
        fitness = fitness - 300
    else:
        fitness = fitness + 10
    if not check_diagonally_up(configuration, num_queen_x, num_queen_y):
        # print('3')
        fitness = fitness - 300
    else:
        fitness = fitness + 10
    if not check_diagonally_down(configuration, num_queen_x, num_queen_y):
        # print('4>')
        fitness = fitness - 300
    else:
        fitness = fitness + 10
    if not check_nb_queens(configuration):
        # print('4>')
        fitness = fitness - 10000
    return fitness


def penalty_function(configuration, num_queen_x, num_queen_y):
    penalty = 0
    # print("x " + str(num_queen_x))
    if not check_horizontally(configuration, num_queen_x, num_queen_y):
        # print('1')
        penalty = penalty + 1

    if not check_vertically(configuration, num_queen_x, num_queen_y):
        # print('2')
        penalty = penalty + 1

    if not check_diagonally_up(configuration, num_queen_x, num_queen_y):
        # print('3')
        penalty = penalty + 1

    if not check_diagonally_down(configuration, num_queen_x, num_queen_y):
        # print('4>')
        penalty = penalty + 1
    return penalty


# obtain coord of queens on a colon
def obtain_coord(configuration, nul_col=0):
    listcoord = []

    if nul_col >= len(configuration):
        return False
    for i in range(0, len(configuration[nul_col])):
        if configuration[i][nul_col] == "R":
            listcoord.append((i, nul_col))
    if listcoord == []:
        return []
    else:
        return listcoord


def calculate_fitness(configuration):
    fitness_total = 0
    for i in range(0, len(configuration)):
        a = obtain_coord(configuration, i)
        for j in range(0, len(a)):
            fitness_total = fitness_total + fitness_function(
                configuration, a[j][0], a[j][1]
            )
    return fitness_total


def calculate_penalty(configuration):
    penalty_total = 0
    for i in range(0, len(configuration)):
        a = obtain_coord(configuration, i)

        for j in range(0, len(a)):
            penalty_total = penalty_total + penalty_function(
                configuration, a[j][0], a[j][1]
            )
    return penalty_total


def convertAConfigurationTo01(configuration):
    bandeau = ""
    for i in range(0, len(configuration)):
        for j in range(0, len(configuration[i])):
            if configuration[i][j] == "-":
                bandeau = bandeau + "0"
            else:
                bandeau = bandeau + "1"
    # print(len(bandeau))
    return bandeau


def convert01ToConfiguration(bandeau):
    tab = []
    sousTab = []
    compteur = 0
    # print(len(bandeau))
    # print(bandeau[0])
    for i in range(0, len(bandeau)):
        charac = "-"
        if bandeau[i] == 1:
            charac = "R"
        sousTab.append(charac)
        compteur = compteur + 1
        if i == size - 1:
            tab.append(sousTab)
            sousTab = []
            compteur = 0
        else:
            if compteur % size == 0:
                tab.append(sousTab)
                sousTab = []
                compteur = 0
    # print(len(tab))
    return tab


def convertAConfigurationTo01(configuration):
    bandeau = ""
    for i in range(0, len(configuration)):
        for j in range(0, len(configuration[i])):
            if configuration[i][j] == "-":
                bandeau = bandeau + "0"
            else:
                bandeau = bandeau + "1"
    # print(len(bandeau))
    return bandeau
