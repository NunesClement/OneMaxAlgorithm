from functools import partial
from typing import List

import matplotlib.pyplot as plt
# from PIL import Image
# from random import randint
import Methods_Genetics
import numpy as np

# a = OneMaxKnapSack.generate_genome(10) #générer un génome de taille 10
# b = OneMaxKnapSack.generate_genome(10)
# print(a)
# print(b)

# a = [0,0,0,0,0,0,0]
# b = [1,1,1,1,1,1,1]
# print(OneMaxKnapSack.single_point_crossover(a, b)) # crossover entre 2 gênes
# Population10par10 = OneMaxKnapSack.generate_population(10, 10) # générer une pop de 10 gênomes de taille 10

# #muter un gène random
# OneMaxKnapSack.mutation(a, 1, 1)
# print(a)

# fonction de fitness
import seed_env


def fitness(genome: List[int]) -> int:
    if len(genome) <= 0:
        raise ValueError("Le genome doit être > 0 ")
    return genome.count(1)


# L'importance d'utiliser les fonctions de bases e Python

# test d'une fonc de fitness ou il faut un 1 une fois sur 2
# def fitness(genome: OneMaxKnapSack.Genome) -> int:
#     if len(genome) <= 0 :
#         raise ValueError("Le genome doit être > 0 ")
#     count = 0
#     for i in range(0, len(genome)):
#         if genome[i] == 1 and i%2 == 0:
#             count = count + 1
#         if genome[i] == 0 and i%2 == 0:
#             count = count - 1
#         if genome[i] == 1 and i%2 != 0:
#             count = count - 1
#     return count


# test d'une fonc de fitness ou il faut du all different
# array = [
#     1, 2, 2, 4 ,5 ,6, 7, 8, 6, 10
# ]
# # un possible résultat attendu [1,1,0,1,1,1,1,0,1]
# def fitness(genome: OneMaxKnapSack.Genome) -> int:
#     array_collecte = [0,0,0,0,0,0,0,0,0,0]
#     if len(genome) <= 0 :
#         raise ValueError("Le genome doit être > 0 ")
#     count = 0
#     for i in range(0, len(genome)):
#         if genome[i] == 1 :
#             array_collecte[i] = array[i]
#         else :
#             array_collecte[i] = 0
#     for i in range(0, len(genome)):
#         if array_collecte.count(array_collecte[i]) > 1:
#             count = count - 1
#         if array_collecte.count(array_collecte[i]) == 1:
#             count = count + 1
#     return count

# fitness de la pop globale
# print(OneMaxKnapSack.population_fitness(Population10par10, fitness))
#
# print(OneMaxKnapSack.selection_pair(Population10par10, fitness))

print("________________")


def launch_with_param(
        mutation_param="1-flip",
        crossover_param="single_point_crossover",
        selection_param="selection_pair_parmis_s_random",
        size=10,
        genome_length=10,
        fitness_limit=10,
        generation_limit=10,
):
    weight_limit = 10
    if mutation_param == "3-flip":
        mutation = partial(Methods_Genetics.mutation, num=3, probability=0.5)
    if mutation_param == "1-flip":
        mutation = partial(Methods_Genetics.mutation, num=1, probability=0.5)
    if mutation_param == "5-flip":
        mutation = partial(Methods_Genetics.mutation, num=5, probability=0.5)
    if mutation_param == "0-flip":
        mutation = partial(Methods_Genetics.mutation, num=0, probability=0.5)
    if crossover_param == "uniform_crossover":
        crossover = Methods_Genetics.uniform_crossover
    else:
        crossover = Methods_Genetics.single_point_crossover

    if selection_param == "selection_pair_parmis_s_random":
        selection = partial(Methods_Genetics.selection_pair_parmis_s_random, s=2)
    if selection_param == "selection_pair_better":
        selection = partial(Methods_Genetics.selection_pair_better)
    if selection_param == "selection_pair":
        selection = partial(Methods_Genetics.selection_pair)
    # a setup via le globalState de l'interface TO DO

    # noinspection PyTupleAssignmentBalance
    (population, generations, collected_data) = Methods_Genetics.run_evolution(
        # taille pop et taille de genome
        populate_func=partial(Methods_Genetics.generate_population, size=size, genome_length=genome_length),
        fitness_func=partial(fitness),
        selection_func=selection,
        crossover_func=crossover,
        mutation_func=mutation,
        # bridage de la fitness
        fitness_limit=fitness_limit,
        # nombre de générations
        generation_limit=generation_limit
    )
    print("One call just finished")
    congig_memory = [str(seed_env.getSeed()), str(mutation_param), str(selection_param),
                     str(crossover_param), str(fitness_limit),
                     str(generation_limit), str(genome_length), str(size)]
    # iteration_array = collected_data[0].astype(np.float)
    iteration_array = np.array_str(collected_data[0])

    fitness_array = np.array_str(collected_data[1])
    # np.savetxt("array_1d.csv", [congig_memory, iteration_array, fitness_array], delimiter=",",
    #            fmt="%s")
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    with open('array_1d.csv', 'a') as csvfile:
        np.savetxt(csvfile, [congig_memory, iteration_array], delimiter=',', fmt="%s")
        np.savetxt(csvfile, [fitness_array], delimiter=',', fmt="%s")

    return population, generations, collected_data


def debugGlobalState(globalState):
    print("Seed "+ str(globalState.seed))
    print("Type de mutation "
          + str(globalState.mutation_params[0])
          + " avec une proba de "
          + str(globalState.mutation_params[1])
          )
    print("Paramètre de sélection " + str(globalState.selection_params))
    print("Limit de fitness " + str(globalState.fitness_limit))
    print("Nb d'itération/génération " + str(globalState.generation_limit))
    print("Taille d'un genome " + str(globalState.genome_length))
    print("Taille d'une population " + str(globalState.taille_pop))


def launch_the_launcher(globalState):
    plt.figure().clear()
    plt.xlabel("Nombre de générations")
    plt.ylabel("Fitness atteinte")
    debugGlobalState(globalState)
    population, generations, collected_data = launch_with_param(
        str(globalState.mutation_params[0]),
        "single_point_crossover",
        "selection_pair_better",
        int(globalState.taille_pop),
        int(globalState.genome_length),
        int(globalState.fitness_limit),
        int(globalState.generation_limit)
    )
    x = collected_data[0]
    y = collected_data[1]
    lbl = "single_point_crossover " + globalState.mutation_params[0] + " selection_pair_better " + str(
        generations) + " " + str(
        collected_data[1][len(collected_data[1]) - 1])
    plt.plot(x, y, label=lbl)

    population, generations, collected_data = launch_with_param(
        str(globalState.mutation_params[0]),
        "uniform_crossover",
        "selection_pair_better",
        int(globalState.taille_pop),
        int(globalState.genome_length),
        int(globalState.fitness_limit),
        int(globalState.generation_limit)
    )

    x = collected_data[0]
    y = collected_data[1]
    lbl = "uniform_crossover + 1 flips + selection_pair " + str(generations) + " " + str(
        collected_data[1][len(collected_data[1]) - 1])
    plt.plot(x, y, label=lbl)

    plt.legend()

    plt.savefig("test")

    # open method used to open different extension image file
    # im = Image.open("test.png")
    #
    # # This method will show image in any image viewer
    # im.show()
    return 0
