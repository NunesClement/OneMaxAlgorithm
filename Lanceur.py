from functools import partial
from typing import List
import matplotlib.pyplot as plt
import Methods_Genetics
import numpy as np
import Nqueen

# fonction de fitness
import seed_env

plt.figure(figsize=(10, 6))


def fitness(genome: List[int]) -> int:
    if len(genome) <= 0:
        raise ValueError("Le genome doit être > 0 ")
    return genome.count(1)


# fitness for nqueen
def fitness_nqueen(genome: List[int]) -> int:
    if len(genome) <= 0:
        raise ValueError("Le genome doit être > 0 ")
    # print(genome)
    return Nqueen.calculate_fitness(Nqueen.convert01ToConfiguration(genome))


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

def launch_with_param(
        mutation_param="1-flip",
        selection_param="",
        selector_operator="1-flip",
        size=10,
        genome_length=10,
        fitness_limit=10,
        generation_limit=10,
        nb_run=10,
        crossover_param="single_point_crossover",
):
    # print(selector_operator)
    weight_limit = 10
    mutation = partial(Methods_Genetics.mutation, num=1, probability=0.5)
    if mutation_param == "bitflip":
        mutation = partial(Methods_Genetics.bitflip)
    if mutation_param == "0-flip":
        mutation = partial(Methods_Genetics.mutation, num=0, probability=0.5)
    if mutation_param == "1-flip":
        mutation = partial(Methods_Genetics.mutation, num=1, probability=0.5)
    if mutation_param == "2-flip":
        mutation = partial(Methods_Genetics.mutation, num=2, probability=0.5)
    if mutation_param == "3-flip":
        mutation = partial(Methods_Genetics.mutation, num=3, probability=0.5)
    if mutation_param == "4-flip":
        mutation = partial(Methods_Genetics.mutation, num=4, probability=0.5)
    if mutation_param == "5-flip":
        mutation = partial(Methods_Genetics.mutation, num=5, probability=0.5)

    # print("crossover_param " + str(crossover_param))
    if crossover_param == "uniform_crossover":
        crossover = Methods_Genetics.uniform_crossover
    else:
        crossover = Methods_Genetics.single_point_crossover

    nb_tournois = 2

    if size < 10:
        nb_tournois = 2
    if 10 < size < 30:
        nb_tournois = 5
    if 30 <= size < 70:
        nb_tournois = 7
    if 70 <= size < 100:
        nb_tournois = 10
    if 100 <= size:
        nb_tournois = round(size/10)

    selection = partial(Methods_Genetics.selection_tournois_parmi_s_randoms, s=nb_tournois)
    if selection_param == "selection_tournois_parmi_s_randoms":
        selection = partial(Methods_Genetics.selection_tournois_parmi_s_randoms, s=nb_tournois)
    if selection_param == "selection_pair_better":
        selection = partial(Methods_Genetics.selection_pair_better)
    if selection_param == "selection_pair":
        selection = partial(Methods_Genetics.selection_pair)

    # noinspection PyTupleAssignmentBalance
    (population, generations, collected_data) = Methods_Genetics.run_evolution(
        # taille pop et taille de genome
        populate_func=partial(Methods_Genetics.generate_population, size=size, genome_length=genome_length),
        fitness_func=partial(fitness),
        selection_func=selection,
        crossover_func=crossover,
        selector_operator=selector_operator,
        mutation_func=mutation,
        # bridage de la fitness
        fitness_limit=fitness_limit,
        # nombre de générations
        generation_limit=generation_limit,
        nb_run=nb_run
    )
    print(selector_operator)
    print("One call just finished")
    congig_memory = [str(seed_env.getSeed()), str(mutation_param), str(selection_param),
                     str(crossover_param), str(fitness_limit),
                     str(generation_limit), str(genome_length), str(size)]

    iteration_array = np.array_str(collected_data[0])

    fitness_array = np.array_str(collected_data[1])

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    with open('array_1d.csv', 'a') as csvfile:
        np.savetxt(csvfile, [congig_memory, iteration_array], delimiter=',', fmt="%s")
        np.savetxt(csvfile, [fitness_array], delimiter=',', fmt="%s")

    return population, generations, collected_data


def debugGlobalState(global_state):
    print("Seed " + str(global_state.seed))
    print("Type de mutation "
          + str(global_state.mutation_params[0])
          + " avec une proba de "
          + str(global_state.mutation_params[1])
          )
    print("Paramètre de croisemement " + str(global_state.croisement_param))
    print("Paramètre de sélection " + str(global_state.selection_params))
    print("Limit de fitness " + str(global_state.fitness_limit))
    print("Nb d'itération/génération " + str(global_state.generation_limit))
    print("Taille d'un genome " + str(global_state.genome_length))
    print("Taille d'une population " + str(global_state.taille_pop))
    print("Type d'AOS " + str(global_state.selector_operator))


def cleanup_graph():
    plt.figure(figsize=(10, 6))
    print("The plot has been cleaned up !")


def launch_the_launcher(global_state):
    plt.xlabel("Nombre de générations")
    plt.ylabel("Fitness atteinte")
    debugGlobalState(global_state)
    population, generations, collected_data = launch_with_param(
        str(global_state.mutation_params[0]),
        str(global_state.selection_params),
        str(global_state.selector_operator),
        int(global_state.taille_pop),
        int(global_state.genome_length),
        int(global_state.fitness_limit),
        int(global_state.generation_limit),
        int(global_state.nb_run),
        str(global_state.croisement_param)
    )

    x = collected_data[0]
    y = collected_data[1]
    lbl = str(global_state.croisement_param) + " " + str(global_state.mutation_params[0]) + " " + str(
        global_state.selection_params) + " " + str(
        generations) + " générations " + str(
        collected_data[1][len(collected_data[1]) - 1])
    plt.plot(x, y, label=lbl)
    plt.title("AG lancé sur " + str(global_state.nb_run) + " executions")

    plt.legend()

    plt.savefig("plot")

    return 0
