# URG cut if fitness is reached
# URG generalize cut for all operator
# URG improve intern methods calculations to 10x
# URG place measurement to delect slower portions of code
# URG fix differents sudoku sizes

from functools import partial
from typing import List
import matplotlib.pyplot as plt
import Genetics_Methods
import numpy as np
import Nqueen
import Sudoku
import csv


# fonction de fitness
import seed_env

plt.figure(figsize=(10, 6))


# fitness for nqueen
def fitness_nqueen(genome: List[int]) -> int:
    if len(genome) <= 0:
        raise ValueError("Le genome doit être > 0 ")
    return Nqueen.calculate_fitness(Nqueen.convert01ToConfiguration(genome))


def fitness_sudoku(genome: List[int]) -> int:
    if len(genome) <= 0:
        raise ValueError("Le genome doit être > 0 ")
    return Sudoku.calculate_fitness(genome)


def fitness(genome: List[int]) -> int:
    if len(genome) <= 0:
        raise ValueError("Le genome doit être > 0 ")
    return genome.count(1)


def fitness_manager(problemChoice):
    if problemChoice == "OneMax":
        return fitness
    if problemChoice == "N-Reine":
        return fitness_nqueen
    if problemChoice == "Sudoku":
        return fitness_sudoku
    return fitness


def launch_with_param(
    mutation_param="1-flip",
    selection_param="",
    selector_operator="1-flip",
    size=10,
    genome_length=10,
    fitness_limit=10,
    generation_limit=10,
    # sudoku_size=4,
    nb_run=10,
    crossover_param="single_point_crossover",
    selected_problem="OneMax",
):
    weight_limit = 10
    mutation = partial(Genetics_Methods.mutation, num=1, probability=0.5)

    # URG : refacto this ugly code
    if mutation_param == "bitflip":
        mutation = partial(Genetics_Methods.bitflip)
    if mutation_param == "0-flip":
        mutation = partial(Genetics_Methods.mutation, num=0, probability=0.5)
    if mutation_param == "1-flip":
        mutation = partial(Genetics_Methods.mutation, num=1, probability=0.5)
    if mutation_param == "2-flip":
        mutation = partial(Genetics_Methods.mutation, num=2, probability=0.5)
    if mutation_param == "3-flip":
        mutation = partial(Genetics_Methods.mutation, num=3, probability=0.5)
    if mutation_param == "4-flip":
        mutation = partial(Genetics_Methods.mutation, num=4, probability=0.5)
    if mutation_param == "5-flip":
        mutation = partial(Genetics_Methods.mutation, num=5, probability=0.5)

    if crossover_param == "uniform_crossover":
        crossover = Genetics_Methods.uniform_crossover
    else:
        crossover = Genetics_Methods.single_point_crossover

    nb_tournois = 2

    if size < 10:
        nb_tournois = 3
    if 10 < size < 30:
        nb_tournois = 6
    if 30 <= size < 70:
        nb_tournois = 13
    if 70 <= size < 100:
        nb_tournois = 25
    if 100 <= size:
        nb_tournois = round(size / 5)

    selection = partial(
        Genetics_Methods.selection_tournois_parmi_s_randoms, s=nb_tournois
    )
    if selection_param == "selection_tournois_parmi_s_randoms":
        selection = partial(
            Genetics_Methods.selection_tournois_parmi_s_randoms, s=nb_tournois
        )
    if selection_param == "selection_pair_better":
        selection = partial(Genetics_Methods.selection_pair_better)
    if selection_param == "selection_pair":
        selection = partial(Genetics_Methods.selection_pair)

    # noinspection PyTupleAssignmentBalance
    (population, generations, collected_data) = Genetics_Methods.run_evolution(
        # taille pop et taille de genome
        populate_func=partial(
            Genetics_Methods.generate_population, size=size, genome_length=genome_length
        ),
        fitness_func=partial(fitness_manager(selected_problem)),
        selection_func=selection,
        crossover_func=crossover,
        selector_operator=selector_operator,
        mutation_func=mutation,
        # bridage de la fitness
        fitness_limit=fitness_limit,
        # nombre de générations
        generation_limit=generation_limit,
        # sudoku_size=sudoku_size,
        nb_run=nb_run,
    )
    print(selector_operator)
    print("One call just finished")
    congig_memory = [
        str(seed_env.getSeed()),
        str(mutation_param),
        str(selection_param),
        str(crossover_param),
        str(fitness_limit),
        str(generation_limit),
        str(genome_length),
        str(size),
    ]

    iteration_array = np.array_str(collected_data[0])

    fitness_array = np.array_str(collected_data[1])

    with open("debug/array_1d.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")

        # Write the congig_memory and iteration_array
        writer.writerow(congig_memory)
        writer.writerow(iteration_array)

        # Write the fitness_array
        writer.writerow(fitness_array)
    return population, generations, collected_data


def debugGlobalState(global_state):
    print("Seed " + str(global_state.seed))
    print(
        "Type de mutation "
        + str(global_state.mutation_params[0])
        + " avec une proba de "
        + str(global_state.mutation_params[1])
    )

    print(
        f"""
    Choix du problème        : {global_state.selected_problem}
    Paramètre de croisement  : {global_state.croisement_param}
    Paramètre de sélection   : {global_state.selection_params}
    Limite de fitness        : {global_state.fitness_limit}
    Nb d'itération/génération: {global_state.generation_limit}
    Taille d'un genome       : {global_state.genome_length}
    Taille d'une population  : {global_state.taille_pop}
    Type d'AOS               : {global_state.selector_operator}
    """
    )


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
        str(global_state.croisement_param),
        str(global_state.selected_problem),
    )

    if global_state.selected_problem == "N-Reine":
        Nqueen.displayConfiguration(Nqueen.convert01ToConfiguration(population[0]))
        print(
            "Penalty  : "
            + str(
                Nqueen.calculate_penalty(Nqueen.convert01ToConfiguration(population[0]))
            )
        )
    if global_state.selected_problem == "Sudoku":
        Sudoku.display_sudoku_grid(Sudoku.convertBinaryToGrid(population[0]))
        print("Fitness : " + str(Sudoku.calculate_fitness(population[0])))

    x = collected_data[0]
    y = collected_data[1]
    lbl = (
        str(global_state.croisement_param)
        + " "
        + str(global_state.mutation_params[0])
        + " "
        + str(global_state.selection_params)
        + " "
        + str(generations + 1)
        + " générations "
        + str(collected_data[1][len(collected_data[1]) - 1])
    )
    plt.plot(x, y, label=lbl)
    plt.title("AG lancé sur " + str(global_state.nb_run) + " executions")

    with open(
        "debug/" + str(global_state.mutation_params[0]) + ".csv", "a", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")

        # Write the Generations data
        writer.writerow(collected_data[0])
        # Write the Fitness data
        writer.writerow(collected_data[1])

    plt.legend()
    plt.savefig("debug/plot")

    return 0
