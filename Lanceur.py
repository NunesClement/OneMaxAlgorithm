from functools import partial
from typing import List
import matplotlib.pyplot as plt
import Genetics_Methods
import numpy as np
import Nqueen


# fonction de fitness
import seed_env

plt.figure(figsize=(10, 6))


# fitness for nqueen
def fitness_nqueen(genome: List[int]) -> int:
    if len(genome) <= 0:
        raise ValueError("Le genome doit être > 0 ")
    return Nqueen.calculate_fitness(Nqueen.convert01ToConfiguration(genome))


def fitness(genome: List[int]) -> int:
    if len(genome) <= 0:
        raise ValueError("Le genome doit être > 0 ")
    return genome.count(1)


def fitness_manager(problemChoice):
    if problemChoice == "OneMax":
        return fitness
    else:
        return fitness_nqueen


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
    selected_problem="OneMax",
):
    weight_limit = 10
    mutation = partial(Genetics_Methods.mutation, num=1, probability=0.5)
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

    # print("crossover_param " + str(crossover_param))
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

    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    with open("array_1d.csv", "a") as csvfile:
        np.savetxt(csvfile, [congig_memory, iteration_array], delimiter=",", fmt="%s")
        np.savetxt(csvfile, [fitness_array], delimiter=",", fmt="%s")

    return population, generations, collected_data


def debugGlobalState(global_state):
    print("Seed " + str(global_state.seed))
    print(
        "Type de mutation "
        + str(global_state.mutation_params[0])
        + " avec une proba de "
        + str(global_state.mutation_params[1])
    )
    print("Choix du problème" + str(global_state.selected_problem))
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

    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    with open("debug/" + str(global_state.mutation_params[0]) + ".csv", "a") as csvfile:
        # Générations
        np.savetxt(csvfile, [collected_data[0]], delimiter=",", fmt="%s")
        # Fitness
        np.savetxt(csvfile, [collected_data[1]], delimiter=",", fmt="%s")

    plt.legend()

    plt.savefig("plot")

    return 0
