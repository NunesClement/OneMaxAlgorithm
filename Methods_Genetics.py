from functools import partial
from random import choices, randint, randrange, random, seed
from typing import List, Optional, Callable, Tuple
import numpy as np
import seed_env
from interface import global_state

np.random.seed(seed_env.getSeed())
seed(seed_env.getSeed())
Genome = List[int]
Population = List[List[int]]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[List[int]], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[List[int], List[int]]]
CrossoverFunc = Callable[[List[int], List[int]], Tuple[List[int], List[int]]]
MutationFunc = Callable[[List[int]], List[int]]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]


# Génération d'un génome personnalisé
def custom_genome(custom_list: List[int]) -> List[int]:
    return custom_list


# Génération d'un génome random
def generate_genome(length: int) -> List[int]:
    return choices([0, 1], k=length)


# Génération d'une population de génome
def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


# 1_point_crossover / échange une portion random sur le bandeau
def single_point_crossover(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
    if len(a) != len(b):
        raise ValueError("Les génomes doivent être de la même taille")
    if len(a) < 2:
        return a, b

    p = randint(1, len(a) - 1)
    a = a[0:p] + b[p:]
    b = b[0:p] + a[p:]

    return a, b


def uniform_crossover(
        individual_1: np.array, individual_2: np.array, thresh: int = 0.5
):
    offspring_1 = individual_1.copy()
    offspring_2 = individual_2.copy()
    for i, _ in enumerate(offspring_1):
        ran_num = np.random.uniform()
        if ran_num > thresh:
            # swap 2 bits at i-th position
            temp_bit = offspring_1[i]
            offspring_1[i] = offspring_2[i]
            offspring_2[i] = temp_bit
        else:
            continue

    return [offspring_1, offspring_2]


# 50% de chance d'effectuer une mutation
def mutation(genome: List[int], num: int = 1, probability: float = 0.5) -> List[int]:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


# être sûr que c'est comme ça ?
def bitflip(
        genome: List[int]
):
    genome = genome.copy()
    index = randrange(len(genome))
    index2 = randrange(len(genome))
    temp = genome[index]
    genome[index] = temp
    genome[index2] = genome[index]
    return genome


# 1/taillePop de chance d'effectuer une mutation
def mutationPop(genome: List[int], num: int = 1, size_pop=10) -> List[int]:
    probability = 1 / size_pop
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)

    return genome


# mesurer la fitness de toute la pop, on choisi la fonctuon  de fitness
def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])


# selectionner 2 gênomes en vu d'un croisement RANDOM
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(gene) for gene in population],
        k=2
    )


# selectionner 2 meilleurs gênomes en vu d'un croisement RANDOM
def selection_pair_better(population: Population, fitness_func: FitnessFunc) -> Population:
    select_pop = sort_population(population, fitness_func)
    # Test en gardant que les plus nuls
    # return select_pop[len(select_pop) -1], select_pop[len(select_pop) -2]
    return select_pop[0], select_pop[1]


# selectionner 2 meilleurs génomes parmis S random
def selection_pair_parmis_s_random(population: Population, fitness_func: FitnessFunc, s: int = 2) -> Population:
    if s >= len(population):
        raise ValueError("L'ensemble S random doit etre < a la taille de la pop")

    index_selection_aleatoire = np.unique(np.random.randint(len(population), size=(1, s)))
    ensemble_pris_aleatoirement = []
    # sécurité
    while index_selection_aleatoire.size < 2:
        index_selection_aleatoire = np.unique(np.random.randint(len(population), size=(1, s)))

    for i in range(0, index_selection_aleatoire.size):
        ensemble_pris_aleatoirement.append(population[i])
    ensemble_pris_aleatoirement = sort_population(ensemble_pris_aleatoirement, fitness_func)
    # Test en gardant que les plus nuls
    return ensemble_pris_aleatoirement[0], ensemble_pris_aleatoirement[1]


# trie de la population en fonction de sa fitness
def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)


# trie de la population en fonction de sa fitness
def greatest(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)[0]


# trie de la population en fonction de sa fitness
def loosest(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=False)[0]


def genome_to_string(genome: List[int]) -> str:
    return "".join(map(str, genome))


# def insertion_best_fitness(population, offspring):
#     for ind in offspring:
#         worst = toolbox.worst(population, 1)
#         if ind.fitness.values[0] > worst[0].fitness.values[0]:
#             population.remove(worst[0])
#             population.append(ind)
#     return population


def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]),
                              fitness_func(sorted_population[-1])))
    print("")

    return sorted_population[0]


# initialisation strutures
# [0, 0, 0] 3 opérateurs
def init_UCB_val(taille):
    l = []
    for o in range(taille):
        l.append(0)
    return l


# sélection operateur
# biais, on prend toujours le premier
# on devrait prendre un au hasard qui a la valeur max et pas juste le premier
# ex 1 2 1 3 1 3 1 2
# faut sélectionner un 3 au hasard
# TODO
def select_op_UCB(UCB_val):
    return UCB_val.index(max(UCB_val))


# Mise à jour de la récompense glissante
def update_reward_sliding(reward_list, reward_history, history_size, index, value):
    if reward_history[index] == [0]:
        reward_history[index] = [value]
    else:
        reward_history[index].append(value)
    if len(reward_history[index]) > history_size:
        reward_history[index] = reward_history[index][1:len(reward_history[index])]
    reward_list[index] = sum(reward_history[index]) / len(reward_history[index])


# MAJ valeurs UCB
# Le biais pour faire décroitre de façon algorithmique le regret
def update_UCB_val(UCB_val, C, op_history, reward_list, generationCounter):
    for o in range(len(op_history)):
        UCB_val[o] = reward_list[o] + C * np.sqrt(
            generationCounter / (2 * np.log(1 + op_history[o][generationCounter]) + 1))


# calcul de l'amélioration/reward immédiate (plusieurs versions possibles)
def improvement(val_init, val_mut):
    # return (val_mut - val_init) + FITNESS_OFFSET
    return max(0, (val_mut - val_init))
    # return max(0,(val_mut-val_init)/ONE_MAX_LENGTH)
    # return (val_mut-val_init)/ONE_MAX_LENGTH


# boucle évol. UCB (même structure que PM )
# à la place on sélectionne la meilleure valeur UCB
def ea_loop_MAB(genome_length, generation_limit, population, maxFitnessValues, meanFitnessValues, op_history, op_list,
                history_size, C):
    generationCounter = 0
    P_MUTATION = 0.1

    reward_list = init_reward_list(len(op_list))
    reward_history = init_reward_history(len(op_list))
    UCB_val = init_UCB_val(len(op_list))
    op_util = []
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue
    fitnessValues = [individual.fitness.values[0] for individual in population]
    while (max(fitnessValues) < genome_length) and (generationCounter < generation_limit):
        generationCounter = generationCounter + 1
        current_op = select_op_UCB(UCB_val)
        for o in range(len(op_list)):
            if o == current_op:
                op_history[o].append(op_history[o][generationCounter - 1] + 1)
            else:
                op_history[o].append(op_history[o][generationCounter - 1])
        op_util.append(op_list[current_op])
        offspring = toolbox.select(population, 1)
        offspring = list(map(toolbox.clone, offspring))
        for mutant in offspring:
            if random.random() < P_MUTATION:
                fitness_init = mutant.fitness.values[0]
                if current_op > 0:
                    n_flips(mutant, op_list[current_op])
                else:
                    toolbox.bitflip(mutant)
                del mutant.fitness.values
                mutant.fitness.values = list(toolbox.evaluate(mutant))

        update_reward_sliding(reward_list, reward_history, history_size, current_op,
                              improvement(fitness_init, mutant.fitness.values[0]))
        update_UCB_val(UCB_val, C, op_history, reward_list, generationCounter)

        if improvement(fitness_init, mutant.fitness.values[0]) > 0:
            population = insertion_best_fitness(population, offspring)

        fitnessValues = [ind.fitness.values[0] for ind in population]
        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)


def init_reward_list(taille):
    l = []
    for o in range(taille):
        l.append(0)
    return l


def init_reward_history(taille):
    l = []
    for o in range(taille):
        l.append([0])
    return l


def init_op_history(l, taille):
    for o in range(taille):
        l.append([0])


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    """

    :rtype: object
    """
    population = populate_func()
    i = 0
    collected_iteration = np.array([])
    collected_fitness = np.array([])

    genome_length = global_state.genome_length
    maxFitnessValues = []
    meanFitnessValues = []
    op_history = []
    # 1 flip puis 3 flips puis 5 flips
    op_list = [1, 3, 5]
    op_history_stat = []
    Max_Fitness_history_stat = []

    p_min = 0.05
    history_size = 10
    C = 4
    # nombre de runs pour les stats
    # moyenne sur plusieurs exec
    NB_RUNS = 10

    long_min = generation_limit

    print("> Taille génôme " + str(global_state.genome_length))
    AOS = "AAA"
    if AOS == "MAB":
        ea_loop_MAB(genome_length, generation_limit, population, maxFitnessValues, meanFitnessValues, op_history,
                    op_list, history_size, C)
    else:
        for i in range(generation_limit):

            maxFitnessValues = []
            meanFitnessValues = []
            op_history = []
            init_op_history(op_history, len(op_list))

            if generation_limit > 1000:
                if i % 500 == 0 and i != 0:
                    print("Itération " + str(i) + " ...")
            if i % 5 == 0:
                # print("Le programme a l'efficacité : " + str(fitness_func(population[0])) + " / " + str(fitness_limit)
                # + " à l'itération " + str(i))
                collected_iteration = np.append(collected_iteration, i)
                collected_fitness = np.append(collected_fitness, fitness_func(population[0]))
            population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)
            # print("Meilleur : " + str(fitness_func(greatest(population, fitness_func))))
            # print("Plus nulle : " + str(fitness_func(loosest(population, fitness_func))))
            # print(fitness_func(sort_population(population, fitness_func)[len(population)-1]))
            if printer is not None:
                printer(population, i, fitness_func)

            if fitness_func(population[0]) >= fitness_limit:
                # print("Le programme a l'efficacité : " + str(fitness_func(population[0])) + " / " + str(
                #     fitness_limit) + " à l'itération " + str(i))
                collected_iteration = np.append(collected_iteration, i)
                collected_fitness = np.append(collected_fitness, fitness_func(population[0]))
                break
            next_generation = population[0:2]

            for j in range(int(len(population) / 2) - 1):
                parents = selection_func(population, fitness_func)
                offspring_a, offspring_b = crossover_func(parents[0], parents[1])
                offspring_a = mutation_func(offspring_a)
                offspring_b = mutation_func(offspring_b)
                next_generation += [offspring_a, offspring_b]

            population = next_generation
        collected_data = [collected_iteration, collected_fitness]
        return population, i, collected_data
