
from functools import partial
from random import choices, randint, randrange, random, seed
from typing import List, Optional, Callable, Tuple
import numpy as np
import seed_env
import Lanceur

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
    # if num == 1:
    #     print("ALERTE")
    # print(num)
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


# calcul de l'amélioration/reward immédiate (plusieurs versions possibles)
def improvement(val_init, val_mut):
    return max(0, (val_mut - val_init))
    # return (val_mut - val_init) + FITNESS_OFFSET
    # return max(0,(val_mut-val_init)/ONE_MAX_LENGTH)
    # return (val_mut-val_init)/ONE_MAX_LENGTH


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
def update_UCB_val(UCB_val, C, op_history, reward_list, i):
    for o in range(len(op_history)):
        UCB_val[o] = reward_list[o] + C * np.sqrt(
            i / (2 * np.log(1 + op_history[o][i]) + 1))


# calcul de l'amélioration/reward immédiate (plusieurs versions possibles)
def improvement(val_init, val_mut):
    # return (val_mut - val_init) + FITNESS_OFFSET
    return max(0, (val_mut - val_init))
    # return max(0,(val_mut-val_init)/ONE_MAX_LENGTH)
    # return (val_mut-val_init)/ONE_MAX_LENGTH


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


# sélection d'un opérateur selon une liste de probabilités
def select_op_proba(proba_list):
    r = random.random()
    somme = 0
    i = 0
    while somme < r and i < len(proba_list):
        somme = somme + proba_list[i]
        if somme < r:
            i = i + 1
    return i


# sélection operateur
# biais, on prend toujours le premier
# on devrait prendre un au hasard qui a la valeur max et pas juste le premier
# ex 1 2 1 3 1 3 1 2
# faut sélectionner un 3 au hasard
# TODO
def select_op_UCB(UCB_val):
    return UCB_val.index(max(UCB_val))


maxFitnessValues = []
meanFitnessValues = []
op_history = []
# 1 flip puis 3 flips puis 5 flips
op_list = [1, 2, 3, 4, 5]
op_history_stat = []
Max_Fitness_history_stat = []
p_min = 0.05
history_size = 10
C = 4


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        nb_run: int = 10,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    # print(mutation_func)
    collected_data = []

    for this_run in range(0, nb_run):
        print("Run actuel : " + str(this_run))
        if mutation_func == "AOS - UCB":
            maxFitnessValues = []
            meanFitnessValues = []
            op_history = []
            init_op_history(op_history, len(op_list))

            reward_list = init_reward_list(len(op_list))
            reward_history = init_reward_history(len(op_list))
            UCB_val = init_UCB_val(len(op_list))
            op_util = []
            # un individu lié à une fitness
            # fitnessValues = list(map(toolbox.evaluate, population))

        population = populate_func()
        i = 0
        collected_iteration = np.array([])
        collected_fitness = np.array([])
        for i in range(generation_limit):
            if mutation_func == "AOS - UCB":
                current_op = select_op_UCB(UCB_val)
                op_util.append(op_list[current_op])
                if mutation_func == "AOS - UCB":
                    for o in range(len(op_list)):
                        if o == current_op:
                            op_history[o].append(op_history[o][i - 1] + 1)
                        else:
                            op_history[o].append(op_history[o][i - 1])

                    mutation_func = partial(mutation, num=op_list[current_op], probability=0.5)

            # print("a " + str(op_list[current_op]))
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

            # if fitness_func(population[0]) >= fitness_limit:
            #     # print("Le programme a l'efficacité : " + str(fitness_func(population[0])) + " / " + str(
            #     #     fitness_limit) + " à l'itération " + str(i))
            #     print("test")
            #     collected_iteration = np.append(collected_iteration, i)
            #     collected_fitness = np.append(collected_fitness, fitness_func(population[0]))
            #     break
            next_generation = population[0:2]

            for j in range(int(len(population) / 2) - 1):
                fitness_init = Lanceur.fitness(greatest(population, fitness_func))
                parents = selection_func(population, fitness_func)
                offspring_a, offspring_b = crossover_func(parents[0], parents[1])
                offspring_a = mutation_func(offspring_a)
                offspring_b = mutation_func(offspring_b)
                next_generation += [offspring_a, offspring_b]

            if mutation_func == "AOS - UCB":
                fitness_now = Lanceur.fitness(greatest(next_generation, fitness_func))
                # fitness_now = Lanceur.fitness(greatest(population, fitness_func))
                update_reward_sliding(reward_list, reward_history, history_size, current_op,
                                      improvement(fitness_init, fitness_now))

                # print(str(fitness_init) + " " + str(fitness_now))
                update_UCB_val(UCB_val, C, op_history, reward_list, i)
                # print(reward_list)

            population = next_generation
            if mutation_func == "AOS - UCB":
                maxFitness = max(collected_fitness)
                meanFitness = sum(collected_fitness) / len(population)
                collected_fitness.append(maxFitness)
                collected_fitness.append(meanFitness)

        collected_data.append(collected_fitness)

    print(" taille collected data : " + str(len(collected_data)))

    collected_data_means = []
    for a in range(0, len(collected_data[0])):
        moy = 0
        for i in range(0, nb_run):
            moy = moy + collected_data[i][a]
        moy = round(moy / len(collected_data))
        # print(moy)
        collected_data_means.append(moy)
    # print([collected_iteration, collected_data_means])
    # print(str(len(collected_iteration)) + " " + str(len(collected_data_means)))
    collected_data_means = np.asarray(collected_data_means)
    # print(collected_data)
    # print(collected_data_means)
    return population, i, [collected_iteration, collected_data_means]
