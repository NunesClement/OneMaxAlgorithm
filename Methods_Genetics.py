from random import choices, randint, randrange, random, seed
from typing import List, Optional, Callable, Tuple
import numpy as np
import seed_env

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


def bitflip(
    genome: List[int], num: np.array, probability: int = 0.5
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

    for i in range(generation_limit):
        if i % 5 == 0:
            # print("Le programme a l'efficacité : " + str(fitness_func(population[0])) + " / " + str(fitness_limit) + " à l'itération " + str(i))
            collected_iteration = np.append(collected_iteration, i)
            collected_fitness = np.append(collected_fitness, fitness_func(population[0]))
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)
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
