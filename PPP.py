from typing import List
import numpy as np
from random import randrange, random
import math
import copy

from numpy.random import f
import Genetics_Methods

pop_size = 20

penalization_factor = 1

prob_hill_climbing_1 = 75

hc_iterations = 1

# Params

periods = 6

yachts_quantity = 42
#
yachts_capacities = [
    6,
    8,
    12,
    12,
    12,
    12,
    12,
    10,
    10,
    10,
    10,
    10,
    8,
    8,
    8,
    12,
    8,
    8,
    8,
    8,
    8,
    8,
    7,
    7,
    7,
    7,
    7,
    7,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    9,
    0,
    0,
    0,
]
yachts_crew_size = [
    2,
    2,
    2,
    2,
    4,
    4,
    4,
    1,
    2,
    2,
    2,
    3,
    4,
    2,
    3,
    6,
    2,
    2,
    4,
    2,
    4,
    5,
    4,
    4,
    2,
    2,
    4,
    5,
    2,
    4,
    2,
    2,
    2,
    2,
    2,
    2,
    4,
    5,
    7,
    2,
    3,
    4,
]


class Chrom:
    # savoir qui a visité qui sur quelle période
    x = np.zeros((42, 42, 6))
    # savoir qui a rencontré qui sur quelle période
    m = np.zeros((42, 42, 6))
    # déterminer les yatchs
    h = np.zeros(42)
    yachts_host_quantity = 0
    yachts_host_indexes = np.zeros((42, 42, 6))
    fit = 0
    unfitness = 0
    faisable = False


a = Chrom()
b = Chrom()


def yachts_host_counter(array):
    acumulator = 0

    for i in range(0, yachts_quantity - 1):
        if array[i] == 1:
            acumulator = acumulator + 1

    return acumulator


def obtain_h(chromosome):
    while (chromosome.yachts_host_quantity < periods - 1) or (
        chromosome.yachts_host_quantity == yachts_quantity
    ):
        for j in range(0, yachts_quantity - 1):
            bit = random() > 0.5
            if bit == 0:
                chromosome.h[j] = 0
            else:
                chromosome.h[j] = 1
            chromosome.yachts_host_quantity = yachts_host_counter(chromosome.h)

    chromosome.yachts_host_indexes = None


def shuffle_array_short_int(array, n):
    if n > 1:
        array = np.random.shuffle(array)


def obtain_x(chromosome):
    for i in range(0, yachts_quantity - 1):
        for j in range(0, yachts_quantity - 1):
            for k in range(0, periods - 1):
                chromosome.x[i][j][k] = 0
    # for j in range(0, yachts_quantity - 1):
    #     for k in range(0, periods - 1):
    #         if chromosome.h[j] != 1 and k < periods:
    #             yacht_host_index = chromosome.yachts_host_indexes[j][k]
    #             chromosome.x[yacht_host_index][j][k] = 1


def obtain_m(chromosome):
    for i in range(0, yachts_quantity - 1):
        for j in range(0, yachts_quantity - 1):
            for k in range(0, periods - 1):
                chromosome.m[i][j][k] = 0
    for i in range(0, yachts_quantity - 1):
        for j in range(0, yachts_quantity - 1):
            for l in range(0, j - 1):
                for k in range(0, periods - 1):
                    if chromosome.x[i][j][k] and chromosome.x[i][l][k]:
                        chromosome.m[j][l][k] = chromosome.m[j][l][k] + 1
                        chromosome.m[l][j][k] = chromosome.m[l][j][k] + 1


def calculate_unfitness(chromosome, verb):
    visits_to_nonhost_yatch = 0
    exceeded_capacity = 0
    idle_or_host_visiting = 0
    visits_same_host = 0
    impossible = 0
    meet_same_people = 0

    if verb:
        print("Il y a un yatch host ", chromosome.yachts_host_quantity)

    if verb:
        print("Les indices des yatchs hosts sont :  ")

        for i in range(0, yachts_quantity - 1):
            print(chromosome.h[i])

    for i in range(0, yachts_quantity - 1):
        for j in range(0, yachts_quantity - 1):
            for k in range(0, periods - 1):
                if chromosome.x[i][j][k] > chromosome.h[i] and (i != j):
                    visits_to_nonhost_yatch = visits_to_nonhost_yatch + 1

    if visits_to_nonhost_yatch > 0 and verb:
        print(
            "Au total, il y a eu "
            + visits_to_nonhost_yatch
            + " d visites sur des yachts non hôtes"
        )

    for i in range(0, yachts_quantity - 1):
        if chromosome.h[i]:
            for k in range(0, periods - 1):
                acumulated_incoming_crew = 0
                for j in range(0, yachts_quantity - 1):
                    if (i != j) and chromosome.h[i]:
                        acumulated_incoming_crew += (
                            yachts_crew_size[j] * chromosome.x[i][j][k]
                        )

                if (
                    acumulated_incoming_crew
                    > yachts_capacities[i] - yachts_crew_size[i]
                ):
                    exceeded_capacity += acumulated_incoming_crew - (
                        yachts_capacities[i] - yachts_crew_size[i]
                    )

    if (exceeded_capacity > 0) and verb:
        print(
            "Au total il y a eu "
            + exceeded_capacity
            + " de dépassements par rapport aux capacités des yachts"
        )

    for j in range(0, yachts_quantity - 1):
        for k in range(0, periods - 1):
            acumulated_visits = 0
            for i in range(0, yachts_quantity - 1):
                acumulated_visits += chromosome.x[i][j][k]
                if chromosome.h[j] + acumulated_visits != 1:
                    idle_or_host_visiting += 1

    if (idle_or_host_visiting > 0) and verb:
        print(idle_or_host_visiting)
        print("Visito  plus d'un yach ")
        print("Visite un yacht en tant qu'hôte")

    for i in range(0, yachts_quantity - 1):
        for j in range(0, yachts_quantity - 1):
            acumulated_visits = 0
            for k in range(0, periods - 1):
                acumulated_visits += chromosome.x[i][j][k]
            if acumulated_visits > 1:
                visits_same_host += acumulated_visits - 1

    if (visits_same_host > 0) and verb:
        print(
            "Il y a eu "
            + visits_same_host
            + " cas dans lesquels les équipages ont été réunis avec les mêmes yachts hôtes, visits_same_host"
        )

    for i in range(0, yachts_quantity - 1):
        for j in range(0, yachts_quantity - 1):
            for l in range(0, j - 1):
                for k in range(0, periods - 1):
                    if (
                        chromosome.x[j][l][k]
                        < chromosome.x[i][j][k] + chromosome.x[i][l][k] - 1
                    ):
                        impossible += (
                            chromosome.x[i][j][k]
                            + chromosome.x[i][l][k]
                            - 1
                            - chromosome.x[j][l][k]
                        )

    if (impossible > 0) and verb:
        print(
            "Il y a eu " + impossible + " cas dans lesquels l'impossible s'est produit"
        )

    for i in range(0, yachts_quantity - 1):
        for j in range(0, i - 1):
            acumulated_meets = 0
            for k in range(0, periods - 1):
                acumulated_meets += chromosome.m[i][j][k]
            if acumulated_meets > 1:
                meet_same_people += acumulated_meets - 1

    if (meet_same_people > 0) and verb:
        print(
            "Il y a eu "
            + meet_same_people
            + " réunions qui n'auraient pas dû avoir lieu entre des yachts non hôtes"
        )

    chromosome.unfitness = (
        penalization_factor
        * (
            visits_to_nonhost_yatch
            + exceeded_capacity
            + idle_or_host_visiting
            + visits_same_host
            + impossible
            + meet_same_people
        )
        + chromosome.yachts_host_quantity
    )

    if chromosome.unfitness > chromosome.yachts_host_quantity:
        chromosome.faisable = False
    else:
        chromosome.faisable = True

    if verb:
        print("Non conforme", chromosome.unfitness)


def hill_climbing_1(chromosome):
    iterations = 0
    actual_unfitness = chromosome.unfitness

    while chromosome.unfitness > actual_unfitness:
        rand_index = 0
        while chromosome.h[rand_index] == 1:
            rand_index = randrange(0, 22) % yachts_quantity
        yachts_host_index_original = chromosome.yachts_host_indexes[rand_index]
        shuffle_array_short_int(chromosome.yachts_host_indexes[rand_index], periods)
        obtain_x(chromosome)
        obtain_m(chromosome)
        calculate_unfitness(chromosome, False)

        if (chromosome.unfitness > actual_unfitness) and (iterations <= hc_iterations):
            chromosome.yachts_host_indexes[rand_index] = yachts_host_index_original
        else:
            if (chromosome.unfitness > actual_unfitness) and (
                iterations > hc_iterations
            ):
                chromosome.yachts_host_indexes[rand_index] = yachts_host_index_original
                obtain_x(chromosome)
                obtain_m(chromosome)
                calculate_unfitness(chromosome, False)

        iterations = iterations + 1


def get_rand_host_index(chromosome, array):
    tmp_array = chromosome.yachts_host_quantity
    added = chromosome.yachts_host_quantity
    actual_index = 0
    i = 0
    while i < yachts_quantity - 1 and (
        actual_index < chromosome.yachts_host_quantity - 1
    ):
        if chromosome.h[i] == 1:
            tmp_array[actual_index] = i
            added[actual_index] = False
            actual_index = actual_index + 1
    i = i + 1

    for i in range(0, periods - 1):
        if chromosome.h[i] == 0:
            index = randrange(0, 22) % chromosome.yachts_host_quantity

        if added[index] == False:
            array[i] = tmp_array[index]
            added[index] = True
        else:
            i = i - 1


def obtain_yachts_host_indexes(chromosome):
    if chromosome.yachts_host_indexes == None:
        chromosome.yachts_host_indexes = np.zeros(42)
        for i in range(0, yachts_quantity):
            chromosome.yachts_host_indexes[i] = np.zeros(periods)
        for i in range(0, pop_size):
            get_rand_host_index(chromosome, chromosome.yachts_host_indexes[i])


def initialize_population(population):
    print("la")
    for i in range(0, pop_size - 1):
        print("la2")
        tmp_pop = copy.deepcopy(population)
        print(i)
        print(len(tmp_pop))

        obtain_h(tmp_pop[i])
        # obtain_yachts_host_indexes(tmp_pop[i])
        obtain_x(tmp_pop[i])
        print("la4")
        obtain_m(tmp_pop[i])
        calculate_unfitness(tmp_pop[i], False)
    population = tmp_pop
    return population


def show_chromosome(chromosome):
    print(chromosome.x)
    print(chromosome.m)
    print(chromosome.h)

    print(chromosome.yachts_host_quantity)
    print(chromosome.yachts_host_indexes)
    print(chromosome.fit)
    print(chromosome.unfitness)
    print(chromosome.faisable)


def show_all_population(population):
    for i in range(0, len(population) - 1):
        show_chromosome(population[i])


def sort_population(population, verb):
    if verb:
        print("Ordonné")
    aux = Chrom()
    for i in range(0, pop_size - 1):
        for j in range(0, pop_size - i - 1):
            if population[j + 1].unfitness < population[j].unfitness:
                aux = population[j + 1]
                population[j + 1] = population[j]
                population[j] = aux

    for i in range(0, pop_size - 1):
        population[i].fit = math.log(pop_size - i + 1) + 1
    if verb:
        print(
            "Le chromosome "
            + i
            + " a fitness de: "
            + population[i].fit
            + " (pas d'aaptitude de "
            + population[i].unfitness
        ),

    if verb:
        if population[i].faisable:
            print("faisable")
        else:
            print("infaisable")


def calculate_next_population(population):
    # print("popilace")
    # print(population)
    acum_fit = copy.deepcopy(population)
    acumulator = 0

    # Calcul de l'aptitude

    for i in range(0, pop_size - 1):
        acumulator = acumulator + population[i].fit
        acum_fit[i] = acumulator
    # print(population)

    tmp_pop = copy.deepcopy(population)

    for i in range(0, pop_size - 1):
        selector = randrange(0, 22) % acumulator + 1
        index = 0

        while acum_fit[index] < selector:
            index = index + 1

        tmp_pop[i] = population[index]

    population = tmp_pop
    for i in range(0, pop_size - 1):
        if prob_hill_climbing_1 > randrange(50, 100) % 100:
            # print("population hill climbing")
            # print(population[i])
            hill_climbing_1(population[i])


def main():
    # clock_t start, end;
    # double cpu_time_used;
    best_solution = 100000
    population = []
    a1 = Chrom
    a2 = Chrom
    a3 = Chrom
    a4 = Chrom
    a5 = Chrom
    a6 = Chrom
    a7 = Chrom
    a8 = Chrom
    a9 = Chrom
    a10 = Chrom
    a11 = Chrom
    a12 = Chrom
    a13 = Chrom
    a14 = Chrom
    a15 = Chrom
    a16 = Chrom
    a17 = Chrom
    a18 = Chrom
    a19 = Chrom
    a20 = Chrom

    population.append(a1)
    population.append(a2)
    population.append(a3)
    population.append(a4)
    population.append(a5)
    population.append(a6)
    population.append(a7)
    population.append(a8)
    population.append(a9)
    population.append(a10)
    population.append(a11)
    population.append(a12)
    population.append(a13)
    population.append(a14)
    population.append(a15)
    population.append(a16)
    population.append(a17)
    population.append(a18)
    population.append(a19)
    population.append(a20)

    # srand(time(0));

    # start = clock();
    if pop_size > 0:
        population = initialize_population(population)
        # show_all_population(population)
        # print("population init")
        # TODO
        sort_population(population, False)
        # print("population sorted")

        best_solution = population[0].unfitness
        print("best solution" + str(best_solution))
        # show_chromosome( & population[0]);

        while population[0].unfitness > periods:
            # TODO
            calculate_next_population(population)
            # print("best next ")
            # print("best population[0].unfitness" +
            #       str(population[0].unfitness))
            # print("periods" + str(periods))
            sort_population(population, False)

            if population[0].unfitness < best_solution:
                # end = clock();
                # cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

                print("SOLUTION RENCONTREE " + population[0].yachts_host_quantity)
                # show_chromosome( & population[0]);
                best_solution = population[0].unfitness
        # print("best solution" + str(best_solution))

    return 0


main()
