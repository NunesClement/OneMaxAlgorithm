from functools import partial
from collections import namedtuple

import Methods_Genetics;
from typing import List, Optional, Callable, Tuple
#
# a = OneMaxKnapSack.generate_genome(10) #générer un génome de taille 10
# b = OneMaxKnapSack.generate_genome(10)
# print(a)
# print(b)
#
# a = [0,0,0,0,0,0,0]
# b = [1,1,1,1,1,1,1]
# print(OneMaxKnapSack.single_point_crossover(a, b)) # crossover entre 2 gênes
# Population10par10 = OneMaxKnapSack.generate_population(10, 10) # générer une pop de 10 gênomes de taille 10
#
# #muter un gène random
# OneMaxKnapSack.mutation(a, 1, 1)
# print(a)

#fonction de fitness
def fitness(genome: Methods_Genetics.Genome) -> int:
    if len(genome) <= 0 :
        raise ValueError("Le genome doit être > 0 ")
    # count = 0
    return genome.count(1)
    # for i in range(0, len(genome)):
        # if genome[i] == 1:
        #     count = count + 1
    # return count
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

#fitness de la pop globale
# print(OneMaxKnapSack.population_fitness(Population10par10, fitness))
#
# print(OneMaxKnapSack.selection_pair(Population10par10, fitness))

print("-----------")

weight_limit = 10
population, generations = Methods_Genetics.run_evolution(
    #taille pop et taille de genome
    populate_func=partial(Methods_Genetics.generate_population, size=11, genome_length=3000),
    fitness_func=partial(fitness),
    # crossover_func=(OneMaxKnapSack.uniform_crossover),
    crossover_func=(Methods_Genetics.single_point_crossover),
    # bridage de la fitness
    fitness_limit=10000,
    #nombre de générations
    generation_limit=18000
)
print("la meilleur solution " +
      str(Methods_Genetics.greatest(population, fitness))
      + " \n a pour fitness : " +
      str(fitness(Methods_Genetics.greatest(population, fitness)))
      )
# print(population);
# print(OneMaxKnapSack.population_fitness(population, fitness))

