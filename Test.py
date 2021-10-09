from functools import partial
from collections import namedtuple

import OneMaxKnapSack;
from typing import List, Optional, Callable, Tuple

a = OneMaxKnapSack.generate_genome(10) #générer un génome de taille 10
b = OneMaxKnapSack.generate_genome(10)
print(a)
print(b)

a = [0,0,0,0,0,0,0]
b = [1,1,1,1,1,1,1]
print(OneMaxKnapSack.single_point_crossover(a, b)) # crossover entre 2 gênes
Population10par10 = OneMaxKnapSack.generate_population(10, 10) # générer une pop de 10 gênomes de taille 10

#muter un gène random
OneMaxKnapSack.mutation(a, 1, 1)
print(a)

#fonction de fitness
def fitness(genome: OneMaxKnapSack.Genome) -> int:
    if len(genome) <= 0 :
        raise ValueError("Le genome doit être > 0 ")
    count = 0
    for i in range(0, len(genome)):
        if genome[i] == 1:
            count = count + 1
    return count

#fitness de la pop globale
print(OneMaxKnapSack.population_fitness(Population10par10, fitness))

print(OneMaxKnapSack.selection_pair(Population10par10, fitness))
# OneMaxKnapSack.run_evolution(OneMaxKnapSack.generate_population, fitness, 8, OneMaxKnapSack.selection_pair, OneMaxKnapSack.single_point_crossover,  OneMaxKnapSack.mutation)

weight_limit = 10
population, generations = OneMaxKnapSack.run_evolution(
    populate_func=partial(OneMaxKnapSack.generate_population, size=10, genome_length=10),
    fitness_func=partial(fitness),
    fitness_limit=10,
    generation_limit=100
)
print(population);
print(OneMaxKnapSack.population_fitness(population, fitness))

# population, generations = OneMaxKnapSack.run_evolution(
#     populate_func=partial(OneMaxKnapSack.generate_population, size=10, genome_length=len(things)),
#     fitness_func=partial(fitness, things=things, weight_limit=weight_limit),
#     fitness_limit=result[0],
#     generation_limit=100
# )