# # S'utilise avec le OneMaxKnapSack
# from functools import partial
#
# import OneMaxKnapSack;
# from collections import namedtuple
#
# Thing = namedtuple('		Avancer async await#
# first_example = [
#     Thing('Laptop', 500, 2200),
#     Thing('Headphones', 150, 160),
#     Thing('Coffee Mug', 60, 350),
#     Thing('Notepad', 40, 333),
#     Thing('Water Bottle', 30, 192),
# ]
#
# second_example = [
#     Thing('Mints', 5, 25),
#     Thing('Socks', 10, 38),
#     Thing('Tissues', 15, 80),
#     Thing('Phone', 500, 200),
#     Thing('Baseball Cap', 100, 70)
# ] + first_example
#
#
# def generate_things(num: int) -> [Thing]:
#     return [Thing(f"thing{i}", i, i) for i in range(1, num+1)]
#
#
# def fitness(genome: OneMaxKnapSack.Genome, things: [Thing], weight_limit: int) -> int:
#     if len(genome) != len(things):
#         raise ValueError("genome and things must be of same length")
#
#     weight = 0
#     value = 0
#     for i, thing in enumerate(things):
#         if genome[i] == 1:
#             weight += thing.weight
#             value += thing.value
#
#             if weight > weight_limit:
#                 return 0
#
#     return value
#
#
# def from_genome(genome: OneMaxKnapSack.Genome, things: [Thing]) -> [Thing]:
#     result = []
#     for i, thing in enumerate(things):
#         if genome[i] == 1:
#             result += [thing]
#
#     return result
#
#
# def to_string(things: [Thing]):
#     return f"[{', '.join([t.name for t in things])}]"
#
#
# def value(things: [Thing]):
#     return sum([t.value for t in things])
#
#
# def weight(things: [Thing]):
#     return sum([p.weight for p in things])
#
#
# def print_stats(things: [Thing]):
#     print(f"Things: {to_string(things)}")
#     print(f"Value {value(things)}")
#     print(f"Weight: {weight(things)}")
#
#
# print_stats(first_example)
#
#
# things = generate_things(22)
# things = second_example
#
# weight_limit = 3000
#
# result = bruteforce(things, weight_limit)
# population, generations = OneMaxKnapSack.run_evolution(
#     populate_func=partial(OneMaxKnapSack.generate_population, size=10, genome_length=len(things)),
#     fitness_func=partial(fitness, things=things, weight_limit=weight_limit),
#     fitness_limit=result[0],
#     generation_limit=100
# )