# import numpy as np
# from random import randint, random, seed
# from math import exp
# from classPpp import *
# from itertools import product, starmap, islice
#
# seed(5)
#
#
# def set_list_boats():
#     if len(boats_capacity) != len(nb_boats_per_capacity):
#         raise ValueError("err in two needed arrays in set_list_boats")
#     for bc in range(0, len(boats_capacity)):
#         for nbc in range(0, nb_boats_per_capacity[bc]):
#             list_boats.append(HostBoat(boats_capacity[bc]))
#     if len(list_boats) != H:
#         raise ValueError("err in the array builted in set_list_boats")
#
#
# def set_list_crews():
#     if len(size_crew) != len(nb_crews_per_size):
#         raise ValueError("err in two needed arrays in set_list_crews")
#     for sc in range(0, len(size_crew)):
#         for nbc in range(0, nb_crews_per_size[sc]):
#             list_crews.append(Crew(size_crew[sc]))
#     if len(list_crews) != G:
#         raise ValueError("err in the array builted in set_list_crews")
#
#
# set_list_crews()
# set_list_boats()
#
#
# # display list crews properly
# def display_list_crews():
#     for i in range(0, len(list_crews)):
#         print("Crew " + str(list_crews[i].id) + " - size of " +
#               str(list_crews[i].size))
#
#
# def display_single_crew(g):
#     print("Crew " + str(g.id) + " - size of " + str(g.size))
#
#
# # display_list_crews()
#
#
# # display list boats properly
# def display_list_boats():
#     for i in range(0, len(list_boats)):
#         print("Boat " + str(list_boats[i].id) + " - capacity of " +
#               str(list_boats[i].capacity))
#
#
# def display_single_boat(h):
#     print("Boat " + str(h.id) + " - capacity of " + str(h.capacity))
#
#
# # display_list_boats()
#
#
# # set configurations properly
# # TODO remettre en random
# def set_configuration_properly():
#     for i in range(0, len(periode_temps)):
#         for j in range(0, len(list_crews)):
#             # Affectation à un bateau random
#             m = randint(0, len(list_boats) - 1)  # a modif en mouvement
#             # S.append(Configuration(list_crews[j], periode_temps[i], list_boats[12]))
#             S.append(Configuration(list_crews[j], periode_temps[i], list_boats[m]))
#
#
# set_configuration_properly()
#
#
# # display configurations properly
# def display_multiple_configurations(lc=None):
#     if lc is None:
#         lc = S
#
#     for i in range(0, len(lc)):
#         print("x[" + str(lc[i].Crew.id) + " , " + str(
#             lc[i].time_period) + "]" + " = " + str(lc[i].Boat.id))
#
#
# def transform_configuration_into_search_space(lc=None):
#     if lc is None:
#         lc = S
#     remplissage = []
#     for i in range(0, len(lc)):
#         for j in range(0, coupureTableau):
#             remplissage.append(lc[i])
#         SearchSpace.append(remplissage)
#         remplissage = []
#
#
# transform_configuration_into_search_space()
#
#
# # display searchspace properly
# def display_search_space_properly(lc=None):
#     if lc is None:
#         lc = SearchSpace
#     print_string = ""
#     for i in range(0, len(lc)):
#         for j in range(0, len(lc[i])):
#             print_string = print_string + " x[" + str(lc[i][j].Crew.id) + " , " + str(
#                 lc[i][j].time_period) + "]" + ": " + str(lc[i][j].Boat.id)
#         print(print_string)
#         print_string = ""
#
#
# print("--")
# display_search_space_properly()
# print("--")
#
#
# def find_Neighbors_in_searchspace(x, y):
#     transform_configuration_into_search_space()
#     xi = (0, -1, 1) if 0 < x < len(SearchSpace) - 1 else ((0, -1) if x > 0 else (0, 1))
#     yi = (0, -1, 1) if 0 < y < len(SearchSpace[0]) - 1 else ((0, -1) if y > 0 else (0, 1))
#     return islice(starmap((lambda a, b: SearchSpace[x + a][y + b]), product(xi, yi)), 1, None)
#
#
# n = list(find_Neighbors_in_searchspace(0, 0))
# print(n)
# display_multiple_configurations(n)
#
#
# def display_single_configuration(sc):
#     print("x[" + str(sc.Crew.id) + " , " + str(
#         sc.time_period) + "]" + " = " + str(sc.Boat.id))
#
#
# def access_single_configuration(sc):
#     return sc.Crew.id, sc.time_period, sc.Boat.id
#
#
# # display_multiple_configurations()
#
#
# def c(g):
#     return g.size
#
#
# def C(hb):
#     return hb.capacity
#
#
# # R&D global
# # Passer par des objets mais devoir tout charger
# # Passer par des id et find plus facilement
# # Passer par des tableaux
# # Possible comment ? Mind
#
# # wip : redo in funcitonal way
# # give a crew and configurations are returned
# def findConfigurationsByCrew(g, list_search=None):
#     if list_search is None:
#         list_search = S
#     configurations_returned = []
#     for lc in list_search:
#         if lc.Crew.id == g.id:
#             configurations_returned.append(lc)
#     return configurations_returned
#
#
# # display_configuration_properly(findConfigurationsByCrew(list_crews[3]))
#
#
# # give a boat and configurations are returned
# def findConfigurationsByBoat(b, list_search=None):
#     if list_search is None:
#         list_search = S
#     configurations_returned = []
#     for lc in list_search:
#         if lc.Boat.id == b.id:
#             configurations_returned.append(lc)
#     return configurations_returned
#
#
# # display_configuration_properly(findConfigurationsByBoat(list_boats[3]))
#
# # give a time periode and configurations are returned
# def findConfigurationsByTimePeriod(p, list_search=None):
#     if list_search is None:
#         list_search = S
#     configurations_returned = []
#     for lc in list_search:
#         if lc.time_period == p:
#             configurations_returned.append(lc)
#     return configurations_returned
#
#
# # display_configuration_properly(findConfigurationsByTimePeriod(periode_temps[0]))
#
#
# # Un crew guest existe ou pas à deux temps différents
# def DIFF(g):
#     print("---DIFF---")
#     configurations_finded = findConfigurationsByCrew(g)
#     if not configurations_finded:
#         return 0
#     else:
#         # display_multiple_configurations(configurations_finded)
#         max = 0
#         for cf in configurations_finded:
#             count = 1
#             for cf2 in configurations_finded:
#                 if cf.Boat == cf2.Boat and cf.id != cf2.id:
#                     count = count + 1
#             if count > max:
#                 max = count
#             # print("Le crew guest est a " + str(count) + " au temps : " + str(cf.time_period))
#         return max
#         # for cf in configurations_finded:
#         #     # print("Le crew guest est en double au temps " + str(cf.time_period))
#         #     display_single_configuration(cf)
#         #
#         # return len(configurations_finded)
#
#
# print(list_crews[2].id)
# DIFF(list_crews[2])
#
#
# def EACH_DIFF():
#     for g in list_crews:
#         DIFF(g)
#     return 0
#
#
# # EACH_DIFF()
#
#
# def PENALTY_DIFF(g):
#     return DIFF(g)
#
#
# # EACH_DIFF()
#
#
# # display_configuration_properly(findConfigurationsByCrew(list_crews[3]))
# #
# # DIFF(list_crews[3])
#
# # Les crew doivent se rencontrer <= 1 fois (pour un couple)
# def ONCE(g1, g2):  # param : crew1 crew2
#     print("---ONCE---")
#     count_meet = 0
#     for t in periode_temps:
#         fcpt = findConfigurationsByTimePeriod(t)
#         g1f = findConfigurationsByCrew(g1, fcpt)
#         g2f = findConfigurationsByCrew(g2, fcpt)
#
#         if len(g1f) > 1 or len(g2f) > 1:
#             raise ValueError("g1f or g2f need more comprehension")
#         # print(g1f[0].Boat.id)
#         # print(g2f[0].Boat.id)
#         # print(g1f[0].Boat == g2f[0].Boat)
#
#         if g1f[0].Boat == g2f[0].Boat:
#             count_meet = count_meet + 1
#             # print(" Rencontre sur " + str(g1f[0].Boat.id))
#             # print("G1 et G2 se rencontrent " + str(count_meet) + " fois sur le bateau " + str(g2f[0].Boat.id))
#     # print("count_meet : " + str(count_meet))
#     return count_meet
#
#
# # display_multiple_configurations(findConfigurationsByCrew(list_crews[2]))
# # print("---")
# # display_multiple_configurations(findConfigurationsByCrew(list_crews[1]))
#
# ONCE(list_crews[2], list_crews[1])
#
#
# def PENALTY_ONCE():
#     meet = ONCE(g1, g2)
#     if meet <= 1:
#         return 0
#     else:
#         return meet - 1
#
#
# def CAPA(h, t):
#     # print("---CAPA---")
#     cg = 0
#     ch = C(h)
#     for g in list_crews:
#         fcc = findConfigurationsByCrew(g)
#         fcb = findConfigurationsByBoat(h, fcc)
#         fct = findConfigurationsByTimePeriod(t, fcb)
#         # display_multiple_configurations(fct)
#         if len(fct) >= 1:
#             cg = cg + c(g)
#     print("c(g) = " + str(cg) + "  / C(h) = " + str(ch))
#     if cg > ch:
#         print(False)
#     else:
#         print(True)
#     return ch - cg
#     # if cg > ch:
#     #     return False
#     # return True
#
#
# # print("---------------")
# # fcb = findConfigurationsByBoat(list_boats[12])
# # fct = findConfigurationsByTimePeriod(periode_temps[0], fcb)
# # display_multiple_configurations(fct)
#
# print(CAPA(list_boats[0], periode_temps[1]))
# # print(c(list_crews[14]) + c(list_crews[15]))
# print(CAPA(list_boats[10], periode_temps[0]))
#
#
# def PENALTY_CAPA(h, t):
#     # TODO : a comprendre !!!!
#     capa = CAPA(h, t)
#     if not capa <= 0:
#         return 0
#     else:
#         return 1 + (capa - 1) / 4
#
#
# # TODO fonction de couts => weighting
# def ponderation_fonctions_couts(configuration):
#     return PENALTY_DIFF(configuration) * 4 + PENALTY_ONCE(configuration) * 2 + PENALTY_CAPA(configuration) * 4
#
#
# # M ne devra pas etre un entier mais un mouvement
#
# M = 50
#
#
# # Algo à revoir
# def metropolis_algorithm(temp, N):
#     notstop = 0
#     sigma = 0  # faux
#     while notstop == 0:
#         m = randint(0, M)  # a modif en mouvement
#         sigma = fonction_cout(m) - fonction_cout(m)
#         if sigma <= 0:
#             s = m
#         else:
#             if random() < exp(-sigma / temp):
#                 s = m
#
# # Fonctions de voisinnage
#
# # oneMove : changer de bateau
# # def oneMove(configurationAModifier):
# #     list(find_Neighbors_in_searchspace(configurationAModifier))
#
#
# # def findNeighbors(grid, x, y):
# #     xi = (0, -1, 1) if 0 < x < len(grid) - 1 else ((0, -1) if x > 0 else (0, 1))
# #     yi = (0, -1, 1) if 0 < y < len(grid[0]) - 1 else ((0, -1) if y > 0 else (0, 1))
# #     return islice(starmap((lambda a, b: grid[x + a][y + b]), product(xi, yi)), 1, None)
# #
# # grid = [[ 0,  1,  2,  3],
# #      [ 4,  5,  6,  7],
# #      [ 8,  9, 10, 11],
# #      [12, 13, 14, 15]]
# #
# # n = list(findNeighbors(grid, 2, 1))   # find neighbors of 9
# # n = list(findNeighbors(grid, 0, 0))   # find neighbors of 9
#
# # descente recherche localedescente recherche locale
# # $ Barbara's paper says 39 boats, max crew size 7
# # $ all data in Table 1 of the paper and in csplib
# # $ csplib says 6 time periods. For the first 5 boats, 4 periods is the max feasible.
# #
# # letting n_boats be 5
# # letting n_periods be 4
# # letting capacity be function(1 --> 6,
# #                              2 --> 8,
# #                              3 --> 12,
# #                              4 --> 12,
# #                              5 --> 12)
# # letting crew be function(1 --> 2,
# #                          2 --> 2,
# #                          3 --> 2,
# #                          4 --> 2,
# #                          5 --> 4)
#
# # faire un système de navigation voisinnage            Finir fouille