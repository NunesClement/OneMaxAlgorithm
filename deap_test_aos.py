from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# taille du problème
ONE_MAX_LENGTH = 1000
# PM = probability matching

# Paramètres AG
POPULATION_SIZE = 1
P_CROSSOVER = 0.0
P_MUTATION = 1.0
MAX_GENERATIONS = 50
FITNESS_OFFSET = 5

# générateur aléatoire
# RANDOM_SEED = 10
# random.seed(RANDOM_SEED)
random.seed()

#############################################
# Définition dess éléments de base pour l'AG #
#############################################
toolbox = base.Toolbox()

# opérateur qui retourne 0 ou 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# fonction mono objectif qui maximise la première composante de fitness (c'est un tuple)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Classe Individual construite avec un containeur list
creator.create("Individual", list, fitness=creator.FitnessMax)


# initialization des individus avec uniquement des 0
def zero():
    return 0


toolbox.register("individualCreator", tools.initRepeat, creator.Individual, zero, ONE_MAX_LENGTH)

# initialisation de la population
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# Calcul de la fitness/ fonction evaluate
def oneMaxFitness(individual):
    return sum(individual),  # return a tuple


toolbox.register("evaluate", oneMaxFitness)

# opérateurss de variation

# Sélection tournoi taille 3
toolbox.register("select", tools.selTournament, tournsize=3)

# Uniform crossover
toolbox.register("mate", tools.cxUniform, indpb=0.5)

# Mutation Bit-Flip
toolbox.register("bitflip", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)


# mutations 1FLip... n flips
def flip(b):
    if b == 1:
        return 0
    else:
        return 1


def one_flip(individual):
    pos = random.randint(0, ONE_MAX_LENGTH - 1)
    individual[pos] = flip(individual[pos])


def n_flips(individual, n):
    lpos = []
    while len(lpos) < n:
        pos = random.randint(0, ONE_MAX_LENGTH - 1)
        if lpos.count(pos) == 0:
            lpos.append(pos)
    for pos in lpos:
        individual[pos] = flip(individual[pos])


# définition le l'opérateur de mutation par défaut pour l'AG simple
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

# insertion best fitness
# sélectionner le moins bon et le remplacer éventuellement
toolbox.register("worst", tools.selWorst, fit_attr='fitness')


def insertion_best_fitness(population, offspring):
    for ind in offspring:
        worst = toolbox.worst(population, 1)
        if ind.fitness.values[0] > worst[0].fitness.values[0]:
            population.remove(worst[0])
            population.append(ind)
    return population


#########################################
# Outils pour la sélection d'opérateurs #
#########################################

# initialisation des structures de stockage des utilités
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


# calcul de l'amélioration/reward immédiate (plusieurs versions possibles)
def improvement(val_init, val_mut):
    return (val_mut - val_init) + FITNESS_OFFSET
    # return max(0,(val_mut-val_init))
    # return max(0,(val_mut-val_init)/ONE_MAX_LENGTH)
    # return (val_mut-val_init)/ONE_MAX_LENGTH


# calcul de moyenne simple
def update_reward(reward_list, iter, index, value):
    reward_list[index] = ((iter - 1) * reward_list[index] + value) / iter


# sliding window
# Mise à jour de la récompense glissante
def update_reward_sliding(reward_list, reward_history, history_size, index, value):
    if reward_history[index] == [0]:
        reward_history[index] = [value]
    else:
        reward_history[index].append(value)
    if len(reward_history[index]) > history_size:
        reward_history[index] = reward_history[index][1:len(reward_history[index])]
    reward_list[index] = sum(reward_history[index]) / len(reward_history[index])


########################
# Probability matching #
########################

# initialisation de la liste des probabilités
def init_proba_list(taille):
    l = []
    for o in range(taille):
        l.append(1 / taille)
    return l


# mise à jour de la roulette
def update_roulette_wheel(reward_list, proba_list, p_min):
    somme_util = sum(reward_list)
    if somme_util > 0:
        for i in range(len(proba_list)):
            proba_list[i] = p_min + (1 - len(proba_list) * p_min) * (reward_list[i] / (somme_util))
    else:
        proba_list = init_proba_list(len(proba_list))


# sélection d'un opérateur selon une liste de probabilités
def select_op_proba(proba_list):
    r = random.random()
    print(r)
    somme = 0
    i = 0
    while somme < r and i < len(proba_list):
        somme = somme + proba_list[i]
        if somme < r:
            i = i + 1
    return i


# boucle principale d'évolution avec PM
def ea_loop_PM(population, maxFitnessValues, meanFitnessValues, op_history, op_list, p_min, history_size):
    generationCounter = 0
    reward_list = init_reward_list(len(op_list))
    reward_history = init_reward_history(len(op_list))
    proba_list = init_proba_list(len(op_list))
    op_util = []

    # calcul de la fitness tuple pur cchaque individu
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    # extraction dde la liste des valeurs de fitness
    fitnessValues = [individual.fitness.values[0] for individual in population]

    # boucle évolutionnaire
    while (max(fitnessValues) < ONE_MAX_LENGTH) and (generationCounter < MAX_GENERATIONS):
        # MAJ compteur
        generationCounter = generationCounter + 1
        # sélectino opérateur
        current_op = select_op_proba(proba_list)
        # MAJ historique des opérateurs (stats)
        op_util.append(op_list[current_op])
        # élection d'un individu pour l'application de l'op
        offspring = toolbox.select(population, 1)
        # clonage (attention pointeur de liste)
        offspring = list(map(toolbox.clone, offspring))
        # application de la mutation (ici un seul mutant)
        for mutant in offspring:
            # la proba peut être ajusté (mais pas utile)
            if random.random() < P_MUTATION:
                fitness_init = mutant.fitness.values[0]
                # choix de l'op en fonction de son numéro (O=bitflip)
                if current_op > 0:
                    n_flips(mutant, op_list[current_op])
                else:
                    toolbox.bitflip(mutant)
                del mutant.fitness.values
                # calccul nouvelle fitness
                mutant.fitness.values = list(toolbox.evaluate(mutant))
        # MAJ des utilités
        update_reward_sliding(reward_list, reward_history, history_size, current_op,
                              improvement(fitness_init, mutant.fitness.values[0]))
        # MAJ roulette
        update_roulette_wheel(reward_list, proba_list, p_min)
        # on effectue la mutation uniquement si elle est améliorante
        if improvement(fitness_init, mutant.fitness.values[0]) > 0:
            population = insertion_best_fitness(population, offspring)
        # on collecte les valeurs de fitness
        fitnessValues = [ind.fitness.values[0] for ind in population]
        # MAJ des historiques d'utiliation des op (stats)
        for o in range(len(op_list)):
            if o == current_op:
                op_history[o].append(op_history[o][generationCounter - 1] + 1)
            else:
                op_history[o].append(op_history[o][generationCounter - 1])

        # MAJ des données statistiques
        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)


######################
# Sélection avec UCB #
######################

# initialisation strutures
# [0, 0, 0] 3 opérateurs
def init_UCB_val(taille):
    l = []
    for o in range(taille):
        l.append(0)
    return l


# MAJ valeurs UCB
# Le biais pour faire décroitre de façon algorithmique le regret
def update_UCB_val(UCB_val, C, op_history, reward_list, generationCounter):
    for o in range(len(op_history)):
        UCB_val[o] = reward_list[o] + C * np.sqrt(
            generationCounter / (2 * np.log(1 + op_history[o][generationCounter]) + 1))


# sélection operateur
# biais, on prend toujours le premier
# on devrait prendre un au hasard qui a la valeur max et pas juste le premier
# ex 1 2 1 3 1 3 1 2
# faut sélectionner un 3 au hasard
# TODO
def select_op_UCB(UCB_val):
    return UCB_val.index(max(UCB_val))


# boucle évol. UCB (même structure que PM )
# à la place on sélectionne la meilleur valeur UCB
def ea_loop_MAB(population, maxFitnessValues, meanFitnessValues, op_history, op_list, history_size, C):
    generationCounter = 0
    reward_list = init_reward_list(len(op_list))
    reward_history = init_reward_history(len(op_list))
    UCB_val = init_UCB_val(len(op_list))
    op_util = []
    fitnessValues = list(map(toolbox.evaluate, population))

    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue
    fitnessValues = [individual.fitness.values[0] for individual in population]
    while (max(fitnessValues) < ONE_MAX_LENGTH) and (generationCounter < MAX_GENERATIONS):
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

        # print(history_size)
        update_UCB_val(UCB_val, C, op_history, reward_list, generationCounter)

        if improvement(fitness_init, mutant.fitness.values[0]) > 0:
            population = insertion_best_fitness(population, offspring)

        fitnessValues = [ind.fitness.values[0] for ind in population]
        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)


# Appels et statistiques
def main():
    # initialisations des acccummulateurs statistiques
    maxFitnessValues = []
    meanFitnessValues = []
    op_history = []
    # 1 flip puis 3 flips puis 5 flips
    op_list = [1, 3, 5]
    op_history_stat = []
    Max_Fitness_history_stat = []

    # Choix des paramètres propres et de la méthode
    # AOS = 'PM'
    AOS = 'PM'
    p_min = 0.05
    history_size = 10
    C = 4
    # nombre de runs pour les stats
    # moyenne sur plusieurs exec
    NB_RUNS = 10
    # taille de la plus petie éxécution (pour normaliser les figures)
    long_min = MAX_GENERATIONS

    # lancement de l'AG avec AOS 5PM ou UCB)
    for i in range(NB_RUNS):
        maxFitnessValues = []
        meanFitnessValues = []
        op_history = []
        init_op_history(op_history, len(op_list))
        # population initiale (generation 0):
        population = toolbox.populationCreator(n=POPULATION_SIZE)
        if AOS == 'PM':
            ea_loop_PM(population, maxFitnessValues, meanFitnessValues, op_history, op_list, p_min, history_size)
        else:
            ea_loop_MAB(population, maxFitnessValues, meanFitnessValues, op_history, op_list, history_size, C)
        # MAJ des stats
        op_history_stat.append(op_history)
        Max_Fitness_history_stat.append(maxFitnessValues)
        if len(op_history[0]) < long_min:
            long_min = len(op_history[0])

            # utiliation des historiques et agrégation des data
    op_history = []
    maxFitnessValues = []
    init_op_history(op_history, len(op_list))
    for i in range(1, long_min - 1):
        som_fit = 0
        for j in range(NB_RUNS):
            som_fit = som_fit + Max_Fitness_history_stat[j][i]
        maxFitnessValues.append(som_fit / NB_RUNS)
        for o in range(len(op_list)):
            som = 0
            for j in range(NB_RUNS):
                som = som + op_history_stat[j][o][i]
            op_history[o].append(som / NB_RUNS)

    # AG Classique
    # multiple runs
    maxFitness_history = []
    for i in range(NB_RUNS):
        population = toolbox.populationCreator(n=POPULATION_SIZE)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", max)
        population, logbook = algorithms.eaSimple(population, toolbox, P_CROSSOVER, P_MUTATION, MAX_GENERATIONS,
                                                  stats=stats, verbose=False)
        maxFitnessValuesClassic = logbook.select("max")
        maxFitness_history.append(maxFitnessValuesClassic)
    # préparation de data pour AG classique
    Mean_maxFitnessValues_classic = []
    for i in range(MAX_GENERATIONS):
        sum = 0
        for r in range(NB_RUNS):
            sum = sum + maxFitness_history[r][i][0]
        Mean_maxFitnessValues_classic.append(sum / NB_RUNS)

    # Génération d'un graphique
    tab_col = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    sns.set_style("whitegrid")
    for o in range(len(op_list)):
        plt.plot(op_history[o], color=tab_col[o], label=str(op_list[o]) + ' fl.')
    plt.plot(maxFitnessValues, color='black', label='max fitness')
    plt.plot(Mean_maxFitnessValues_classic, color='orange', label='max fitness classic EA')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Max Fitness/Number of application of operators')
    plt.title('Max Fitness over Generations and operators uses for ' + str(AOS) + ' using ' + str(NB_RUNS) + ' runs.')
    plt.show()


if __name__ == '__main__':
    main()
