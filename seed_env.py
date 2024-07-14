from random import random


class Seed:
    def __init__(self, seed):
        self.seed = seed


a = Seed(round(random() * 10))
# a = 17


def getSeed():
    return a.seed


def setSeed(new_seed):
    a.seed = new_seed
