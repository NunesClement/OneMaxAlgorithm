class Seed:
    def __init__(self, seed):
        self.seed = seed


a = Seed(13)


def getSeed():
    return a.seed


def setSeed(new_seed):
    a.seed = new_seed
