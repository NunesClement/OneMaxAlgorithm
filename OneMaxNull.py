import random

def arrayOfZero():
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def displayer(objectToDisplay):
    print(objectToDisplay)

populationSolution = arrayOfZero()

random.randrange(len(populationSolution))

for x in range(0,  len(populationSolution)):
  choix = populationSolution[random.randrange(len(populationSolution))]
  if choix == 1:
      populationSolution[random.randrange(len(populationSolution))] = 0
  else :
      populationSolution[random.randrange(len(populationSolution))] = 1

displayer(populationSolution)



