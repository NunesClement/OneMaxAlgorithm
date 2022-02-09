import math
import numpy as np

y_actual = [12, 7, 7]
y_predicted = [10, 8, 6]

MSE = np.square(np.subtract(y_actual, y_predicted)).mean()

RMSE = math.sqrt(MSE)


print("Différence 2-flips / PM")
print(RMSE)

print("Différence 2-flips / UCB")
print(RMSE)

print("Différence 2-flips / 1-flip")
print(RMSE)

print("Différence 2-flips / 3-flip")
print(RMSE)

print("Différence 2-flips / 5-flip")
print(RMSE)

