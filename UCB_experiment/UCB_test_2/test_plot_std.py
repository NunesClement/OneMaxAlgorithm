import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 10, 0.001)
plt.plot(x_axis, norm.pdf(x_axis, 0, 2))

y_axis = np.arange(-5, 5, 0.01)
plt.plot(y_axis, norm.pdf(y_axis, 0, 1))
plt.show()

