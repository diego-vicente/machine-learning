import numpy as np
import matplotlib.pyplot as plt
import random

# Create a population of 500 greyhounds and 500 labradours
greyhounds = 500
labs = 500

# Greyhounds will be 4 inches taller in average, and the population will have
# 4 inches of deviation.
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

# We can visualize the population in a histogram. Greyhounds will be red, and 
# labradours will be blue. You can check out the png in the file saved
plt.figure(0)
plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.savefig("good-attribute.png")

# To simulate the eye-color property, we can use coin-flipping
grey_eyes = [random.randint(0, 1) for x in range(greyhounds)]
labs_eyes = [random.randint(0, 1) for x in range(labs)]
plt.figure(1)
plt.hist([grey_eyes, labs_eyes], stacked=True, color=['r', 'b'])
plt.savefig("bad-attribute.png")


