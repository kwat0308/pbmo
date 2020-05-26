# import sys
# import os
# sys.path.append(os.path.join(os.getcwd(), 'build'))
# sys.path.append('/build/lib.linux-x86_64-3.7')

import funcs
import numpy as np
import matplotlib.pyplot as plt

print(funcs.factorial(3))

size = 10

arr = np.arange(size)
square_arr = funcs.squares(size)

print(arr, square_arr)
plt.plot(arr, square_arr)
plt.show()