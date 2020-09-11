from dse_lib import *
import numpy as np

x1 = [2, 3, 0, 1, 0, -3]
x2 = [5, -2, -1, -1, 2, -2]

print(dual_relative_obs_jacobian(x1, x2))
