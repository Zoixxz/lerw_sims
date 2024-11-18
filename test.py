# from mpmath import exp, inf, nsum

# d_a = 3.5

# def e1(n):
#     return 1.0 / pow(n, d_a)

# def e2(a):
#     def e3(b):
#         y = pow(4, a) * (8*b + 7)
#         return 1.0 / pow(y, d_a)
#     return nsum(e3, [0, inf])

# sum1 = nsum(e1, [1, inf])
# sum2 = nsum(e2, [0, inf])

# print(sum1 - sum2)

# import bisect
# import numpy as np

# a = np.array([0.5, 0.7])
# print(1 + bisect.bisect_left(a, 0.6))

from scipy.stats import geom

p = 1 - 1.0 / (8192 ** (1.5))
r = geom.rvs(p)
print(r)