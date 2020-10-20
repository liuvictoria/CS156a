# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np

# +
def sample_size_from_M(M, epsilon = .05, delta = .03):
    """
    Rearranges Hoeffding to find the least number of samples required,
    given the other parameters
    """
    const = -1 / (2 * epsilon ** 2)
    N = const * np.log(delta / (2 * M))
    #ceiling divide, recall // operator takes smallest integer
    N = -(-N // 1)
    return N

print ("M = 1 sample size: ", sample_size_from_M(1))
print ("M = 10 sample size: ", sample_size_from_M(10))
print ("M = 100 sample size: ", sample_size_from_M(100))
