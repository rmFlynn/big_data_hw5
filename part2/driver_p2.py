import numpy as np
from dt_stream import estimate

# variable declaration
dataset_path = "words_stream.txt"
counts_path = "counts.txt"
hash_path = "hash_params.txt"
delta = np.exp(-5)
epsilon = np.exp(1)*(10**(-4))
p = 123457

# execute the program
estimate(dataset_path, counts_path, hash_path, delta, epsilon, p)
