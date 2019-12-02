import numpy as np

dataset_path = "words_stream.txt" # each line corresponding to the ID of a word in the stream.
counts_path = "counts.txt"  # each line is a pair of numbers separated by a tab.
            # The first number is an ID of a word
            # the second number is its associated exact frequency count in the stream.
hash_path = "hash_params.txt" # each line is a pair of numbers separated by a tab, corresponding to
                #parameters a and b for the hash functions (See explanation below).
p = 123457
dataset = np.genfromtxt(dataset_path)
counts = np.genfromtxt(counts_path, delimiter="\t")
ab_vals = np.genfromtxt(hash_path, delimiter="\t")

def get_data(xpath, ypath):
    x = np.genfromtxt(xpath, delimiter=",")
    y = np.genfromtxt(ypath)
    y = y.reshape(len(y),1)
    return np.concatenate((x,y), axis=1)

xy = get_data(xpath, ypath)
gx = lambda xy :  xy[list(range(len(xy)-1))]
gy = lambda xy :  xy[len(xy)-1]

gx(xy[1,:])
gy(xy[1,:])

stimate(dataset_path, counts_path, hash_path, delta, epsilon, p)
k
