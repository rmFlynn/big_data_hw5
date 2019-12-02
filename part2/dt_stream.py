import numpy as np

dataset_path = "words_stream.txt" # each line corresponding to the ID of a word in the stream.

counts_path = "counts.txt"  # each line is a pair of numbers separated by a tab.
                            # The first number is an ID of a word
                            # the second number is its associated exact frequency count in the stream.

hash_path = "hash_params.txt" # each line is a pair of numbers separated by a tab, corresponding to
                              #parameters a and b for the hash functions (See explanation below).
delta = np.exp(-5)
epsilon = np.exp(1)*(10**(-4))
p = 123457



def make_hasher(a, b, p, bucket_num):
    # Returns hash(x) for hash function given by parameters a, b, p and buckets
    def hashy(x):
        y = x % p
        hash_val = (a*y+b) % p
        return hash_val % bucket_num
    return hashy

ab_vals = np.genfromtxt(hash_path, delimiter="\t")
bucket_num = np.ceil(np.exp(1)/epsilon)
hashes = []
for a, b in ab_vals:
    hashes.append(make_hasher(a, b, p, bucket_num))

hashes[0](55)
#dataset = np.genfromtxt(dataset_path)

with open(dataset_path) as f:
    lines=f.readlines()
    for line in lines:
        #myarray = np.fromstring(line, dtype=float, sep=',')
        print(myarray)

counts = np.genfromtxt(counts_path, delimiter="\t")



estimate(dataset_path, counts_path, hash_path, delta, epsilon, p)



