import numpy as np
# All this to draw pretty pictures
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def make_hasher(a, b, p, bucket_num):
    """Returns hash(x) for hash function given by parameters a, b, p and buckets"""
    def hashy(x):
        y = x % p
        hash_val = (a*y+b) % p
        # minus one limits the values to the correct dimensions
        index = int(hash_val % bucket_num)
        return index - 1
    return hashy

def calk_rel_error(F, Fbar):
    """Calculate the error and format it nicely"""
    ids = F[:,0]
    f = F[:,1]
    fbar = Fbar[:,1]
    e = np.zeros(F.shape)
    e[:,0] = ids
    e[:,1] = (fbar-f)/f
    return e


def estimate(dataset_path, counts_path, hash_path, delta, epsilon, p):
    """Perform all the required tasks for hw4 part 2"""
    # generate all the hash functions, specifications in hash_path
    ab_vals = np.genfromtxt(hash_path, delimiter="\t")

    # calculate buckets
    bucket_num = int(np.ceil(np.exp(1)/epsilon))
    hashes = []
    for a, b in ab_vals:
        hashes.append(make_hasher(a, b, p, bucket_num))

    # Make the hash table matrix (actually 5 hash tables)
    hash_tb = np.zeros([len(hashes),bucket_num-1], dtype = int)

    # Loop through the lines of the stream file and increment
    # the appropriate locations in the hash table.
    print("Now hashing values.")
    maxvalue = int(0)
    with open(dataset_path) as f:
        lines=f.readlines()
        for line in lines:
            wid =  int(line)
            for i, hasher in enumerate(hashes):
                # Also track the max value
                if wid > maxvalue:
                    maxvalue = wid
                hash_tb[i, hasher(wid)] += 1

    # Loop over all the values of ID and get the estimate
    print("Now getting estimates.")
    Fbar = np.zeros((maxvalue, 2), dtype= 'int')
    for i in range(1, maxvalue + 1):
        Fbar[i-1] = [i, min([hash_tb[j, h(i)] for j, h in enumerate(hashes)])]
        #Fbar[i] = [int(i+1), int(min([hash_tb[j, h(i)] for j in range(len(hashes))]]


    # Make plots of the results

    # Load the true frequency
    F = np.genfromtxt(counts_path, delimiter="\t", dtype='int')

    # Calculate relative error
    Er = calk_rel_error(F, Fbar)

    print("Now starting figure generation.")

    # Plot of F
    plot_dat = pd.DataFrame(F, columns = ["id", "count"])
    plot1 = sns.distplot(plot_dat['id'], hist_kws={'weights': plot_dat['count']}, bins=len(plot_dat), kde=False)
    plot1.set(xscale="log")
    plot1.set(yscale="log")
    plot1.set(xlabel = "ID (i), log scale")
    plot1.set(ylabel = "Frequency (F[i]), log scale")
    plot1.set(title = "Plot for $F[i]$")
    plot1.figure.savefig('plotF.png')
    print("the figure for F has been generated!")
    # Clear the stage
    plt.clf()

    # Plot of F_bar
    plot_dat = pd.DataFrame(Fbar, columns = ["id", "count"])
    plot2 = sns.distplot(plot_dat['id'], hist_kws={'weights': plot_dat['count']}, bins=len(plot_dat), kde=False)
    plot2.set(xscale="log")
    plot2.set(yscale="log")
    plot2.set(xlabel = "ID (i), log scale")
    plot2.set(ylabel = "Frequency ($\\bar{F}[i]$), log scale")
    plot2.set(title = "Plot for $\\bar{F}[i]$")
    plot2.figure.savefig('plotFbar.png')
    print("the figure for F bar has been generated!")
    # Clear the stage
    plt.clf()

    # Plot of Er
    plot_dat = pd.DataFrame(Er, columns = ["id", "count"])
    plot3 = sns.distplot(plot_dat['id'], hist_kws={'weights': plot_dat['count']}, bins=len(plot_dat), kde=False)
    plot3.set(xscale="log")
    plot3.set(yscale="log")
    plot3.set(xlabel = "ID (i), log scale")
    plot3.set(ylabel = "Frequency ($E_r[i]$), log scale")
    plot3.set(title = "Plot for $E_r[i]$")
    plot3.figure.savefig('plotEr.png')
    print("the figure for E has been generated!")
    # Clear the stage
    plt.clf()


