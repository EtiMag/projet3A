import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from multiprocessing import Pool
import multiprocessing as mp
import random
from math import sqrt

## Chunks
def chunkify(M,R):
    return np.vsplit(M,R)

## Norms
def norms(M):
    tM = M.T
    norms_array = []
    for row in tM:
        sum = 0
        for elem in row:
            sum += elem*elem
        norms_array.append(sqrt(sum))
    return norms_array

## Mapper and reducer
def DIMSUM_mapper(subM):
    pairs = dict()
    for i in range(len(subM)):
        row = subM[i]
        if i == 0:
            print('process id:',os.getpid())
        for j in range(len(row)):
            for k in range(len(row)):
                if random.randint(0,1) >= min(1, gamma/(norms_array[j]*norms_array[k])): #same as naive algorithm + proba
                    if (j,k) in pairs:
                        pairs[(j,k)] += row[j]*row[k]
                    else:
                        pairs[(j,k)] = row[j]*row[k]    
    return pairs

def DIMSUM_reducer(i,j, outputs):
    elem = 0
    for dictionnary in outputs:
        elem += dictionnary[i,j]
    return (1/(min(gamma,(norms_array[i]*norms_array[j]))))*elem


## Create the matrix
n = 100000
m = 100
gamma = 1

A = np.random.rand(n,m)
norms_array = norms(A)


if __name__ == '__main__':
    nb_process = 8
    pool = Pool(nb_process)
    data_chunks = chunkify(A, nb_process)

    ## Map
    import time
    start_time = time.time()
    print(mp.cpu_count())
    mapped = pool.map(DIMSUM_mapper, data_chunks)

    ## Reduce
    M = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            M[i,j] = DIMSUM_reducer(i,j,mapped)
    print(M)
    print('With pool:', time.time()-start_time,'seconds to execute')   




