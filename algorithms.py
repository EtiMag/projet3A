from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
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
def DIMSUM_mapper(subM, gamma):
    pairs = dict()
    for i in range(len(subM)):
        row = subM[i]
        for j in range(len(row)):
            print(j)
            for k in range(len(row)):
                if random.randint(0,1) >= min(1, gamma/(norms_array[j]*norms_array[k])):
                    if (j,k) in pairs:
                        pairs[(j,k)] += row[j]*row[k]
                    else:
                        pairs[(j,k)] = row[j]*row[k]    
    return pairs

def DIMSUM_reducer(i,j, outputs, gamma):
    elem = 0
    for dictionnary in outputs:
        elem += dictionnary[i,j]
    return (1/(min(gamma,(norms_array[i]*norms_array[j]))))*elem



## Create the matrix
n = 10000
m = 100


A = np.random.rand(n,m)
norms_array = norms(A)

pool = Pool(mp.cpu_count())
data_chunks = chunkify(A, mp.cpu_count())


## Map
import time
start_time = time.time()
mapped = pool.map(DIMSUM_mapper, data_chunks)



## Reduce
M = np.zeros([m,m])
for i in range(m):
    for j in range(m):
        M[i,j] = DIMSUM_reducer(i,j,mapped)
print(M)
print('With pool:', time.time()-start_time,'seconds to execute')   
