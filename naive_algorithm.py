from multiprocessing import Pool
import numpy as np
import random


## Create the matrix
n = 10000
m = 100
A = np.random.rand(n,m)

## Chunks
def chunkify(M,R):
    return np.vsplit(M,R)

## Mapper and reducer
def naive_mapper(subM):
    pairs = dict()
    for i in range(len(subM)):
        row = subM[i]
        for j in range(len(row)):
            for k in range(len(row)):
                if (j,k) in pairs:
                    pairs[(j,k)] += row[j]*row[k]
                else:
                    pairs[(j,k)] = row[j]*row[k]
    return pairs

def naive_reducer(i,j, outputs):
    elem = 0
    for dictionnary in outputs:
        elem += dictionnary[i,j]
    return elem


pool = Pool(8)
data_chunks = chunkify(A, 8)

## Map
import time
start_time = time.time()
mapped = pool.map(naive_mapper, data_chunks)
print(time.time()-start_time)


## Reduce
M = np.zeros([m,m])
for i in range(m):
  for j in range(m):
    M[i,j] = naive_reducer(i,j,mapped)
    
print(M)