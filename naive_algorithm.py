from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import random


## Create the matrix
n = 80
m = 10


A = np.random.rand(n,m)

## Chunks
def chunkify(M,R):
    return np.vsplit(M,R)

## Mapper and reducer
def naive_mapper(subM):
    pairs = dict()
    for i in range(len(subM)):
        print(mp.current_process(),i)
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


pool = Pool(mp.cpu_count())
data_chunks = chunkify(A, mp.cpu_count())

## Map
import time
start_time = time.time()
print(mp.cpu_count())
mapped = pool.map(naive_mapper, data_chunks)



## Reduce
M = np.zeros([m,m])
for i in range(m):
  for j in range(m):
    M[i,j] = naive_reducer(i,j,mapped)

print('With pool:', time.time()-start_time,'seconds to execute')   

'''
## Mapper and reducer
def simple_naive(A):
    pairs = dict()
    for j in range(len(A)):
        for k in range(len(A)):
            if (j,k) in pairs:
                pairs[(j,k)] += A[j]*A[k]
            else:
                pairs[(j,k)] = A[j]*A[k]    
    return pairs

def simple_reducer(i,j, pairs):
    return pairs[i,j]

start_time = time.time()

pairs = simple_naive(A)

M2 = np.zeros([m,m])
for i in range(m):
  for j in range(m):
    M2[i,j] = simple_reducer(i,j,pairs)

print('With no pool:', time.time()-start_time,'seconds to execute')   
'''