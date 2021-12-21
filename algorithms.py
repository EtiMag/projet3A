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
    norms_array = norms(M) 
    for i in range(len(subM)):
        row = subM[i]
        for j in range(len(row)):
            for k in range(len(row)):
                if random.randint(0,1) >= min(1, gamma/(norms_array[j]*norms_array[k]))
                    if (j,k) in pairs:
                        pairs[(j,k)] += row[j]*row[k]
                    else:
                        pairs[(j,k)] = row[j]*row[k]    
    return pairs

def DIMSUM_reducer(i,j, outputs, gamma):
    elem = 0
    norms_array = norms(M) 
    for dictionnary in outputs:
        elem += dictionnary[i,j]
    return (1/(min(gamma,(norms_array[i]*norms_array[j]))))*elem