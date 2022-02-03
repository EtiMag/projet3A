import os
from venv import create
# force numpy to use only a single processor, by changing the environment of the underlying libraries
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import time
from scipy.stats import unitary_group


def chunkify(Mat, n_split):
    """Splits the matrix Mat into nsplit matrices, returns a list of np.ndarray"""
    (nrow, ncol) = Mat.shape
    indexes = (nrow // n_split) * np.arange(1, n_split)
    Mat_list = np.vsplit(Mat, indexes)
    assert len(Mat_list) == n_split
    return Mat_list


# Create a sparse matrix
def create_big_matrix(nrow_big_matrix, ncol_big_matrix, rank):
    # Generate 2 random unitary matrix
    u, v = unitary_group.rvs(nrow_big_matrix), unitary_group.rvs(ncol_big_matrix)
    
    U, V = np.dot(u, u.conj().T), np.dot(v, v.conj().T)
    

    n = min(ncol_big_matrix,nrow_big_matrix)

    # Then generate a diagonal matrix (singular values) with the same rank as the big matrix
    D = np.zeros((nrow_big_matrix, ncol_big_matrix))
    non_zeros = random.sample([i for i in range(n)], rank)
    for elem in non_zeros:
        D[elem,elem] = 1
    
    # Return the singular value decomposition of the big matrix
    return U.dot(D.dot(V))


def time_basic(big_matrix):
    start_time = time.time()
    result = big_matrix.T@big_matrix
    end_time = time.time()
    return result, end_time - start_time




### We use several norms

#Recall that in finite dimension, all norms are equivalent
def max_diff(matrix1, matrix2):
    """Returns the maximum absolute difference between matrix1 and matrix2 (Linf distance)"""
    return np.max(np.absolute(matrix1 - matrix2))

def norm(Mat):
    """returns an array with the norm of the columns of matrix mat"""
    return np.sqrt(np.sum(np.square(Mat), axis=0))

def distance(mat1, mat2, norm = None):
    # return the distance between 2 matrix using different norms:
    # norm = 
        # 'fro' for the Froebenius norm
        # 'nuc' for the nuclear norm
        #  inf  for the spectral norm
    return np.linalg.norm(mat1-mat2,ord = norm)

