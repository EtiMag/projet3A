import os
from venv import create
# force numpy to use only a single processor, by changing the environment of the underlying libraries
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import time
import random
import tqdm


def chunkify(Mat, n_split):
    """Splits the matrix Mat into nsplit matrices, returns a list of np.ndarray"""
    (nrow, ncol) = Mat.shape
    indexes = (nrow // n_split) * np.arange(1, n_split)
    Mat_list = np.vsplit(Mat, indexes)
    assert len(Mat_list) == n_split
    return Mat_list


# create a sparse matrix
def create_big_matrix(nrow_big_matrix, ncol_big_matrix, nonzero):
    assert ncol_big_matrix >= nonzero
    A = np.zeros((nrow_big_matrix, ncol_big_matrix))
    for i in tqdm.tqdm(range(nrow_big_matrix)):
        nonzero_index = random.sample(list(range(ncol_big_matrix)), k=nonzero)
        values = np.random.rand(nonzero)
        A[i, nonzero_index] = values
    return A

def time_basic(big_matrix):
    start_time = time.time()
    result = big_matrix.T@big_matrix
    end_time = time.time()
    return result, end_time - start_time


### Define several norms

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

