import numpy as np
import os

os.environ['MKL_NUM_THREADS'] = '1'
np.__config__.show()


#Recall that in finite dimension, all norms are equivalent
def distance(mat1, mat2, norm = None):
    # return the distance between 2 matrix using different norms:
    # norm = 
        # 'fro' for the Froebenius norm
        # 'nuc' for the nuclear norm
        #  inf  for the spectral norm
    return np.linalg.norm(mat1-mat2,ord = norm)

