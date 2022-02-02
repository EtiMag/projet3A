import numpy as np

def mapper(mat, norms_array, gamma):
    gamma_copy = gamma
    nrow, ncol = mat.shape
    output = np.zeros((ncol, ncol)) # note that ncol << nrow, so the for loops are OK
    for i_output in range(ncol):
        for j_output in range(ncol):
            # randomly choose pairs
            random_values = np.random.rand(nrow)
            probas = gamma_copy/(norms_array[i_output]*norms_array[j_output])*np.ones((nrow,))
            bool_vect = (probas < random_values)
            # sum chosen pairs
            output[i_output, j_output] = np.sum(mat[bool_vect, i_output]*mat[bool_vect, j_output])
    return output


def reducer(mat_list, norms_array, gamma):
    return 1/np.minimum(np.outer(norms_array, norms_array), gamma)*sum(mat_list)