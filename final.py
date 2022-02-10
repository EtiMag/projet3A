import numpy as np
import random

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


def reducer(mat_list, norms_array, gamma, ncol_big_matrix):
    return 1/np.minimum(np.outer(norms_array, norms_array), gamma)*sum(mat_list)


def mapper_python(mat, norms_array, gamma):
    pairs = dict()
    for i in range(len(mat)):
        row = mat[i]
        for j in range(len(row)):
            for k in range(len(row)):
                if random.randint(0, 1) >= min(1, gamma / (
                        norms_array[j] * norms_array[k])):  # same as naive algorithm + proba
                    if (j, k) in pairs:
                        pairs[(j, k)] += row[j] * row[k]
                    else:
                        pairs[(j, k)] = row[j] * row[k]
    return pairs


def reducer_python(sub_dict_list, norms_array, gamma, ncol_big_matrix):
    result = np.zeros((ncol_big_matrix, ncol_big_matrix))
    for i in range(ncol_big_matrix):
        for j in range(ncol_big_matrix):
            elem = 0
            for dictionnary in sub_dict_list:
                elem += dictionnary[i, j]
            result[i, j] = (1 / (min(gamma, (norms_array[i] * norms_array[j])))) * elem
    return result
