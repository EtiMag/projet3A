import numpy as np


def mapper(mat, norms_array, gamma):
    return mat.T@mat


def reducer(mat_list, norms_array, gamma, ncol_big_matrix):
    return sum(mat_list)


def mapper_python(subM, norms_array, gamma):
    pairs = dict()
    for i in range(len(subM)):
        row = subM[i]
        for j in range(len(row)):
            for k in range(len(row)):
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
            for dict in sub_dict_list:
                elem += dict[i, j]
            result[i, j] = elem
    return result