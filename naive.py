import numpy as np


def mapper(mat, norms_array, gamma):
    return mat.T@mat


def reducer(mat_list, norms_array, gamma):
    return sum(mat_list)
