import numpy as np


def mapper(mat, norms_array):
    return mat.T@mat


def reducer(mat_list):
    return sum(mat_list)
