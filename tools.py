import numpy as np


def chunkify(Mat, nsplit):
    """Splits the matrix Mat into nsplit matrices, returns a list of np.ndarray"""
    (nrow, ncol) = Mat.shape
    indexes = (nrow // nsplit) * np.arange(1, nsplit)
    Mat_list = np.vsplit(Mat, indexes)
    assert len(Mat_list) == nsplit
    return Mat_list


def norm(Mat):
    """returns an array with the norm of the columns of matrix mat"""
    return np.sqrt(np.sum(np.square(Mat), axis=0))


