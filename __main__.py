import os

# force numpy to use only a single processor, by changing the environment of the underlying libraries
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pickle
# import other files
import execution
import naive
import final
import tools
import performance_tests


if __name__ == "__main__":
    ### first test

    # create big matrix (input)
    nrow_big_matrix, ncol_big_matrix = int(1e3), 50
    big_matrix = tools.create_big_matrix(nrow_big_matrix=nrow_big_matrix, ncol_big_matrix=ncol_big_matrix, nonzero=5)

    # check rank
    print("Big matrix size =", big_matrix.shape)
    print("Big matrix rank =", np.linalg.matrix_rank(big_matrix))

    gamma = 1.

    result_parallel, exec_time_parallel = execution.execute(big_matrix=big_matrix,
                                                            mapper=final.mapper_python,
                                                            reducer=final.reducer_python,
                                                            python_or_np="python",
                                                            type="T",
                                                            n_split=6,
                                                            gamma=gamma)

    # display execution time
    print("Execution time parallel=", exec_time_parallel)

    # compare with basic approach
    result_basic, exec_time_basic = tools.time_basic(big_matrix)
    print("Execution time basic=", exec_time_basic)
    print("Distance between results", tools.distance(result_basic, result_parallel, norm=float("Inf")))

    ## measure performance

    # generate list of big matrix
    # list_nrow = 2 ** np.arange(10, 21, 1)
    # list_ncol = list(50 * np.ones((len(list_nrow, )), dtype=int))  # always 50
    # list_nonzero = list(5 * np.ones((len(list_ncol, )), dtype=int))
    # list_matrix = []
    #
    # for nrow, ncol, nonzero in zip(list_nrow, list_ncol, list_nonzero):
    #     list_matrix.append(tools.create_big_matrix(nrow, ncol, nonzero))
    #
    # pickle.dump(list_matrix, open("list_big_matrix.pickle", "wb"))

    # load list of big matrix
    list_big_matrix = pickle.load(open("list_big_matrix_python.pickle", "rb"))

    performance_tests.perf_test_and_plot(list_big_matrix=list_big_matrix,
                                         mapper_naive=naive.mapper_python,
                                         reducer_naive=naive.reducer_python,
                                         mapper_final=final.mapper_python,
                                         reducer_final=final.reducer_python,
                                         python_or_np="python",
                                         load=False)

