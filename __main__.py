import os
# force numpy to use only a single processor, by changing the environment of the underlying libraries
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

# import other files
import execution
import naive
import final
import tools

if __name__ == "__main__":
    # create big matrix (input)
    nrow_big_matrix, ncol_big_matrix = int(1e6), 50
    big_matrix = tools.create_big_matrix(nrow_big_matrix=nrow_big_matrix, ncol_big_matrix=ncol_big_matrix)

    # execute algorithm
    result_parallel, exec_time_parallel = execution.execute(big_matrix=big_matrix,
                                          mapper=naive.mapper,
                                          reducer=naive.reducer,
                                          type="T",
                                          n_split=2)

    # check result shape
    assert result_parallel.shape == (ncol_big_matrix, ncol_big_matrix)

    # display execution time
    print("Execution time parallel=", exec_time_parallel)

    # compare with basic approach
    result_basic, exec_time_basic = tools.time_basic(big_matrix)
    print("Execution time basic=", exec_time_basic)
    print("Max absolute diff between results", tools.max_diff(result_basic, result_parallel))


