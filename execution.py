import tools
import multiprocessing as mp
import os
# force numpy to use only a single processor, by changing the environment of the underlying libraries
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import threading as th
import time


def execute(big_matrix, mapper, reducer, type = "T", n_split = 6):
    """Execute map - reduce algorithm based on input mapper and reducer, using Threads / Processes depending on type,
    returns a tuple (result_matrix, execution_time)"""
    (nrow_big_matrix, ncol_big_matrix) = big_matrix.shape
    sub_matrix_list = tools.chunkify(big_matrix, n_split=n_split)
    norms_array = tools.norm(big_matrix)

    # initialize execution time
    start_time = 0
    end_time = 0

    if type == "P":
        pool = mp.Pool(n_split)

        # create list of arguments
        args_list = []
        for sub_mat in sub_matrix_list:
            args_list.append((sub_mat, norms_array.copy()))

        start_time = time.time()
        # map
        mapped = pool.starmap(mapper, args_list)
        # reduce
        result = reducer(mapped)
        end_time = time.time()
        return result, end_time - start_time

    if type == "T":
        thread_list = []
        # allocate output
        sub_output_list = []
        for i in range(n_split):
            sub_output_list.append(np.zeros((ncol_big_matrix, ncol_big_matrix)))
        start_time = time.time()

        # define thread content
        def thread_content(mapper, sub_matrix, sub_output, norms_array):
            """Calls the mapper on the sub_matrix and copies the values in sub_output"""
            sub_output[:, :] = mapper(sub_matrix, norms_array)

        # start threads
        for i in range(n_split):
            args = (mapper, sub_matrix_list[i], sub_output_list[i], norms_array.copy())
            thread_current = th.Thread(target=thread_content, args=args)
            thread_current.start()
            thread_list.append(thread_current)
        # Wait until all threads are finished
        for thread in thread_list:
            thread.join()

        # reduce
        result = reducer(sub_output_list)
        end_time = time.time()
        return result, end_time - start_time

    if not(isinstance(type, str)):
        raise TypeError("type must be an instance of str")
    raise ValueError("type must be either T for threads or P for processes")
