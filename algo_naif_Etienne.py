import numpy as np
from multiprocessing import pool
import threading
import time


def test_multiprocessing():
    
    print("test_multiprocessing")
    return None


def test_thread_content(input, output, id_thread):
    print("Thread id", id_thread, "start")
    for row in input:
        print(row)
        time.sleep(0.2)
    output[0, 0] = id_thread
    print("Thread id", id_thread, "end")


def test_threading():

    print("Begin test_threading")

    big_matrix = np.diag(range(1, 7)).dot(np.ones((6, 2)))
    print("big_matrix:")
    print(big_matrix)

    nrow_big_matrix, ncol_big_matrix = big_matrix.shape
    input_th1 = big_matrix[:(nrow_big_matrix//2 + 1), :]
    input_th2 = big_matrix[(nrow_big_matrix//2 + 1):, :]

    output_th1 = np.zeros((2, 2))
    output_th2 = np.zeros((2, 2))

    th1 = threading.Thread(target=test_thread_content, args=(input_th1, output_th1, 1))
    th1.start()
    th1.join()

    th2 = threading.Thread(target=test_thread_content, args=(input_th2, output_th2, 2))
    th2.start()
    th2.join()

    print("Output th1")
    print(output_th1)

    print("Output th2")
    print(output_th2)

    print("End test_threading")


def compute_partial_product(input_mat, output_mat):
    output_mat[:, :] = input_mat.T@input_mat

def threading_naif(n_thread=2, nrow_big_matrix=2000, ncol_big_matrix=20, verbose=True, compare=True):
    print("Threading algo naïf")

    # create matrix
    big_matrix = np.random.rand(nrow_big_matrix, ncol_big_matrix)

    time_start = time.time()
    # Split input for the threads
    indexes = (nrow_big_matrix//n_thread)*np.arange(1, n_thread)
    input_mat_list = np.vsplit(big_matrix, indexes)
    assert len(input_mat_list) == n_thread

    # Allocate output
    output_mat_list = []
    for id_thread in range(1, n_thread+1):
        output_mat_list.append(np.zeros((ncol_big_matrix, ncol_big_matrix)))

    # parallel computing of outputs
    thread_list = []
    for id_thread in range(1, n_thread + 1):
        args = (input_mat_list[id_thread-1], output_mat_list[id_thread-1])
        th = threading.Thread(target=compute_partial_product, args=args)
        th.start()
        thread_list.append(th)


    # wait until all threads are finished
    for th in thread_list:
        th.join()

    # reduce: compute output
    result = sum(output_mat_list)

    # execution time
    time_end = time.time()
    exec_time = time_end - time_start

    # display
    if verbose:
        # print("Big matrix")
        # print(big_matrix)
        # print()
        # print("Splitting indexes = ", indexes)
        # for id_thread in range(1, n_thread + 1):
        #     print("Input thread", id_thread)
        #     print(input_mat_list[id_thread - 1])
        #     print("Output thread", id_thread)
        #     print(output_mat_list[id_thread - 1])
        # print("Final output")
        # print(result)
        print("Execution time (s)")
        print(exec_time)
    if compare:
        time_comp_start = time.time()
        result_compare = big_matrix.T@big_matrix
        time_comp_end = time.time()
        exec_time_comp = time_comp_end - time_comp_start
        # print("Result of basic computation")
        # print(result_compare)
        print("time basic/time parallel", exec_time_comp/exec_time)
