import os
# force numpy to use only a single processor, by changing the environment of the underlying libraries
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

import pickle
import matplotlib.pyplot as plt


def perf_test_and_plot(list_big_matrix, mapper_naive, reducer_naive, mapper_final, reducer_final, load=True):
    if load:
        list_of_list_time = pickle.load(open("times.pickle", "rb"))
        list_of_list_dist = pickle.load(open("dist.pickle", "rb"))
    else:
        # execute all the tests
        list_time_naive_thread = []
        list_time_naive_process = []
        list_time_final_thread = []
        list_time_final_process = []
        list_time_basic = []
        list_dist_naive_thread = []
        list_dist_naive_process = []
        list_dist_final_thread = []
        list_dist_final_process = []
        n_split = 6
        gamma=1.
        for big_matrix in list_big_matrix:
            print("Handle big matrix shape", big_matrix.shape)
            result_naive_thread, time_naive_thread = execution.execute(big_matrix,
                                                                       mapper=naive.mapper,
                                                                       reducer=naive.reducer,
                                                                       type="T",
                                                                       n_split=n_split)
            print("time_naive_thread=", time_naive_thread)
            result_naive_process, time_naive_process = execution.execute(big_matrix,
                                                                         mapper=naive.mapper,
                                                                         reducer=naive.reducer,
                                                                         type="P",
                                                                         n_split=n_split)
            print("time_naive_process=", time_naive_process)
            result_final_thread, time_final_thread = execution.execute(big_matrix,
                                                                       mapper=final.mapper,
                                                                       reducer=final.reducer,
                                                                       type="T",
                                                                       n_split=n_split,
                                                                       gamma=gamma)
            print("time_final_thread=", time_final_thread)
            result_final_process, time_final_process = execution.execute(big_matrix,
                                                                         mapper=final.mapper,
                                                                         reducer=final.reducer,
                                                                         type="P",
                                                                         n_split=n_split,
                                                                         gamma=gamma)
            print("time_final_process=", time_final_process)
            result_basic, time_basic = tools.time_basic(big_matrix)
            print("time_basic=", time_basic)
            list_time_basic.append(time_basic)
            list_time_naive_thread.append(time_naive_thread)
            list_dist_naive_thread.append(tools.distance(result_naive_thread, result_basic, norm=float("Inf")))
            list_time_naive_process.append(time_naive_process)
            list_dist_naive_process.append(tools.distance(result_naive_process, result_basic, norm=float("Inf")))
            list_time_final_thread.append(time_final_thread)
            list_dist_final_thread.append(tools.distance(result_final_thread, result_basic, norm=float("Inf")))
            list_time_final_process.append(time_final_process)
            list_dist_final_process.append(tools.distance(result_final_process, result_basic, norm=float("Inf")))
            print("Handle big matrix shape", big_matrix.shape, " [OK]")

        # save performance
        list_of_list_time = [list_time_naive_thread, list_time_naive_process, list_time_final_thread, list_time_final_process, list_time_basic]
        list_of_list_dist = [list_dist_naive_thread, list_dist_naive_process, list_dist_final_thread, list_dist_final_process]

        pickle.dump(list_of_list_time, open("times.pickle", "wb"))
        pickle.dump(list_of_list_dist, open("dist.pickle", "wb"))

    list_nrow_log = [np.log2(mat.shape[0]) for mat in list_big_matrix]

    # time and dist plot
    fig = plt.figure(figsize = (20, 10))

    # plot times
    subplot = fig.add_subplot(1, 2, 1)
    names_times = ["naive Thread", "naive Process", "final Thread", "final Process", "basic"]

    for list_of_times, name in zip(list_of_list_time, names_times):
        subplot.plot(list_nrow_log, list_of_times, label=name)
    subplot.legend()
    subplot.set_title("Execution time (s)")

    # plot dists
    subplot = fig.add_subplot(1,2, 2)
    names_dists = ["naive Thread", "naive Process", "final Thread", "final Process"]
    for list_of_dists, name in zip(list_of_list_dist, names_dists):
        subplot.plot(list_nrow_log, list_of_dists, label=name)
    subplot.legend()
    subplot.set_title("Distance to real result (spectral norm)")
    plt.show()