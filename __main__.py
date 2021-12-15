
import sys


if __name__ == "__main__":
    from algo_naif_Etienne import *
    #test_multiprocessing()
    #test_threading()

    threading_naif(n_thread = 6, nrow_big_matrix=400000, ncol_big_matrix=200)


