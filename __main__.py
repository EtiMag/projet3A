import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

if __name__ == "__main__":
    from algo_naif_Etienne import *
    #test_multiprocessing()
    #test_threading()

    threading_naif(n_thread = 3, nrow_big_matrix=70000000, ncol_big_matrix=20)


