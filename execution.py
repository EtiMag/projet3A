import tools
import multiprocessing as mp

def execute(big_matrix, mapper, reducer, type = "T", n_split = 6):
    """Execute map - reduce algorithm based on input mapper and reducer, using Threads / Processes depending on type"""
    sub_matrix_list = tools.chunkify(big_matrix)

    if type == "P":
        pool = mp.Pool(n_split)

        # map
        mapped = pool.map(mapper, sub_matrix_list)
        # reduce
        result = reducer(mapped)
        return result
    if type == "T":

    if not(isinstance(type, str)):
        raise TypeError("")
    raise(Exception("too bad, type must be either T for threads or P for processes, try again"))
