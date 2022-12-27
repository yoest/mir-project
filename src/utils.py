import os
import time


def compute_execution_time(fct, args):
    """ Compute the execution time of a function. """
    start = time.time()
    fct(*args)
    end = time.time()
    print(f'Time taken (in seconds): {end - start}')


def compute_file_size(filename):
    """ Compute the size of a file given by its name. """
    file_stats = os.stat(filename)

    print(f'File Size in Bytes is {file_stats.st_size}')
    print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')