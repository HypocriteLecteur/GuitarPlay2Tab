from itertools import tee

def get_two_iters(iter):
    now_it, next_it = tee(iter)
    next(next_it, None)
    return now_it, next_it