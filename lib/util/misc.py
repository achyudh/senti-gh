import numpy as np


def shuffle_list(a):
    np.random.shuffle(a)
    return a


def shuffle_lists(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    return a, b