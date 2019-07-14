import numpy as np
import math
import time

# EPS = np.spacing(10)
EPS = 1e-8
SPACING = 10 * EPS
BIG = 1e10


def is_zero(x):
    return np.abs(x) < EPS


def debug(func):
    """debug wrapper
    
    Arguments:
        func {Callable} -- A function/method need to be wrapped for debug message
    
    Returns:
        Callable -- wrapped function/method
    """

    def wrapper(*args, **kwargs):
        print(">>> DEBUG MODE <<<")
        ret = func(*args, **kwargs)
        print(">>> ========== <<<")
        return ret

    return wrapper


def log(func):
    """log wrapper
    
    Arguments:
        func {Callable} -- A function/method need to be wrapped for log message
    
    Returns:
        Callable -- wrapped function/method
    """

    def wrapper(*args, **kwargs):
        args_ = tuple([
            "<ndarray{}>".format(_.shape) if isinstance(_, np.ndarray) else _
            for _ in args[1:]
        ])
        time_str = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
        print("{} | call {}{}:".format(time_str, func.__name__, args_))
        return func(*args, **kwargs)

    return wrapper


def display_progress(msg, current, display_sep, current_id, total, start_time):
    current += display_sep
    percentage = current_id / total
    duration = time.time() - start_time
    eta = duration / percentage - duration
    print(f"{msg} processing {percentage * 100:.0f}%", end=", ")
    print(f"working on {current_id}/{total}", end=", ")
    print(f"eta {formatting_time(eta)}")
    return current


def formatting_time(time_lag):
    hh = int(math.floor(time_lag / 3600))
    mm = int(math.floor((time_lag - hh * 3600) / 60))
    ss = time_lag - hh * 3600 - mm * 60
    return f"{hh}h {mm}min {ss:.2f}s"


def refine_vector(vec):
    return np.reshape(vec, (-1, 1))