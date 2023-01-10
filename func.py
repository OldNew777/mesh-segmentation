import numpy as np
import time

from mylogger import logger


def length(x: np.ndarray) -> float:
    return np.linalg.norm(x)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / length(x)


def distance(x: np.ndarray, y: np.ndarray) -> float:
    return length(x - y)


def time_it(func):
    def wrapper(*args, **kwargs):
        logger.info(f'Running {func.__name__}...')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"Time taken by {func.__name__} is {end - start:.04f} seconds")
        return result
    return wrapper
