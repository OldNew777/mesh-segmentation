import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.linalg.norm(x - y)
