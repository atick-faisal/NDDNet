from tkinter import N
import numpy as np

from typing import Tuple
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class Scaler:
    min: float = 0.0
    max: float = 1.0
    mean: float = 0.0
    std: float = 1.0


def safe_division(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(
        a, b, out=np.zeros_like(a), where=b!=0
    )


def normalize(
    x: NDArray[np.float32],
    scaler: Scaler = None
) -> Tuple[NDArray[np.float32], Scaler]:

    if scaler == None:
        scaler = Scaler()

        scaler.mean = np.mean(x, axis=0)
        scaler.std = np.std(x, axis=0)
        x = safe_division((x - scaler.mean), scaler.std)

        scaler.min = np.min(x, axis=0)
        scaler.max = np.max(x, axis=0)
        x = safe_division((x - scaler.min), (scaler.max - scaler.min))

    else:
        x = safe_division((x - scaler.mean), scaler.std)
        x = safe_division((x - scaler.min), (scaler.max - scaler.min))

    return x, scaler
