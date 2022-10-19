import numpy as np

from numpy.typing import NDArray
from utils.normalization import normalize


def interpolate(
    x: NDArray[np.number],
    target_length: int
) -> NDArray[np.number]:
    x1 = np.arange(x.shape[0])
    x2 = np.linspace(0, x.shape[0], target_length)
    return np.interp(x2, x1, x)


def preprocess_grf(
    x: NDArray[np.float32],
    segment_length: int
) -> NDArray[np.float32]:
    x = interpolate(x, segment_length)
    x, _ = normalize(x, scaler=None)
    return x
