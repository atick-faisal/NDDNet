import numpy as np

from numpy.typing import NDArray
from .normalization import normalize
from .lpf import LowPassFilter


def get_outlier_mask(
    x: NDArray[np.float32]
) -> np.ndarray:
    mask = np.ones((x.shape[0], ), dtype="bool")

    for i in range(x.shape[1]):
        values = x[:, i]
        avg = np.mean(values)
        std = np.std(values)
        mask = mask \
            & (values > (avg - 3.0 * std)) \
            & (values < (avg + 3.0 * std))

    # print(f"Number of outliers: {features.shape[0] - np.sum(mask)}")

    return mask


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
