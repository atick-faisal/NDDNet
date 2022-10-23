import numpy as np

from typing import Tuple
from numpy.typing import NDArray
from scipy.signal import butter, lfilter


class LowPassFilter(object):
    @staticmethod
    def butter_lowpass(
        cutoff: int,
        fs: int,
        order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    def apply(
        data,
        cutoff=15,
        fs=300,
        order=2,
        axis=-1
    ) -> NDArray[np.float32]:
        b, a = LowPassFilter.butter_lowpass(cutoff, fs, order=order)
        padded_data = np.pad(data, (40, 0), mode="edge")
        return lfilter(b, a, padded_data, axis=axis)[40:]
