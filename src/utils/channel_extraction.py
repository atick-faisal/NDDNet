import numpy as np

from rich.progress import track
from numpy.typing import NDArray
from utils.preprocessing import preprocess_grf


def get_grf_channels(
    subject_id: str,
    grf_signals: NDArray[np.float32],
    gait_params: NDArray[np.float32],
    segment_length: int = 150,
    sampling_rate: int = 300
) -> NDArray[np.float32]:

    grf_channels = np.array([], dtype=np.float32)

    for i in track(range(1, gait_params.shape[0]), "{:11s}".format(subject_id)):
        end_l = gait_params[i, 0]
        start_l = end_l - gait_params[i, 1]

        start_r = start_l - gait_params[i, 8] + gait_params[i, 11] / 2.0
        end_r = start_r + gait_params[i, 2]

        start_l = round(start_l * sampling_rate)
        start_r = round(start_r * sampling_rate)
        end_l = round(end_l * sampling_rate)
        end_r = round(end_r * sampling_rate)

        segment_l = preprocess_grf(
            grf_signals[start_l: end_l, 0], segment_length)
        segment_r = preprocess_grf(
            grf_signals[start_r: end_r, 1], segment_length)

        _grf_channels = np.dstack([segment_l, segment_r])

        if grf_channels.size == 0:
            grf_channels = _grf_channels
        else:
            grf_channels = np.vstack([grf_channels, _grf_channels])

    return grf_channels
