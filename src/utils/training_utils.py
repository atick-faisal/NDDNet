import os
import numpy as np
import pandas as pd

from wfdb import rdsamp
from typing import Tuple, List
from rich.progress import track
from numpy.typing import NDArray


from .normalization import Scaler, normalize
from .channel_extraction import get_grf_channels
from .feature_extraction import get_grf_features


def get_data_for_training(
    data_dir: str,
    subject_ids: List[str],
    scaler: Scaler = None
) -> Tuple[
    List[NDArray[np.float32]],
    NDArray[np.uint8],
    Scaler
]:

    subject_description = pd.read_csv(
        os.path.join(data_dir, "subject-description.csv"))

    grf_channels = np.array([], dtype=np.float32)
    grf_features = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.uint8)

    for subject_id in subject_ids:
        
        grf_recording_path = os.path.join(data_dir, subject_id)
        gait_params_path = os.path.join(data_dir, subject_id + ".ts")

        grf_signals = np.nan_to_num(rdsamp(grf_recording_path)[0])
        gait_params = np.loadtxt(gait_params_path, dtype=np.float32)

        subject_mask = subject_description["ID"] == subject_id
        current_subject_description = subject_description[subject_mask].iloc[0]

        _grf_channels = get_grf_channels(subject_id, grf_signals, gait_params)
        _grf_features, _ = get_grf_features(
            gait_params, current_subject_description)

        _label = 0 if "control" in subject_id else 1
        _labels = np.repeat(_label, _grf_channels.shape[0]).astype(np.uint8)

        if grf_channels.size == 0:
            grf_channels = _grf_channels
            grf_features = _grf_features
            labels = _labels
        else:
            grf_channels = np.vstack([grf_channels, _grf_channels])
            grf_features = np.vstack([grf_features, _grf_features])
            labels = np.concatenate([labels, _labels], axis=0)

    grf_features, scaler = normalize(grf_features, scaler)

    n_channels = grf_channels.shape[-1]
    combined_data = np.split(grf_channels, n_channels,
                             axis=-1) + [grf_features]

    return combined_data, labels, scaler
