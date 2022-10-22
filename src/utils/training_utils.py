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


class DataLoader():
    def __init__(self, data_dir: str, subject_ids: List[str]):

        subject_description = pd.read_csv(
            os.path.join(data_dir, "subject-description.csv"))

        self.grf_channels = np.array([], dtype=np.float32)
        self.grf_features = np.array([], dtype=np.float32)
        self.subject_ids = []
        self.labels = []

        for subject_id in subject_ids:

            grf_recording_path = os.path.join(data_dir, subject_id)
            gait_params_path = os.path.join(data_dir, subject_id + ".ts")

            grf_signals = np.nan_to_num(rdsamp(grf_recording_path)[0])
            gait_params = np.loadtxt(gait_params_path, dtype=np.float32)

            subject_mask = subject_description["ID"] == subject_id
            current_subject_description = subject_description[subject_mask].iloc[0]

            _grf_channels = get_grf_channels(
                subject_id, grf_signals, gait_params)
            _grf_features, _ = get_grf_features(
                gait_params, current_subject_description)

            label = 0
            if "control" in subject_id:
                label = 0
            elif "als" in subject_id:
                label = 1
            elif "hunt" in subject_id:
                label = 2
            elif "park" in subject_id:
                label = 3
            else:
                raise(ValueError("invalid subject id"))
                
            self.labels += [label] * _grf_channels.shape[0]
            self.subject_ids += [subject_id] * _grf_channels.shape[0]

            if self.grf_channels.size == 0:
                self.grf_channels = _grf_channels
                self.grf_features = _grf_features
            else:
                self.grf_channels = np.vstack(
                    [self.grf_channels, _grf_channels])
                self.grf_features = np.vstack(
                    [self.grf_features, _grf_features])

    def get_data_for_training(
        self,
        subject_ids: List[str],
        scaler: Scaler = None
    ) -> Tuple[
        List[NDArray[np.float32]],
        NDArray[np.uint8],
        Scaler
    ]:
        labels = np.array(self.labels, dtype=np.uint8)
        all_subject_ids = np.array(self.subject_ids, dtype=np.str_)

        mask = np.zeros((self.grf_channels.shape[0], ), dtype="bool")
        for id in subject_ids:
            mask = mask | (all_subject_ids == id)

        grf_features, scaler = normalize(self.grf_features[mask], scaler)

        n_channels = self.grf_channels.shape[-1]
        combined_data = np.split(self.grf_channels[mask], n_channels,
                                 axis=-1) + [grf_features]

        return combined_data, labels[mask], scaler
