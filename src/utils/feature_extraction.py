import tsfel
import numpy as np
import pandas as pd

from typing import Tuple, List
from numpy.typing import NDArray

FEATURE_CONFIG = tsfel.get_features_by_domain("temporal")


def get_demographic_features(subject_description: pd.Series) -> pd.DataFrame:
    age = subject_description["AGE(YRS)"]
    weight = subject_description["Weight(kg)"]
    height = subject_description["HEIGHT(meters)"]
    gender = 0 if subject_description["gender"] == "m" else 1
    speed = subject_description["GaitSpeed(m/sec)"]

    return pd.DataFrame(
        [[age, weight, height, gender, speed]],
        columns=["age", "weight", "height", "gender", "speed"]
    )


def get_gait_features(
    gait_params: NDArray[np.float32]
) -> pd.DataFrame:

    gait_features = tsfel.time_series_features_extractor(
        FEATURE_CONFIG,
        gait_params[:, 1:],
        fs=1,
        window_size=gait_params.shape[0],
        verbose=0
    )

    return gait_features.fillna(0)


def get_grf_features(
    gait_params: NDArray[np.float32],
    subject_description: pd.Series
) -> Tuple[NDArray[np.float32], List[str]]:

    n_gait_cycles = gait_params.shape[0]
    demographic_features = get_demographic_features(subject_description)
    gait_features = get_gait_features(gait_params)
    all_features = pd.concat([demographic_features, gait_features], axis=1)
    grf_features = np.repeat(
        all_features.to_numpy(dtype=np.float32).reshape(1, -1, 1),
        repeats=n_gait_cycles - 1,
        axis=0
    )
    feature_names = all_features.columns

    return grf_features, feature_names
