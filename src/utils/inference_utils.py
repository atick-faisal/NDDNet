import os
import gc
import numpy as np
import tensorflow as tf

from numpy.typing import NDArray
from typing import Tuple, List, Any

from .training_utils import DataLoader


def run_inference(
    test_subject: str,
    subject_ids: List[str],
    dataloader: DataLoader,
    network: Any,
    weight_dir: str
) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:

    train_subjects = subject_ids.copy()
    train_subjects.remove(test_subject)

    _, _, scaler = dataloader.get_data_for_training(train_subjects)
    test_x, test_y, _ = dataloader.get_data_for_training(
        [test_subject], scaler)

    model = network.get_model(
        n_grf_channels=len(test_x) - 1,
        n_feature_channels=1,
        grf_channel_shape=test_x[0].shape[1:],
        feature_channel_shape=test_x[-1].shape[1:]
    )

    model.load_weights(os.path.join(weight_dir, test_subject))
    pred_y = np.argmax(model.predict(test_x, verbose=0), axis=1)

    tf.keras.backend.clear_session()
    gc.collect()

    return test_y.ravel(), pred_y.ravel()
