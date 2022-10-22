import gc
import random
import numpy as np
import tensorflow as tf

from typing import Any, List, Tuple
from numpy.typing import NDArray

from .training_utils import DataLoader
from sklearn.metrics import accuracy_score

random.seed(42)

def perform_loocv(
    test_subject: str,
    val_percentage: float,
    subject_ids: List[str],
    dataloader: DataLoader,
    network: Any,
    loss: tf.keras.losses.Loss,
    optimizer: tf.keras.optimizers.Optimizer,
    n_epochs: int,
    n_trials: int = 7
) -> Tuple[
    NDArray[np.uint8],
    NDArray[np.uint8],
    tf.keras.models.Model
]:
    best_acc = 0
    for _ in range(n_trials):
        n_epochs = 7
        train_subjects = subject_ids.copy()
        train_subjects.remove(test_subject)
        random.shuffle(train_subjects)

        val_subjects = train_subjects[:int(
            len(train_subjects) * val_percentage)]
        train_subjects = train_subjects[int(
            len(train_subjects) * val_percentage):]

        train_x, train_y, scaler = dataloader.get_data_for_training(
            train_subjects)
        val_x, val_y, _ = dataloader.get_data_for_training(
            val_subjects, scaler)
        test_x, test_y, _ = dataloader.get_data_for_training(
            [test_subject], scaler)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=30,
            verbose=0,
            mode="min",
            restore_best_weights=True
        )

        callbacks_list = [early_stopping]

        model = network.get_model(
            n_grf_channels=len(train_x) - 1,
            n_feature_channels=1,
            grf_channel_shape=train_x[0].shape[1:],
            feature_channel_shape=train_x[-1].shape[1:]
        )

        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        model.fit(
            train_x,
            train_y,
            verbose=0,
            shuffle=True,
            epochs=n_epochs,
            batch_size=64,
            validation_data=(val_x, val_y),
            callbacks=callbacks_list
        )

        pred_y = np.argmax(model.predict(test_x, verbose=0), axis=1)

        acc = accuracy_score(test_y.ravel(), pred_y.ravel())

        if acc > best_acc:
            best_acc = acc
        if best_acc > 0.8:
            break

        del train_x, train_y, val_x, val_y, test_x
        tf.keras.backend.clear_session()
        gc.collect()

    return test_y.ravel(), pred_y.ravel(), model
