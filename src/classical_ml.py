
import os
import glob
import tsfel
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wfdb import rdsamp
from rich.progress import track
from sklearn.utils import shuffle

from typing import Literal, Tuple, List
from numpy.typing import NDArray
from dataclasses import dataclass

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.seterr(divide='ignore', invalid='ignore')

DATA_DIR = "../data/dataset/"
FS = 300
SEGMENT_LEN = 512
N_FEATURES = 7
N_TRIALS = 7
VAL_PERCENTAGE = 0.3
DISEASE = "ndd"


subject_description = pd.read_csv(
    os.path.join(DATA_DIR, "subject-description.csv")
)


@dataclass
class Scaler:
    min: float
    max: float
    mean: float
    std: float


def normalize(
    x: NDArray[np.float32],
    scaler: Scaler = None
) -> Tuple[NDArray[np.float32], Scaler]:
    if scaler == None:
        scaler = Scaler(
            min=np.min(x, axis=0),
            max=np.max(x, axis=0),
            mean=np.mean(x, axis=0),
            std=np.std(x, axis=0)
        )

    z_score = np.divide((x - scaler.mean), scaler.std)
    return (z_score - scaler.min) / (scaler.max - scaler.min), scaler


def get_subject_ids(
    disease: Literal["als", "hunt", "park", "ndd"]
) -> Tuple[List[str], List[str]]:

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*hea")))
    subject_ids = list(
        map(
            lambda filename: ((filename.split("/"))[-1])[:-4],
            all_files
        )
    )

    if disease == "ndd":
        return subject_ids
    else:
        return list(
            filter(
                lambda subject_id: "control" in subject_id or disease in subject_id,
                subject_ids
            )
        )


def get_demographic_features(subject_description: pd.Series) -> NDArray[np.float32]:
    age = subject_description["AGE(YRS)"]
    weight = subject_description["Weight(kg)"]
    height = subject_description["HEIGHT(meters)"]
    gender = 0 if subject_description["gender"] == "m" else 1
    speed = subject_description["GaitSpeed(m/sec)"]

    return np.array([age, weight, height, gender, speed])


def get_train_val_test_sets(
    test_subject: str,
    subject_ids: List[str]
) -> Tuple[
    NDArray[np.float32],
    NDArray[np.uint8],
    NDArray[np.float32],
    NDArray[np.uint8],
    NDArray[np.float32],
    NDArray[np.uint8],
]:
    train_segments = []
    val_segments = []
    test_segments = []
    train_labels = []
    val_labels = []
    test_labels = []

    cfg = tsfel.get_features_by_domain("temporal")

    for subject_id in subject_ids:
        ts = np.loadtxt(os.path.join(
            DATA_DIR, subject_id + ".ts"), dtype=np.float32)
        demographics = subject_description[subject_description["ID"]
                                           == subject_id].iloc[0]

        ts_features = np.mean(ts[:, 1:], axis=0, dtype=np.float32)
        ts_features = tsfel.time_series_features_extractor(
            cfg,
            ts[:, 1:],
            fs=1,
            window_size=ts.shape[0],
            verbose=0
        ).to_numpy().ravel()

        # print(ts_features.shape)

        dg_features = get_demographic_features(demographics)
        features = np.concatenate([ts_features, dg_features], axis=0)
        # label = 0 if "control" in subject_id else 1

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

        train_subjects = subject_ids.copy()
        train_subjects.remove(test_subject)
        random.shuffle(train_subjects)
        val_subjects = train_subjects[:int(
            len(train_subjects) * VAL_PERCENTAGE)]

        # print(val_subjects)

        if subject_id == test_subject:
            test_segments.append(features)
            test_labels.append(label)
        elif subject_id in val_subjects:
            val_segments.append(features)
            val_labels.append(label)
        else:
            train_segments.append(features)
            train_labels.append(label)

    train_x = np.array(train_segments, dtype=np.float32)
    train_y = np.array(train_labels, dtype=np.uint8)
    val_x = np.array(val_segments, dtype=np.float32)
    val_y = np.array(val_labels, dtype=np.uint8)
    test_x = np.array(test_segments, dtype=np.float32)
    test_y = np.array(test_labels, dtype=np.uint8)

    train_x, scaler = normalize(train_x)
    val_x, _ = normalize(val_x, scaler)
    test_x, _ = normalize(test_x, scaler)

    train_x = np.nan_to_num(train_x)
    val_x = np.nan_to_num(val_x)
    test_x = np.nan_to_num(test_x)

    train_x, train_y = shuffle(train_x, train_y, random_state=42)

    fs = SelectKBest(score_func=mutual_info_classif, k=N_FEATURES)
    fs.fit(train_x, train_y)
    train_x = fs.transform(train_x)
    val_x = fs.transform(val_x)
    test_x = fs.transform(test_x)

    return train_x, train_y, val_x, val_y, test_x, test_y


subject_ids = get_subject_ids(DISEASE)

accuracies = []

for test_subject in subject_ids:
    best_acc = 0.0
    for i in range(N_TRIALS):
        train_x, train_y, _, _, test_x, test_y = get_train_val_test_sets(
            test_subject=test_subject,
            subject_ids=subject_ids
        )

        clf = RandomForestClassifier()
        clf.fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        acc = accuracy_score(test_y.ravel(), pred_y.ravel())

        if acc > best_acc:
            best_acc = acc
        if best_acc == 1:
            break

    accuracies.append(acc)
    print("{:10s} ... {:.02f}".format(test_subject, acc))

    # break

mean_acc = np.mean(accuracies, dtype=np.float32)
print("-" * 20)
print("{:10s} ... {:.02f}".format("Accuracy", mean_acc))
