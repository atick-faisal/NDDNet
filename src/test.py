import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import get_subject_ids, get_data_for_training
from utils.performance_utils import PerformanceMetrics

DATA_DIR = "../data/dataset/"
FS = 300
SEGMENT_LEN = 512
N_FEATURES = 15
N_TRIALS = 7
VAL_PERCENTAGE = 0.3
DISEASE = "als"
MODE = "gait"

subject_ids = get_subject_ids(DATA_DIR, "park")
subject_description = pd.read_csv(
    os.path.join(DATA_DIR, "subject-description.csv"))

metrics = PerformanceMetrics()

for test_subject in subject_ids:
    print("-" * 70 + f"\n{test_subject}\n" + "-" * 70)
    train_subjects = subject_ids.copy()
    train_subjects.remove(test_subject)
    random.shuffle(train_subjects)
    val_subjects = train_subjects[:int(len(train_subjects) * VAL_PERCENTAGE)]

    train_x, train_y, scaler = get_data_for_training(DATA_DIR, train_subjects)
    val_x, val_y, _ = get_data_for_training(DATA_DIR, val_subjects, scaler)
    test_x, test_y, _ = get_data_for_training(DATA_DIR, [test_subject], scaler)

    if MODE == "grf":
        train_x, val_x, test_x = train_x[:-1], val_x[:-1], test_x[:-1]
    else:
        train_x, val_x, test_x = train_x[-1], val_x[-1], test_x[-1]
        train_x = np.squeeze(train_x)
        val_x = np.squeeze(val_x)
        test_x = np.squeeze(test_x)

    train_x, train_y = shuffle(train_x, train_y, random_state=42)
    fs = SelectKBest(score_func=mutual_info_classif, k=N_FEATURES)
    fs.fit(train_x, train_y)
    train_x = fs.transform(train_x)
    val_x = fs.transform(val_x)
    test_x = fs.transform(test_x)

    clf = LogisticRegression()
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    
    metrics.add_entry(test_subject, test_y.ravel(), pred_y.ravel())

    break

metrics.print_metrics()

