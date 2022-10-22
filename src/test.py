import os, sys
import gc
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import get_subject_ids, DataLoader, PerformanceMetrics
from model import get_stacked_model

# ... Hide tensorflow grabage messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ... Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


DATA_DIR = "../data/dataset/"
FS = 300
SEGMENT_LEN = 512
N_FEATURES = 15
N_TRIALS = 7
VAL_PERCENTAGE = 0.3
DISEASE = "hunt"
MODE = "gait"

# subject_ids = get_subject_ids(DATA_DIR, DISEASE)
subject_description = pd.read_csv(
    os.path.join(DATA_DIR, "subject-description.csv"))
subject_ids = subject_description.loc[subject_description["gender"] == "f", "ID"].to_list()
subject_ids = list(filter(lambda id: DISEASE in id or "control" in id, subject_ids))

# print(subject_description.loc[subject_description["gender"] == "m", "ID"].to_list())
# sys.exit(0)

loader = DataLoader(DATA_DIR, subject_ids)
metrics = PerformanceMetrics()
ground_truth = []
prediction = []

for test_subject in subject_ids:
    print("-" * 70 + f"\n{test_subject}\n" + "-" * 70)
    best_acc = 0.0
    for i in range(N_TRIALS):
        train_subjects = subject_ids.copy()
        train_subjects.remove(test_subject)
        random.shuffle(train_subjects)
        val_subjects = train_subjects[:int(
            len(train_subjects) * VAL_PERCENTAGE)]
        train_subjects = train_subjects[int(
            len(train_subjects) * VAL_PERCENTAGE):]

        train_x, train_y, scaler = loader.get_data_for_training(train_subjects)
        val_x, val_y, _ = loader.get_data_for_training(val_subjects, scaler)
        test_x, test_y, _ = loader.get_data_for_training(
            [test_subject], scaler)

        # print(train_subjects)
        # print(val_subjects)
        # print(test_subject)
        # print(train_y.shape)
        # print(val_y.shape)
        # print(test_y.shape)

        # break

        if MODE == "grf":
            train_x, val_x, test_x = train_x[:-1], val_x[:-1], test_x[:-1]
        elif MODE == "gait":
            train_x, val_x, test_x = train_x[-1], val_x[-1], test_x[-1]
            train_x = np.squeeze(train_x)
            val_x = np.squeeze(val_x)
            test_x = np.squeeze(test_x)
        else:
            pass

        # train_x, train_y = shuffle(train_x, train_y, random_state=42)
        # fs = SelectKBest(score_func=mutual_info_classif, k=N_FEATURES)
        # fs.fit(train_x, train_y)
        # train_x = fs.transform(train_x)
        # val_x = fs.transform(val_x)
        # test_x = fs.transform(test_x)

        # clf = LogisticRegression()
        # clf.fit(train_x, train_y)
        # pred_y = clf.predict(test_x)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(221, )),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(4, activation="softmax"),
        ])

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        # model = get_stacked_model(
        #     n_conv_channels=2,
        #     n_mlp_channels=0,
        #     n_blocks_conv=3,
        #     n_layers_mlp=0,
        #     conv_channel_width=16,
        #     mlp_channel_width=32,
        #     kernel_size=3,
        #     conv_channel_input_dim=(150, 1),
        #     mlp_channel_input_dim=(221, ),
        #     mode="grf"
        # )

        # early_stopping = EarlyStopping(
        #     monitor="val_loss",
        #     min_delta=0.001,
        #     patience=30,
        #     verbose=0,
        #     mode="min",
        #     restore_best_weights=True
        # )
        # callbacks_list = [early_stopping]

        # model.summary()

        model.fit(
            train_x,
            train_y,
            verbose=0,
            shuffle=True,
            epochs=11,
            batch_size=64,
            # validation_data=(testing_data, test_y),
            validation_data=(val_x, val_y),
            callbacks=[]
        )

        # pred_y = np.round(model.predict(test_x, verbose=0))
        pred_y = np.argmax(model.predict(test_x, verbose=0), axis=1)

        # print(test_y)
        # print(pred_y)

        acc = accuracy_score(test_y.ravel(), pred_y.ravel())

        if acc > best_acc:
            best_acc = acc
        if best_acc == 1:
            break

        del train_x, train_y, val_x, val_y, test_x, model
        gc.collect()
        tf.keras.backend.clear_session()

    metrics.add_entry(test_subject, test_y.ravel(), pred_y.ravel())
    ground_truth.append(mode(test_y.ravel())[0])
    prediction.append(mode(pred_y.ravel())[0])



    # break

metrics.print_metrics()

print(confusion_matrix(ground_truth, prediction))
