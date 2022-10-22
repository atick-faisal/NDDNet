import os
import tensorflow as tf

from utils import (
    DataLoader,
    PerformanceMetrics,
    get_subject_ids,
    perform_loocv,
    reset_weights
)
from models import NDDNet

# ... Hide tensorflow debug messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ... Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


# ------------------------- CONFIG -----------------------

ROOT_DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir)
DATA_DIR = os.path.join(ROOT_DIR, "data", "gaitndd")

FS = 300
N_EPOCHS = 300
LEARNING_RATE = 3e-4
GRF_SEGMENT_LEN = 120
VAL_PERCENTAGE = 0.3
MODE = "combined"

# ---------------------- UTILS -----------------------

all_subject_ids = get_subject_ids(DATA_DIR, "ndd")
dataloader = DataLoader(DATA_DIR, all_subject_ids)
metrics = PerformanceMetrics()

# ----------------------------- MODEL ----------------------------

network = NDDNet(
    n_classes=2,
    n_conv_blocks=3,
    n_mlp_layers=0,
    kernel_size=3,
    conv_channel_width=16,
    mlp_channel_width=32,
    mode="combined"
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


# -------------------------------- ALS -----------------------------

DISEASE = "als"

metrics = PerformanceMetrics()
subject_ids = get_subject_ids(DATA_DIR, DISEASE)

for test_subject in subject_ids:
    print(f"Training on {test_subject} ... ")
    y_true, y_pred, model = perform_loocv(
        test_subject=test_subject,
        val_percentage=VAL_PERCENTAGE,
        subject_ids=subject_ids,
        dataloader=dataloader,
        network=network,
        loss=loss,
        optimizer=optimizer,
        n_epochs=N_EPOCHS
    )

    metrics.add_entry(test_subject, y_true, y_pred)
    # model.save_weights(f"../weights/{DISEASE}/{test_subject}", save_format="h5")

    # break

metrics.print_metrics()


# ---------------------------- HUNT --------------------------------

DISEASE = "hunt"

metrics = PerformanceMetrics()
subject_ids = get_subject_ids(DATA_DIR, DISEASE)

for test_subject in subject_ids:
    print(f"Training on {test_subject} ... ")
    y_true, y_pred, model = perform_loocv(
        test_subject=test_subject,
        val_percentage=VAL_PERCENTAGE,
        subject_ids=subject_ids,
        dataloader=dataloader,
        network=network,
        loss=loss,
        optimizer=optimizer,
        n_epochs=N_EPOCHS
    )

    metrics.add_entry(test_subject, y_true, y_pred)
    # model.save_weights(f"../weights/{DISEASE}/{test_subject}", save_format="h5")

    # break

metrics.print_metrics()


# ----------------------------- PARK -------------------------------

DISEASE = "park"

metrics = PerformanceMetrics()
subject_ids = get_subject_ids(DATA_DIR, DISEASE)

for test_subject in subject_ids:
    print(f"Training on {test_subject} ... ")
    y_true, y_pred, model = perform_loocv(
        test_subject=test_subject,
        val_percentage=VAL_PERCENTAGE,
        subject_ids=subject_ids,
        dataloader=dataloader,
        network=network,
        loss=loss,
        optimizer=optimizer,
        n_epochs=N_EPOCHS
    )

    metrics.add_entry(test_subject, y_true, y_pred)
    # model.save_weights(f"../weights/{DISEASE}/{test_subject}", save_format="h5")

    # break

metrics.print_metrics()


# -------------------------------- NDD -----------------------------

DISEASE = "ndd"

metrics = PerformanceMetrics()
subject_ids = get_subject_ids(DATA_DIR, DISEASE)

for test_subject in subject_ids:
    print(f"Training on {test_subject} ... ")
    y_true, y_pred, model = perform_loocv(
        test_subject=test_subject,
        val_percentage=VAL_PERCENTAGE,
        subject_ids=subject_ids,
        dataloader=dataloader,
        network=network,
        loss=loss,
        optimizer=optimizer,
        n_epochs=N_EPOCHS
    )

    metrics.add_entry(test_subject, y_true, y_pred)
    # model.save_weights(f"../weights/{DISEASE}/{test_subject}", save_format="h5")

    # break

metrics.print_metrics()
