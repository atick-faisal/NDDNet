# %%
import os

import tensorflow as tf

from utils import (
    DataLoader,
    PerformanceMetrics,
    get_subject_ids,
    run_inference,
    reset_weights
)
from models import NDDNet

# ... Hide tensorflow debug messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ... Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# %% [markdown]
# # Config

# %%
DATA_DIR = "../data/gaitndd/"

FS = 300
N_EPOCHS = 300
LEARNING_RATE = 3e-4
GRF_SEGMENT_LEN = 120
VAL_PERCENTAGE = 0.3
MODE = "combined"

# %% [markdown]
# # Utils

# %%
all_subject_ids = get_subject_ids(DATA_DIR, "ndd")
dataloader = DataLoader(DATA_DIR, all_subject_ids)
metrics = PerformanceMetrics()


# %% [markdown]
# # Model

# %%
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


# %%
DISEASE = "als"
WEIGHT_DIR = os.path.join(f"../weights/{DISEASE}")

metrics = PerformanceMetrics()
subject_ids = get_subject_ids(DATA_DIR, DISEASE)

for test_subject in subject_ids:
    print(f"Running inference on {test_subject} ... ")
    y_true, y_pred = run_inference(
        test_subject=test_subject,
        subject_ids=subject_ids,
        dataloader=dataloader,
        network=network,
        weight_dir=WEIGHT_DIR
    )

    metrics.add_entry(test_subject, y_true, y_pred)
    # model.save_weights(f"../weights/{DISEASE}/{test_subject}", save_format="h5")

    # break

metrics.print_metrics()


# %% [markdown]
# # HD vs HC

# %%
DISEASE = "hunt"
WEIGHT_DIR = os.path.join(f"../weights/{DISEASE}")

metrics = PerformanceMetrics()
subject_ids = get_subject_ids(DATA_DIR, DISEASE)

for test_subject in subject_ids:
    print(f"Running inference on {test_subject} ... ")
    y_true, y_pred = run_inference(
        test_subject=test_subject,
        subject_ids=subject_ids,
        dataloader=dataloader,
        network=network,
        weight_dir=WEIGHT_DIR
    )

    metrics.add_entry(test_subject, y_true, y_pred)
    # model.save_weights(f"../weights/{DISEASE}/{test_subject}", save_format="h5")

    # break

metrics.print_metrics()


# %% [markdown]
# # PD vs HC

# %%
DISEASE = "park"
WEIGHT_DIR = os.path.join(f"../weights/{DISEASE}")

metrics = PerformanceMetrics()
subject_ids = get_subject_ids(DATA_DIR, DISEASE)

for test_subject in subject_ids:
    print(f"Running inference on {test_subject} ... ")
    y_true, y_pred = run_inference(
        test_subject=test_subject,
        subject_ids=subject_ids,
        dataloader=dataloader,
        network=network,
        weight_dir=WEIGHT_DIR
    )

    metrics.add_entry(test_subject, y_true, y_pred)
    # model.save_weights(f"../weights/{DISEASE}/{test_subject}", save_format="h5")

    # break

metrics.print_metrics()


# %% [markdown]
# # NDD vs HC

# %%
DISEASE = "ndd"
WEIGHT_DIR = os.path.join(f"../weights/{DISEASE}")

metrics = PerformanceMetrics()
subject_ids = get_subject_ids(DATA_DIR, DISEASE)

for test_subject in subject_ids:
    print(f"Running inference on {test_subject} ... ")
    y_true, y_pred = run_inference(
        test_subject=test_subject,
        subject_ids=subject_ids,
        dataloader=dataloader,
        network=network,
        weight_dir=WEIGHT_DIR
    )

    metrics.add_entry(test_subject, y_true, y_pred)
    # model.save_weights(f"../weights/{DISEASE}/{test_subject}", save_format="h5")

    # break

metrics.print_metrics()


# %%



