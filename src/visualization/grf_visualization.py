import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from wfdb import rdsamp

from typing import Tuple, List
from numpy.typing import NDArray

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 18,
    "text.color": "#212121",
    "axes.edgecolor": "#212121",
    "xtick.color": "#212121",
    "ytick.color": "#212121",
    "axes.labelcolor": "#212121",
    'legend.frameon': False,
})


DATA_DIR = "../../data/dataset/"
FIGURE_DIR = "../../figures/"
FS = 300
SEGMENT_LEN = 512
DISEASE = "park"


subject_description = pd.read_csv(
    os.path.join(DATA_DIR, "subject-description.csv")
)


mean_control = np.load(os.path.join(DATA_DIR, "mean_control_r.npy"))
std_control = np.load(os.path.join(DATA_DIR, "std_control_r.npy"))
mean_ndd = np.load(os.path.join(DATA_DIR, f"mean_{DISEASE}_r.npy"))
std_ndd = np.load(os.path.join(DATA_DIR, f"std_{DISEASE}_r.npy"))


def normalize(x: NDArray[np.float32]) -> NDArray[np.float32]:
    z_score = (x - np.mean(x)) / np.std(x)
    return (z_score - np.min(z_score)) / (np.max(z_score) - np.min(z_score))


def get_file_and_ids(disease: str) -> Tuple[List[str], List[str]]:
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*hea")))
    files = []
    subject_ids = []

    for filename in all_files:
        if filename.find("control") != -1 or filename.find(disease) != -1:
            files.append(filename)
            subject_ids.append(((filename.split("/"))[-1])[:-4])

    return files, subject_ids


def preprocess_data(
    x: NDArray[np.float32]
) -> NDArray[np.float32]:
    x = interpolate(x, SEGMENT_LEN)

    # ... Fix scaling issue
    # if (np.mean(x[-100:]) > -1.0):
    #     x = x * 3.0

    return normalize(x)


def interpolate(
    x: NDArray[np.number],
    target_length: int
) -> NDArray[np.number]:
    x1 = np.arange(x.shape[0])
    x2 = np.linspace(0, x.shape[0], target_length)

    return np.interp(x2, x1, x)


data_files, subject_ids = get_file_and_ids(DISEASE)

control_segments = []
ndd_segments = []
t = np.linspace(0, 100, SEGMENT_LEN)

axes = []

for filename, subject_id in tqdm(zip(data_files, subject_ids), desc="processing ... "):
    counter = {"control": 0, "ndd": 0}
    data = np.nan_to_num(rdsamp(filename[:-4])[0])
    ts = np.loadtxt(filename[:-4] + ".ts", dtype=np.float32)

    for i in range(1, ts.shape[0]):
        end_l = ts[i, 0]
        start_l = end_l - ts[i, 1]

        start_r = start_l - ts[i, 8] + ts[i, 11] / 2.0
        end_r = start_r + ts[i, 2]

        start_l = round(start_l * FS)
        start_r = round(start_r * FS)
        end_l = round(end_l * FS)
        end_r = round(end_r * FS)

        # ... Double Support Time
        dsi_start = round(ts[i - 1, -2] * FS)
        dsi_end = round(ts[i, -2] * FS)

        # if end + dsi > data.shape[0]:
        #     continue

        segment_l = preprocess_data(data[start_l: end_l, 0])
        segment_r = preprocess_data(data[start_r: end_r, 1])

        if "control" in filename:
            control_segments.append(segment_r)
            counter["control"] = counter["control"] + 1
        else:
            ndd_segments.append(segment_r)
            counter["ndd"] = counter["ndd"] + 1

    subject_segments = None

    if "control" in filename:
        subject_segments = np.array(
            control_segments[-counter["control"]:],
            dtype=np.float32
        )

    else:
        subject_segments = np.array(
            ndd_segments[-counter["ndd"]:],
            dtype=np.float32
        )

    subject_mean = np.mean(subject_segments, axis=0)
    subject_std = np.std(subject_segments, axis=0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    ax.plot(t, mean_control, label="Control")
    ax.fill_between(t, mean_control + std_control,
                    mean_control - std_control, alpha=0.3)

    ax.plot(t, mean_ndd, label=DISEASE.upper())
    ax.fill_between(t, mean_ndd + std_ndd,
                    mean_ndd - std_ndd, alpha=0.3)

    ax.plot(t, subject_mean, label=subject_id)
    ax.fill_between(t, subject_mean + subject_std,
                    subject_mean - subject_std, alpha=0.3)

    plt.xlabel("Gait Cycle (\%)")
    plt.ylabel(r"Normalized Force ($N$)")
    plt.title(f"{subject_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"{subject_id}_r.png"))
    plt.close()

    # break

control_segments = np.array(control_segments, dtype=np.float32)
ndd_segments = np.array(ndd_segments, dtype=np.float32)

# mean_control = np.mean(control_segments, axis=0)
# std_control = np.std(control_segments, axis=0)

# mean_ndd = np.mean(ndd_segments, axis=0)
# std_ndd = np.std(ndd_segments, axis=0)

# np.save(os.path.join(DATA_DIR, "mean_control_r"), mean_control)
# np.save(os.path.join(DATA_DIR, "std_control_r"), std_control)
# np.save(os.path.join(DATA_DIR, f"mean_{DISEASE}_r"), mean_ndd)
# np.save(os.path.join(DATA_DIR, f"std_{DISEASE}_r"), std_ndd)
