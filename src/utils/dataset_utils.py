import os
import glob
from typing import List, Literal, Tuple


def get_file_and_ids(
    dataset_dir: str,
    disease: str
) -> Tuple[List[str], List[str]]:
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*hea")))
    files = []
    subject_ids = []

    for filename in all_files:
        if filename.find("control") != -1 or filename.find(disease) != -1:
            files.append(filename)
            subject_ids.append(((filename.split(os.sep))[-1])[:-4])

    return files, subject_ids


def get_subject_ids(
    dataset_dir: str,
    disease: Literal["als", "hunt", "park", "ndd"]
) -> List[str]:

    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*hea")))
    subject_ids = list(
        map(
            lambda filename: ((filename.split(os.sep))[-1])[:-4],
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
