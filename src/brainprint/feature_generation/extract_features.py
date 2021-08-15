"""
Utility functions for extracting individual features from the generated files.
"""
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
import tqdm
from brainprint.feature_generation.utils.features import FEATURES
from brainprint.utils import Modality

from utils.parcellation import parcellate_metric, parcellation_labels


def parcellate_subject(
    needed_files: dict,
    atlas_template: pd.DataFrame = parcellation_labels,
    features: dict = FEATURES,
) -> pd.DataFrame:
    """
    Parcellate subject's data into a DataFrame.

    Parameters
    ----------
    needed_files : dict
        Dictionary containing locations of needed files
    atlas_template : pd.DataFrame, optional
        Parcellation's template DataFrame, by default parcellation_labels

    Returns
    -------
    pd.DataFrame
        Subject's metrics parcellated into a DataFrame
    """
    data_dict = needed_files.get("Features", {})
    structural_derivatives = data_dict.get(Modality.STRUCTURAL, {})
    native_parcellation = structural_derivatives.get("native_parcellation")
    sessions = list(data_dict.get(Modality.DIFFUSION).keys())
    for session_id in sessions:
        subject_data = atlas_template.copy()
        for modality in features.keys():
            for metric in features.get(modality):
                session_id = (
                    FIRST_SESSION
                    if modality == Modality.STRUCTURAL
                    else session_id
                )
                metric_fname = (
                    data_dict.get(modality).get(session_id).get(metric)
                )
                subject_data[metric] = parcellate_metric(
                    metric_fname, native_parcellation, subject_data
                )
        print(subject_data)


if __name__ == "__main__":
    base_dir = Path("/media/groot/Yalla/media/MRI")
    out_dir = base_dir / "derivatives" / "data_processing" / DEFAULT_ATLAS_NAME
    out_dir.mkdir(exist_ok=True, parents=True)
    valid_subjects = generate_preprocessed_subjects(base_dir)
    for subj in tqdm.tqdm(valid_subjects):
        needed_files = locate_subject_files(base_dir, subj)
        parcellate_subject(needed_files)
        break
