"""
Utility functions for extracting individual features from the generated files.
"""
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
import tqdm
from brainprint.feature_generation.cli import parser
from brainprint.feature_generation.subject_results import SubjectResults
from brainprint.feature_generation.utils.features import FEATURES
from brainprint.feature_generation.utils.parcellation import (
    parcellate_metric,
    parcellation_labels,
)
from brainprint.utils import Modality


def parcellate_subject(
    results_dict: dict,
    atlas_template: pd.DataFrame = parcellation_labels,
    features: dict = FEATURES,
) -> pd.DataFrame:
    """
    Parcellate subject's data into a DataFrame.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing locations of needed files
    atlas_template : pd.DataFrame, optional
        Parcellation's template DataFrame, by default parcellation_labels

    Returns
    -------
    pd.DataFrame
        Subject's metrics parcellated into a DataFrame
    """
    data_dict = results_dict.get("Features", {})
    structural_derivatives = data_dict.get(Modality.STRUCTURAL, {})
    native_parcellation = structural_derivatives.get("native_parcellation")
    sessions = list(data_dict.get(Modality.DIFFUSION).keys())
    for session_id in sessions:
        subject_data = atlas_template.copy()
        for modality in features.keys():
            for metric in features.get(modality):
                session_id = (
                    "ses-1" if modality == Modality.STRUCTURAL else session_id
                )
                metric_fname = (
                    data_dict.get(modality).get(session_id).get(metric)
                )
                subject_data[metric] = parcellate_metric(
                    metric_fname, native_parcellation, subject_data
                )
        print(subject_data)


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    out_dir = base_dir / "derivatives" / "data_processing" / args.atlas
    out_dir.mkdir(exist_ok=True, parents=True)
    subjects = (base_dir / "NIfTI").glob("sub-*")
    for subject_dir in tqdm.tqdm(subjects):
        results = SubjectResults(base_dir, subject_dir.name)
        parcellate_subject(results.results_dict)
