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

DEFAULT_ATLAS_NAME: str = "Brainnetome"
BIDS_DIR_NAME: str = "NIfTI"
DIFFUSION_RELATIVE_PATH: str = "derivatives/dwiprep"
FUNCTIONAL_RELATIVE_PATH: str = "derivatives/fmriprep"
STRUCTURAL_DERIVATIVE_DIR: str = "anat"

SESSION_DIRECTORY_PATTERN: str = "ses-*"
FIRST_SESSION: str = SESSION_DIRECTORY_PATTERN.replace("*", "1")
SUBJECT_DIRECTORY_PATTERN: str = "sub-*"
TENSOR_DIRECTORY_PATH: str = "tensors_parameters/coreg_FS"


def get_diffusion_derivates_path(base_dir: Path, subject_id: str) -> Path:
    return base_dir / DIFFUSION_RELATIVE_PATH / subject_id


def get_functional_derivates_path(base_dir: Path, subject_id: str) -> Path:
    return base_dir / FUNCTIONAL_RELATIVE_PATH / subject_id


def get_structural_derivates_path(base_dir: Path, subject_id: str) -> Path:
    functional = get_functional_derivates_path(base_dir, subject_id)
    sessions = list(functional.glob(SESSION_DIRECTORY_PATTERN))
    if not sessions:
        return
    longitudinal = len(sessions) > 1
    buffer_dir = sessions[0] if longitudinal else ""
    return functional / buffer_dir / STRUCTURAL_DERIVATIVE_DIR


DERIVATIVE_PATH_GETTER: Dict[Modality, str] = {
    Modality.DIFFUSION: get_diffusion_derivates_path,
    Modality.FUNCTIONAL: get_functional_derivates_path,
    Modality.STRUCTURAL: get_structural_derivates_path,
}


def check_subject_derivatives(base_dir: Path, subject_id: str) -> bool:
    """
    Checks whether a given subject has the required derivatives within
    *base_dir*.

    Parameters
    ----------
    base_dir : Path
        Base project directory
    subject_id : str
        Subject ID (and expected directory name)

    Returns
    -------
    bool
        Whether this subject has the required derivates or not
    """
    return all(
        getter(base_dir, subject_id).exists()
        for getter in DERIVATIVE_PATH_GETTER.values()
    )


def generate_preprocessed_subjects(base_dir: Path) -> list:
    """
    Iterate over the main directory to find subjects that have all necessary
    files.

    Parameters
    ----------
    base_dir : Path
        Path to project's main directory

    Returns
    -------
    list
        List of subjects' identifiers comprised only of "valid" subjects
    """
    bids_dir = Path(base_dir) / BIDS_DIR_NAME
    subject_dirs = bids_dir.glob(SUBJECT_DIRECTORY_PATTERN)
    for subject_dir in subject_dirs:
        has_derivates = check_subject_derivatives(base_dir, subject_dir.name)
        if has_derivates:
            yield subject_dir.name


def get_dwi_paths(base_dir: Path, subject_id: str, parameters: list) -> dict:
    """
    Locates tensor-derived metrics files within subject's dwiprep outputs.

    Parameters
    ----------
    base_dir : Path
        Base project directory
    subject_id : str
        Subject ID
    parameters : list
        A list of tensor-derived parameters to locate

    Returns
    -------
    dict
        A dictionary comprised of tensor-derived metrics files' paths for each
        of subject's sessions
    """
    path_getter = DERIVATIVE_PATH_GETTER[Modality.DIFFUSION]
    derivatives_dir = path_getter(base_dir, subject_id)
    subject_derivatives = defaultdict(dict)
    session_dirs = derivatives_dir.glob(SESSION_DIRECTORY_PATTERN)
    for session_dir in session_dirs:
        session_id = session_dir.name
        tensor_dir = derivatives_dir / session_id / TENSOR_DIRECTORY_PATH
        for parameter in parameters:
            derivative_path = tensor_dir / f"{parameter}.nii.gz"
            if derivative_path.exists():
                subject_derivatives[session_id][parameter] = derivative_path
    return subject_derivatives


def get_smri_paths(
    base_dir: Path,
    subject_id: str,
    parameters: list,
    atlas_name: str = DEFAULT_ATLAS_NAME,
) -> dict:
    """
    Returns the paths of structural preprocessing derivatives of the first
    session.

    Parameters
    ----------
    base_dir : Path
        Base project directory
    parameters : list
        List of parameter IDs to be extracted
    subject_id : str
        Subject ID

    Returns
    -------
    dict
        Derivated file path by parameter ID
    """
    path_getter = DERIVATIVE_PATH_GETTER[Modality.STRUCTURAL]
    derivatives_dir = path_getter(base_dir, subject_id)
    subject_derivatives = defaultdict(dict)
    atlas_file = f"{atlas_name}_native_GM.nii.gz"
    subject_derivatives["native_parcellation"] = derivatives_dir / atlas_file
    for parameter in parameters:
        derivative_file = f"{subject_id}_{parameter.lower()}.nii.gz"
        derivative_path = derivatives_dir / derivative_file
        if derivative_path.exists():
            subject_derivatives[FIRST_SESSION][parameter] = derivative_path
    return subject_derivatives


DERIVATES_FROM_MODALITY: Dict[Modality, Callable] = {
    Modality.DIFFUSION: get_dwi_paths,
    Modality.STRUCTURAL: get_smri_paths,
}


def locate_subject_files(
    base_dir: Path, subject_id: str, atlas_name: str = DEFAULT_ATLAS_NAME
) -> dict:
    """
    Locate subject's needed files for parcellation of features.

    Parameters
    ----------
    base_dir : Path
        Path to project's main directory
    subject_id : str
        Subject ID

    Returns
    -------
    dict
        Dictionary comprised of paths to needed files and their corresponding
        keys
    """
    subject_dict = defaultdict(dict)
    anat_dir = DERIVATIVE_PATH_GETTER[Modality.STRUCTURAL](
        base_dir, subject_id
    )
    atlas_file = f"{atlas_name}_native_GM.nii.gz"
    subject_dict["native_parcellation"] = anat_dir / atlas_file
    for modality, parameters in FEATURES.items():
        getter = DERIVATES_FROM_MODALITY[modality]
        subject_dict["Features"][modality] = getter(
            base_dir, subject_id, parameters
        )
    return subject_dict


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
