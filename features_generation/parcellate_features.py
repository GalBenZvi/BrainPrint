from numpy import mod
import pandas as pd
import tqdm
from pathlib import Path
from utils.features import FEATURES
from utils.parcellation import parcellation_labels, parcellate_metric


def get_available_subjects(mother_dir: Path) -> list:
    """
    Iterate over the main directory to find subjects that should have all necessary files
    Parameters
    ----------
    mother_dir : Path
        Path to project's main directory

    Returns
    -------
    list
        List of subjects' identifiers comprised only of "valid" subjects
    """
    bids_dir, dwi_dir, func_dir = [
        mother_dir / sub_dir
        for sub_dir in [
            "NIfTI",
            "derivatives/dwiprep",
            "derivatives/fmriprep",
        ]
    ]
    valid_subjects = [
        subj.name
        for subj in sorted(bids_dir.glob("sub-*"))
        if (dwi_dir / subj.name).exists() and (func_dir / subj.name).exists()
    ]
    return valid_subjects


def get_dwi_paths(dwi_dir: Path, parameters: list) -> dict:
    """
    Located tensor-derived metrics files within subject's dwiprep outputs
    Parameters
    ----------
    dwi_dir : Path
        Path to subject's dwiprep output
    parameters : list
        A list of tensor-derived parameters to locate

    Returns
    -------
    dict
        A dictionary comprised of tensor-derived metrics files' paths for each of subject's sessions
    """
    subj_dict = {}
    sessions = [ses.name for ses in dwi_dir.glob("ses-*")]
    for ses in sessions:
        subj_dict[ses] = {}
        tensor_dir = dwi_dir / ses / "tensors_parameters" / "coreg_FS"
        for param in parameters:
            fname = tensor_dir / f"{param}.nii.gz"
            if fname.exists():
                subj_dict[ses][param] = fname
            else:
                subj_dict[ses][param] = None
    return subj_dict


def get_smri_paths(anat_dir: Path, parameters: list, subj: str) -> dict:
    """

    Parameters
    ----------
    anat_dir : Path
        [description]
    parameters : list
        [description]
    """
    subj_dict = {"ses-1": {}}
    for param in parameters:
        fname = anat_dir / f"{subj}_{param.lower()}.nii.gz"
        if fname.exists():
            subj_dict["ses-1"][param] = fname
        else:
            subj_dict["ses-1"][param] = None
    return subj_dict


def locate_subject_files(
    mother_dir: Path, subj: str, atlas_name: str = "Brainnetome"
) -> dict:
    """
    Locate subject's needed files for parcellation of features
    Parameters
    ----------
    mother_dir : Path
        Path to project's main directory
    subj : str
        Subject's identifier ("sub-xxx")

    Returns
    -------
    dict
        Dictionary comprised of paths to needed files and their corresponding keys
    """
    subject_dict = {}
    dwi_dir, func_dir = [
        mother_dir / sub_dir / subj
        for sub_dir in ["derivatives/dwiprep", "derivatives/fmriprep"]
    ]
    sessions = [f for f in func_dir.glob("ses-*")]
    if not sessions:
        return
    longitudinal = True if len(sessions) > 1 else False
    anat_dir = (
        func_dir / "anat" if longitudinal else func_dir / sessions[0] / "anat"
    )
    subject_dict["native_parcellation"] = (
        anat_dir / f"{atlas_name}_native_GM.nii.gz"
    )
    subject_dict["Features"] = {}
    for key, sub_dir in zip(FEATURES.keys(), [dwi_dir, anat_dir]):
        if key == "DWI":
            subject_dict["Features"][key] = get_dwi_paths(
                sub_dir, FEATURES.get(key)
            )
        elif key == "SMRI":
            subject_dict["Features"][key] = get_smri_paths(
                anat_dir, FEATURES.get(key), subj
            )
    return subject_dict


def parcellate_subject(
    needed_files: dict,
    atlas_template: pd.DataFrame = parcellation_labels,
    features: dict = FEATURES,
) -> pd.DataFrame:
    """
    Parcellate subject's data into a DataFrame
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
    native_parcellation = needed_files.get("native_parcellation")
    data_dict = needed_files.get("Features")
    sessions = list(data_dict.get("DWI").keys())
    for ses in sessions:
        subject_data = atlas_template.copy()
        for modality in features.keys():
            for metric in features.get(modality):
                if modality == "SMRI":
                    ses_id = "ses-1"
                else:
                    ses_id = ses
                metric_fname = data_dict.get(modality).get(ses_id).get(metric)
                subject_data[metric] = parcellate_metric(
                    metric_fname, native_parcellation, subject_data
                )
        print(subject_data)


if __name__ == "__main__":
    mother_dir = Path("/media/groot/Yalla/media/MRI")
    atlas_name = "Brainnetome"
    out_dir = mother_dir / "derivatives" / "data_processing" / atlas_name
    out_dir.mkdir(exist_ok=True, parents=True)
    valid_subjects = get_available_subjects(mother_dir)
    for subj in tqdm.tqdm(valid_subjects):
        needed_files = locate_subject_files(mother_dir, subj)
        parcellate_subject(needed_files)
        break
