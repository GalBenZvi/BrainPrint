import os
from nipype.interfaces import ants
from pathlib import Path

from numpy import fabs
from pandas.core.reshape.merge import merge


def get_subject_files(derivatives_dir: Path, subj_id: str) -> dict:
    """
    Finds subject's relevant files as were derived from FS's analysis pipeline
    Parameters
    ----------
    derivatives_dir : Path
        Path to main derivatives directory (underwhich there will be "freesurfer" and "fmriprep" subdirectories)
    subj_id : str
        Subject's identifier (sub-xxx)

    Returns
    -------
    dict
        A dictionary with keys representing relevant files
    """
    subj_dict = {}
    for hemi in ["lh", "rh"]:
        for val in ["thickness", "volume", "sulc", "white"]:
            subj_dict[f"{hemi}_{val}"] = (
                derivatives_dir
                / "freesurfer"
                / subj_id
                / "surf"
                / f"{hemi}.{val}"
            )
    subj_dict["ribbon"] = (
        derivatives_dir / "freesurfer" / subj_id / "mri" / "ribbon.mgz"
    )
    anat_dir = derivatives_dir / "fmriprep" / subj_id / "anat"
    if anat_dir.exists():
        subj_dict["FS_transform"] = (
            anat_dir / f"{subj_id}_from-fsnative_to-T1w_mode-image_xfm.txt"
        )
        subj_dict["anat"] = anat_dir / f"{subj_id}_desc-preproc_T1w.nii.gz"
        subj_dict["output_dir"] = anat_dir
    else:
        session = [
            ses.name
            for ses in derivatives_dir.glob(f"fmriprep/{subj_id}/ses-*")
        ][0]
        anat_dir = derivatives_dir / "fmriprep" / subj_id / session / "anat"
        subj_dict["FS_transform"] = (
            anat_dir
            / f"{subj_id}_{session}_from-fsnative_to-T1w_mode-image_xfm.txt"
        )
        subj_dict["anat"] = (
            anat_dir / f"{subj_id}_{session}_desc-preproc_T1w.nii.gz"
        )
        subj_dict["output_dir"] = anat_dir
    to_process = True
    flags = [val.exists() for val in subj_dict.values()]
    if not all(flags):
        print(f"Couldn't find neccesary files for {subj_id}:")
        for key, val in subj_dict.items():
            if not val.exists():
                print(key)
        to_process = False
    return subj_dict, to_process


def surface_to_volume(subj_dict: dict, subj_id: str) -> dict:
    """
    Utilizes Freesurfer's mri_surf2vol to transfrom the parametric surface to volumes in subject's space
    Parameters
    ----------
    subj_dict : dict
        [description]

    Returns
    -------
    dict
        [description]
    """
    ribbon = subj_dict.get("ribbon")
    out_dir = ribbon.parent
    for metric in ["thickness", "volume", "sulc"]:
        out_file = out_dir / f"{subj_id}_{metric}.nii.gz"
        subj_dict[f"FS_{metric}"] = out_file
        if out_file.exists():
            continue
        cmd = f"mri_surf2vol --o {out_file}"
        for hemi in ["lh", "rh"]:
            white, val = [
                subj_dict.get(f"{hemi}_{val}") for val in ["white", metric]
            ]
            cmd += f" --so {white} {val}"
        cmd += f" --ribbon {ribbon}"
        print(cmd)
        os.system(cmd)
    return subj_dict


def transform_coords(subj_dict: dict, subj_id: str) -> dict:
    """
    Utilizes ANTs' ApplyTransforms to transform metrics' images to "MNI" coordinates
    Parameters
    ----------
    subj_dict : dict
        [description]
    subj_id : str
        [description]

    Returns
    -------
    dict
        [description]
    """
    out_dir, aff, ref = [
        subj_dict.get(key) for key in ["output_dir", "FS_transform", "anat"]
    ]
    for metric in ["thickness", "volume", "sulc"]:
        input_image = subj_dict.get(f"FS_{metric}")
        out_file = out_dir / f"{subj_id}_{metric}.nii.gz"
        subj_dict[metric] = out_file
        if out_file.exists():
            continue
        at = ants.ApplyTransforms()
        at.inputs.input_image = str(input_image)
        at.inputs.reference_image = str(ref)
        at.inputs.transforms = str(aff)
        at.inputs.output_image = str(out_file)
        print(at.cmdline)
        at.run()
    return subj_dict


if __name__ == "__main__":
    derivatives_dir = Path("/media/groot/Yalla/media/MRI/derivatives")
    subjects = sorted(
        [d.name for d in derivatives_dir.glob("fmriprep/sub-*") if d.is_dir()]
    )
    for subj in subjects:
        try:
            subj_dict, to_process = get_subject_files(derivatives_dir, subj)
            if not to_process:
                continue
            subj_dict = surface_to_volume(subj_dict, subj)
            subj_dict = transform_coords(subj_dict, subj)
        except:
            continue
        # break
