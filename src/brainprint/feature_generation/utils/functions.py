import os
import numpy as np
import nibabel as nib
from nipype.interfaces.ants import ApplyTransforms
from pathlib import Path


def at_ants(
    in_file: Path,
    ref: Path,
    xfm: Path,
    outfile: Path,
    nn: bool,
    invert_xfm: bool = False,
):
    """
    Apply pre-calculated transformations between images of different spaces
    Parameters
    ----------
    in_file : Path
        Path to the "moving" file
    ref : Path
        Path to the "static" file
    xfm : Path
        Path to a pre-calculated transformation
    outfile : Path
        Path to output file
    nn : bool
        Whether to use Nearest Neighbout interpolation (for atlas registrations)
    invert_xfm : bool, optional
        Whether to invert the transformation file before applying it, by default False
    """
    at = ApplyTransforms()
    at.inputs.input_image = in_file
    at.inputs.reference_image = ref
    at.inputs.transforms = xfm
    at.inputs.output_image = str(outfile)
    if nn:
        at.inputs.interpolation = "NearestNeighbor"
    if invert_xfm:
        at.inputs.invert_transform_flags = True
    at.run()


def crop_to_mask(in_file: Path, mask: Path, out_file: Path):
    """
    Crops an image according to a given mask
    Parameters
    ----------
    in_file : Path
        Image to be cropped
    mask : Path
        Mask to crop by
    out_file : Path
        Image to write results to
    """
    mask = nib.load(mask).get_fdata().astype(bool)
    orig_img = nib.load(in_file)
    orig_data = orig_img.get_fdata()
    orig_data[~mask] = np.nan
    new_img = nib.Nifti1Image(orig_data, orig_img.affine)
    nib.save(new_img, out_file)


def extract_brain(wholehead: Path, brain: Path, brain_mask: Path):
    cmd = f"fslmaths {wholehead} -mul {brain_mask} {brain}"
    os.system(cmd)


def epi_reg(in_file: Path, t1: Path, t1_brain: Path, out_file: Path):
    if not out_file.exists():
        cmd = f"epi_reg --epi={in_file} --t1={t1} --t1brain={t1_brain} --out={out_file}"
        os.system(cmd)
    return out_file.parent / f"{out_file.name.split('.')[0]}.mat"


def applyxfm_fsl(in_file, xfm, ref, out_file):
    ax = fsl.ApplyXFM()
    ax.inputs.in_file = in_file
    ax.inputs.in_matrix_file = xfm
    ax.inputs.reference = ref
    ax.inputs.out_file = out_file
    ax.run()


def coregister_tensors_longitudinal(
    registerations_dir: Path,
    fs_dir: Path,
    fs_brain: Path,
    mean_epi_to_t1w: Path,
    tensors_dict: dict,
):
    for ses, aff_name in zip(["ses-1", "ses-2"], ["pre2post", "post2pre"]):
        print("\t", ses)
        between_sessions = registerations_dir / f"mean_b0_{aff_name}_half.mat"
        aff_full = fs_dir / f"{ses}_epi2anatomical.mat"
        if not aff_full.exists():
            os.system(
                f"convert_xfm -omat {aff_full} -concat {mean_epi_to_t1w} {between_sessions}"
            )
        for key, path in tensors_dict.get(ses).items():
            param_nii = path.with_suffix(".nii.gz")
            if not param_nii.exists():
                os.system(f"mrconvert {path} {param_nii} -force")
            out_file = param_nii.parent.parent / "coreg_FS" / param_nii.name
            out_file.parent.mkdir(exist_ok=True)
            if not out_file.exists():
                applyxfm_fsl(param_nii, aff_full, fs_brain, out_file)


def coregister_tensors_single_session(
    fs_brain: Path,
    mean_epi_to_t1w: Path,
    tensors_dict: dict,
):
    for key, path in tensors_dict.get("ses-1").items():
        param_nii = path.with_suffix(".nii.gz")
        if not param_nii.exists():
            os.system(f"mrconvert {path} {param_nii} -force")
        out_file = param_nii.parent.parent / "coreg_FS" / param_nii.name
        out_file.parent.mkdir(exist_ok=True)
        if not out_file.exists():
            applyxfm_fsl(param_nii, mean_epi_to_t1w, fs_brain, out_file)
