from logging import log
from pathlib import Path
from re import L
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces import fsl
import os
import nibabel as nib
import numpy as np


def at_ants(in_file, ref, xfm, outfile, nn: bool, invert_xfm: bool = False):
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


def extract_brain(wholehead: Path, brain: Path, brain_mask: Path):
    cmd = f"fslmaths {wholehead} -mul {brain_mask} {brain}"
    os.system(cmd)


def epi_reg(in_file: Path, t1: Path, t1_brain: Path, out_file: Path):
    cmd = f"epi_reg --epi={in_file} --t1={t1} --t1brain={t1_brain} --out={out_file}"
    print(cmd)
    os.system(cmd)


def applyxfm_fsl(in_file, xfm, ref, out_file):
    ax = fsl.ApplyXFM()
    ax.inputs.in_file = in_file
    ax.inputs.in_matrix_file = xfm
    ax.inputs.reference = ref
    ax.inputs.out_file = out_file
    ax.run()


def crop_to_gm(native_parcels: Path, gm_probseg: Path):
    cropped_parcels = (
        native_parcels.parent
        / f"{native_parcels.name.split('.')[0]}_GM.nii.gz"
    )
    if not cropped_parcels.exists():
        gm_mask = nib.load(gm_probseg).get_fdata().astype(bool)
        orig_img = nib.load(native_parcels)
        gm_parcels = orig_img.get_fdata()
        gm_parcels[~gm_mask] = np.nan
        gm_img = nib.Nifti1Image(gm_parcels, orig_img.affine)
        nib.save(gm_img, cropped_parcels)
    return cropped_parcels


def atlas_to_subject_space(
    func_derivatives: Path,
    atlas_file: Path,
    atlas_name: str,
    subj: Path,
    ses: str,
):
    fs_transform = (
        func_derivatives
        / subj.name
        / ses
        / "anat"
        / f"{subj.name}_{ses}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
    )
    gm_prob = (
        func_derivatives
        / subj.name
        / ses
        / "anat"
        / f"{subj.name}_{ses}_label-GM_probseg.nii.gz"
    )
    ref = fs_transform.with_name(f"{subj.name}_{ses}_desc-preproc_T1w.nii.gz")
    out_file = fs_transform.with_name(f"{atlas_name}_native.nii.gz")
    out_file.parent.mkdir(exist_ok=True, parents=True)
    if out_file.exists():
        return
    # try:
    at_ants(atlas_file, ref, fs_transform, out_file, nn=True)
    crop_to_gm(out_file, gm_prob)


def coreg_to_freesurfer(func_derivatives: Path, subj: Path, ses: str):
    subj_id = subj.name
    fs_ref = (
        func_derivatives
        / subj_id
        / ses
        / "anat"
        / f"{subj_id}_{ses}_desc-preproc_T1w.nii.gz"
    )
    fs_mask = (
        func_derivatives
        / subj_id
        / ses
        / "anat"
        / f"{subj_id}_{ses}_desc-brain_mask.nii.gz"
    )
    fs_brain = fs_mask.with_name(fs_mask.name.replace("_mask", ""))
    if not fs_brain.exists():
        ### Extract brain ###
        extract_brain(fs_ref, fs_brain, fs_mask)
    epi_b0 = subj / "registrations" / "mean_b0" / "mean_b0_ses-1.nii.gz"
    out_file = (
        subj
        / "registrations"
        / "preprocessed_FS"
        / f"mean_epi2anatomical.nii.gz"
    )
    if not out_file.exists():
        epi_reg(epi_b0, fs_ref, fs_brain, out_file)
    aff_2 = out_file.parent / f"{out_file.name.split('.')[0]}.mat"
    aff_full = aff_2
    for param in subj.glob(f"ses-1/tensors*/native/*.mif"):
        print(param)
        param_nii = param.with_suffix(".nii.gz")
        if not param_nii.exists():
            os.system(f"mrconvert {param} {param_nii} -force")
        out_file = param_nii.parent.parent / "coreg_FS" / param_nii.name
        out_file.parent.mkdir(exist_ok=True)
        print("\t\t", param.name)
        if not out_file.exists():
            applyxfm_fsl(param_nii, aff_full, fs_brain, out_file)
            param_nii.unlink()
