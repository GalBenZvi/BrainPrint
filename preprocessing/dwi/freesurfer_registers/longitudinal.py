from logging import log
from pathlib import Path
from re import L
import nipype
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces import fsl
import os


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


def atlas_to_subject_space(
    func_derivatives: Path,
    atlas_file: Path,
    atlas_name: str,
    subj: Path,
):
    fs_transform = (
        func_derivatives
        / subj.name
        / "anat"
        / f"{subj.name}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
    )
    ref = fs_transform.with_name(f"{subj.name}_desc-preproc_T1w.nii.gz")
    out_file = fs_transform.with_name(f"{atlas_name}_native.nii.gz")
    # out_file.parent.mkdir(exist_ok=True, parents=True)
    if out_file.exists():
        return
    # try:
    at_ants(atlas_file, ref, fs_transform, out_file, nn=True)


def coreg_to_freesurfer(func_derivatives: Path, subj: Path):
    subj_id = subj.name
    fs_ref = (
        func_derivatives
        / subj_id
        / "anat"
        / f"{subj_id}_desc-preproc_T1w.nii.gz"
    )
    fs_mask = (
        func_derivatives
        / subj_id
        / "anat"
        / f"{subj_id}_desc-brain_mask.nii.gz"
    )
    try:
        fs_brain = fs_mask.with_name(fs_mask.name.replace("_mask", ""))
        if not fs_brain.exists():
            ### Extract brain ###
            extract_brain(fs_ref, fs_brain, fs_mask)
        epi_b0 = (
            subj
            / "registrations"
            / "mean_b0"
            / "mean_coregistered_mean_b0.nii.gz"
        )
        out_file = (
            subj
            / "registrations"
            / "preprocessed_FS"
            / f"mean_epi2anatomical.nii.gz"
        )
        if not out_file.exists():
            epi_reg(epi_b0, fs_ref, fs_brain, out_file)
        aff_2 = out_file.parent / f"{out_file.name.split('.')[0]}.mat"
        for ses, aff_name in zip(["ses-1", "ses-2"], ["pre2post", "post2pre"]):
            print("\t", ses)
            aff_1 = (
                subj
                / "registrations"
                / "mean_b0"
                / f"mean_b0_{aff_name}_half.mat"
            )
            aff_full = (
                subj
                / "registrations"
                / "preprocessed_FS"
                / f"{ses}_epi2anatomical.mat"
            )
            if not aff_full.exists():
                os.system(
                    f"convert_xfm -omat {aff_full} -concat {aff_2} {aff_1}"
                )
            inv_aff_full = aff_full.with_name(f"{ses}_anatomical2epi.mat")
            if not inv_aff_full.exists():
                os.system(
                    f"convert_xfm -omat {inv_aff_full} -inverse {aff_full}"
                )
            for param in subj.glob(f"{ses}/tensors*/native/*.mif"):
                param_nii = param.with_suffix(".nii.gz")
                if not param_nii.exists():
                    os.system(f"mrconvert {param} {param_nii} -force")
                out_file = (
                    param_nii.parent.parent / "coreg_FS" / param_nii.name
                )
                out_file.parent.mkdir(exist_ok=True)
                print("\t\t", param.name)
                if not out_file.exists():
                    applyxfm_fsl(param_nii, aff_full, fs_brain, out_file)
                    param_nii.unlink()
    except:
        return
