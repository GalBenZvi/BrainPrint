import os
import numpy as np
import nibabel as nib
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces import fsl
from nipype.interfaces import spm
from pathlib import Path


def at_ants(
    in_file: Path,
    ref: Path,
    xfm: Path,
    outfile: Path,
    nn: bool,
    invert_xfm: bool = False,
    run: bool = True,
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
    if run:
        at.run()
    else:
        return at


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
    out_mat = out_file.parent / f"{out_file.name.split('.')[0]}.mat"
    if not out_file.exists():
        out_mat.parent.mkdir(exist_ok=True)
        cmd = f"epi_reg --epi={in_file} --t1={t1} --t1brain={t1_brain} --out={out_file}"
        os.system(cmd)
        if out_mat.exists():
            return out_mat
        cmd = f"flirt -in {t1} -ref {in_file} -out trial2.nii.gz -cost mutualinfo -dof 6 -out {out_file} -omat {out_mat}"
        os.system(cmd)
    return out_mat


def apply_xfm(in_file: Path, aff: Path, ref: Path, out_file: Path, nn=False):
    applyxfm = fsl.preprocess.ApplyXFM()
    applyxfm.inputs.in_file = in_file
    applyxfm.inputs.in_matrix_file = aff
    applyxfm.inputs.out_file = out_file
    applyxfm.inputs.reference = ref
    applyxfm.inputs.apply_xfm = True
    if nn:
        applyxfm.inputs.interp = "nearestneighbour"
    return applyxfm


def coregister_tensors_longitudinal(
    registerations_dir: Path,
    fs_dir: Path,
    fs_brain: Path,
    mean_epi_to_t1w: Path,
    tensors_dict: dict,
):
    aff_dict = {}
    for ses, aff_name in zip(["ses-1", "ses-2"], ["pre2post", "post2pre"]):
        print("\t", ses)
        between_sessions = registerations_dir / f"mean_b0_{aff_name}_half.mat"
        aff_full = fs_dir / f"{ses}_epi2anatomical.mat"
        if not aff_full.exists():
            os.system(
                f"convert_xfm -omat {aff_full} -concat {mean_epi_to_t1w} {between_sessions}"
            )
        inv_aff_full = aff_full.with_name(f"{ses}_anatomical2epi.mat")
        if not inv_aff_full.exists():
            os.system(f"convert_xfm -omat {inv_aff_full} -inverse {aff_full}")
        aff_dict[ses] = inv_aff_full
        for key, path in tensors_dict.get(ses).items():
            param_nii = path.with_suffix(".nii.gz")
            if not param_nii.exists():
                os.system(f"mrconvert {path} {param_nii} -force")
            out_file = param_nii.parent.parent / "coreg_FS" / param_nii.name
            out_file.parent.mkdir(exist_ok=True)
            if not out_file.exists():
                apply_xfm(param_nii, aff_full, fs_brain, out_file)
    return aff_dict


def coregister_tensors_single_session(
    fs_brain: Path,
    mean_epi_to_t1w: Path,
    tensors_dict: dict,
):
    inv_aff_full = mean_epi_to_t1w.with_name(f"ses-1_anatomical2epi.mat")
    if not inv_aff_full.exists():
        os.system(
            f"convert_xfm -omat {inv_aff_full} -inverse {mean_epi_to_t1w}"
        )
    aff_dict = {"ses-1": inv_aff_full}
    for key, path in tensors_dict.get("ses-1").items():
        param_nii = path.with_suffix(".nii.gz")
        if not param_nii.exists():
            os.system(f"mrconvert {path} {param_nii} -force")
        out_file = param_nii.parent.parent / "coreg_FS" / param_nii.name
        out_file.parent.mkdir(exist_ok=True)
        if not out_file.exists():
            apply_xfm(param_nii, mean_epi_to_t1w, fs_brain, out_file)
    return aff_dict


def gen_5tt(anat, t2, anat_coreg, mask):
    # anat, t2, anat_coreg, mask = [
    #     subj_files.get(key)
    #     for key in ["anat", "T2w", "anat_coreg", "anat_mask"]
    # ]
    t2_coreg = anat.with_name(anat.name.replace("T1", "T2"))
    if not t2_coreg.exists():
        cmd = at_ants(
            t2,
            anat,
            anat_coreg,
            t2_coreg,
            nn=False,
            invert_xfm=False,
            run=False,
        )
        print(cmd.cmdline)
        cmd.run()
    out_file = anat.with_name("5TT_nocoreg.mif")
    if not out_file.exists():
        cmd = (
            f"5ttgen fsl {anat} {out_file} -t2 {t2_coreg} -mask {mask} -force"
        )
        print(cmd)
        os.system(cmd)
        if not out_file.exists():
            cmd = f"5ttgen fsl {anat} {out_file} -mask {mask} -force"
            print(cmd)
            os.system(cmd)
    return out_file


def mask_dwi(dwi: Path, out_dir: Path):
    out_file = out_dir / "dwi_mask.mif"
    if not out_file.exists():
        cmd = f"dwi2mask {dwi} {out_file} -force"
        os.system(cmd)
    return out_file


def dwi2response(dwi: Path, mask: Path, out_dir: Path):
    out_wm, out_gm, out_csf = [
        out_dir / f"{tissue}_RF.txt" for tissue in ["WM", "GM", "CSF"]
    ]
    out_voxels = out_dir / "RF_voxels.mif"
    flags = [fname.exists() for fname in [out_wm, out_gm, out_csf, out_voxels]]
    if not all(flags):
        cmd = f"dwi2response dhollander {dwi} {out_wm} {out_gm} {out_csf} -voxels {out_voxels} -mask {mask} -force"
        os.system(cmd)
    responses = {}
    for tissue, out_file in zip(
        ["WM", "GM", "CSF"], [out_wm, out_gm, out_csf]
    ):
        responses[tissue] = out_file
    return responses


def dwi2fod(dwi: Path, mask: Path, out_dir: Path, responses: dict):
    out_wm, out_gm, out_csf = [
        out_dir / f"{tissue}_FODs.mif" for tissue in ["WM", "GM", "CSF"]
    ]
    flags = [fname.exists() for fname in [out_wm, out_gm, out_csf]]
    if not all(flags):
        cmd = f"dwi2fod msmt_csd {dwi} {responses.get('WM')} {out_wm} {responses.get('GM')} {out_gm} {responses.get('CSF')} {out_csf} -mask {mask} -force"
        os.system(cmd)
    fods = {}
    for tissue, out_file in zip(
        ["WM", "GM", "CSF"], [out_wm, out_gm, out_csf]
    ):
        fods[tissue] = out_file
    return fods


def normalise_intensity(out_dir: Path, mask: Path, fods: dict):
    wm_fod, gm_fod, csf_fod = [
        fods.get(tissue) for tissue in ["WM", "GM", "CSF"]
    ]
    out_wm, out_gm, out_csf = [
        out_dir / f"{tissue}_FODs_norm.mif" for tissue in ["WM", "GM", "CSF"]
    ]
    flags = [fname.exists() for fname in [out_wm, out_gm, out_csf]]
    if not all(flags):
        cmd = f"mtnormalise {wm_fod} {out_wm} {gm_fod} {out_gm} {csf_fod} {out_csf} -mask {mask} -force"
        os.system(cmd)
    for tissue, out_file in zip(
        ["WM", "GM", "CSF"], [out_wm, out_gm, out_csf]
    ):
        fods[f"{tissue}_norm"] = out_file
    return fods


def coregister_anat(
    anat: Path, mean_b0: Path, t1w2epi: Path, five_tissue: Path, out_dir: Path
):
    t1w2epi_mrtrix = out_dir / "struct2dwi_mrtrix.txt"
    if not t1w2epi_mrtrix.exists():
        cmd = f"transformconvert {t1w2epi} {anat} {mean_b0} flirt_import {t1w2epi_mrtrix} -force"
        os.system(cmd)
    coreg_anat = {}
    for in_file, ftype in zip([anat, five_tissue], ["T1w_coreg", "5TT_coreg"]):
        out_file = out_dir / f"{ftype}.mif"
        if not out_file.exists():
            cmd = f"mrtransform {in_file} -linear {t1w2epi_mrtrix} {out_file} -force"
            os.system(cmd)
        coreg_anat[ftype] = out_file
    return coreg_anat


def perform_tractography(
    coreg_anat: dict, out_dir: Path, fods: dict, num_streamlines: str = "5M"
):
    five_tissue = coreg_anat.get("5TT_coreg")
    tractogram = out_dir / f"{num_streamlines}.tck"
    if not tractogram.exists():
        cmd = f"tckgen {fods.get('WM_norm')} {tractogram} -algorithm SD_Stream -act {five_tissue} -minlength 30 -maxlength 500 -crop_at_gmwmi -seed_dynamic {fods.get('WM')} -select {num_streamlines} -force"
        os.system(cmd)
    return tractogram


def sift_cleanup(
    tractogram: Path,
    anat_coreg: dict,
    out_dir: Path,
    fods: dict,
    num_streamlines: str = "1M",
):
    five_tissue = anat_coreg.get("5TT_coreg")
    sift_tractogram = out_dir / f"SIFT_{num_streamlines}.tck"
    if not sift_tractogram.exists():
        cmd = f"tcksift {tractogram} {fods.get('WM_norm')} {sift_tractogram} -act {five_tissue} -term_number {num_streamlines} -nthreads 16 -force"
        os.system(cmd)
    return sift_tractogram


def atlas_to_dwi(
    atlas: Path,
    anat2epi: Path,
    mean_bzero: Path,
    out_dir: Path,
    atlas_name: str = "Brainnetome",
):
    out_file = out_dir / f"{atlas_name}_in_DWI_space.nii.gz"
    if not out_file.exists():
        cmd = apply_xfm(atlas, anat2epi, mean_bzero, out_file, nn=True)
        cmd.run()
    dwi_parcellation = out_file
    out_file = out_dir / f"{atlas_name}_nodes.mif"
    if not out_file.exists():
        cmd = (
            f"mrconvert {dwi_parcellation} {out_file} -datatype uint64 -force"
        )
        os.system(cmd)
    dwi_nodes = out_file
    return dwi_parcellation, dwi_nodes


def generate_connectome(
    atlas: Path,  # dwi_nodes
    sift_tracts: Path,
    out_dir: Path,
    num_streamlines: int,
    atlas_name: str = "Brainnetome",
    scaled: str = "none",
):
    if scaled == "none":
        out_conn = out_dir / f"{atlas_name}_{num_streamlines}_connectome.csv"
        out_assignments = (
            out_dir / f"{atlas_name}_{num_streamlines}_assignments.txt"
        )
        if (not out_conn.exists()) or (not out_assignments.exists()):
            cmd = f"tck2connectome {sift_tracts} {atlas} {out_conn} -out_assignments {out_assignments} -symmetric -nthreads 10 -force"
            os.system(cmd)
    elif scaled == "vol":
        out_conn = (
            out_dir / f"{atlas_name}_{num_streamlines}_connectome_scaled.csv"
        )
        out_assignments = (
            out_dir / f"{atlas_name}_{num_streamlines}_assignments_scaled.txt"
        )
        if (not out_conn.exists()) or (not out_assignments.exists()):
            cmd = f"tck2connectome {sift_tracts} {atlas} {out_conn} -out_assignments {out_assignments} -symmetric -scale_invnodevol -nthreads 10 -force"
            os.system(cmd)
    elif scaled == "length":
        out_conn = (
            out_dir
            / f"{atlas_name}_{num_streamlines}_connectome_streamlines_lengths.csv"
        )
        out_assignments = (
            out_dir
            / f"{atlas_name}_{num_streamlines}_assignments_streamlines_lengths.txt"
        )
        if (not out_conn.exists()) or (not out_assignments.exists()):
            cmd = f"tck2connectome {sift_tracts} {atlas} {out_conn} -out_assignments {out_assignments} -symmetric -scale_length -stat_edge mean -nthreads 10 -force"
            os.system(cmd)
    return out_conn


def tractography_pipeline(
    anat_preproc: Path,
    anat_mask: Path,
    t1w2epi: Path,
    anat_coreg: Path,
    t2w: Path,
    dwi: Path,
    native_parcellation: Path,
    mean_bzero: Path,
    out_dir: Path,
    streamlines_init: str = "5M",
    streamlines_post_cleanup: str = "1M",
    atlas_name: str = "Brainnetome",
):
    out_dir.mkdir(exist_ok=True)
    five_tissue_no_coreg = gen_5tt(anat_preproc, t2w, anat_coreg, anat_mask)
    dwi_mask = mask_dwi(dwi, out_dir)
    responses = dwi2response(dwi, dwi_mask, out_dir)
    fods = dwi2fod(dwi, dwi_mask, out_dir, responses)
    fods = normalise_intensity(out_dir, dwi_mask, fods)
    anat2dwi = coregister_anat(
        anat_preproc, mean_bzero, t1w2epi, five_tissue_no_coreg, out_dir
    )
    tractogram = perform_tractography(
        anat2dwi, out_dir, fods, streamlines_init
    )
    sift_tracts = sift_cleanup(
        tractogram, anat2dwi, out_dir, fods, streamlines_post_cleanup
    )
    dwi_parcellation, dwi_nodes = atlas_to_dwi(
        native_parcellation, t1w2epi, mean_bzero, out_dir, atlas_name
    )
    connectomes = {}
    for scale, key in zip([False, True], ["Unscaled", "Scaled"]):
        conn_file = generate_connectome(
            dwi_nodes,
            sift_tracts,
            out_dir,
            streamlines_post_cleanup,
            atlas_name,
            scale,
        )
        connectomes[key] = conn_file
    return connectomes
