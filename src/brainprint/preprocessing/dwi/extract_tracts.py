import os
from pathlib import Path
from nipype.interfaces import fsl, ants


def get_available_parcellations(mother_dir: Path):
    parcellations = []
    for f in mother_dir.rglob("registrations/*_FS/*_native.nii.gz"):
        parcellations.append("_".join(f.name.split("_")[:-1]))
    return list(set(parcellations))


def gather_subject_files(
    mother_dir: Path,
    bids_dir: Path,
    subject: str,
    anat_method: str = "FS",
    atlas_name: str = "Brainnetome",
    ses: str = "ses-1",
) -> dict:
    """[summary]

    Parameters
    ----------
    derivatives_dir : Path
        [description]
    subject : str
        [description]

    Returns
    -------
    dict
        [description]
    """
    subj_dict = {}
    dwi_dir = mother_dir / "derivatives" / "dwiprep" / subject
    fmriprep_dir = mother_dir / "derivatives" / "fmriprep" / subject
    if anat_method == "FS":
        subj_dict["anat"] = (
            fmriprep_dir / "anat" / f"{subject}_desc-preproc_T1w.nii.gz"
        )
        subj_dict["anat_mask"] = (
            fmriprep_dir / "anat" / f"{subject}_desc-brain_mask.nii.gz"
        )
        subj_dict["t1w2epi"] = (
            dwi_dir
            / "registrations"
            / f"preprocessed_{anat_method}"
            / f"{ses}_anatomical2epi.mat"
        )
        subj_dict["anat_coreg"] = (
            fmriprep_dir
            / ses
            / "anat"
            / f"{subject}_{ses}_acq-corrected_from-orig_to-T1w_mode-image_xfm.txt"
        )
    subj_dict["T2w"] = (
        bids_dir
        / subject
        / ses
        / "anat"
        / f"{subject}_{ses}_acq-corrected_T2w.nii.gz"
    )
    subj_dict["dwi"] = dwi_dir / ses / "bias_corrected.mif"
    subj_dict["parcellation"] = (
        dwi_dir
        / "registrations"
        / f"preprocessed_{anat_method}"
        / f"{atlas_name}_native.nii.gz"
    )
    subj_dict["mean_bzero"] = (
        dwi_dir / "registrations" / "mean_b0" / f"mean_b0_{ses}.nii.gz"
    )
    flags = [d.exists() for d in subj_dict.values()]
    # for d in subj_dict.values():
    #     print(d)
    #     print(d.exists())
    if not all(flags):
        print(
            f"Subject {subject} does not have neccessary files required for tractography. Aborting..."
        )
        print(f"Evidence(s):")
        for i, key, fname in zip(flags, subj_dict.keys(), subj_dict.values()):
            if not i:
                print(key, ":", fname)
        flag = False
    else:
        subj_dict["out_dir"] = (
            mother_dir
            / "derivatives"
            / "dwiprep"
            / subject
            / ses
            / "tractography"
        )
        subj_dict["out_dir"].mkdir(exist_ok=True)
        flag = True

    return subj_dict, flag


def gen_5tt(subj_files: dict):
    anat, t2, anat_coreg, mask = [
        subj_files.get(key)
        for key in ["anat", "T2w", "anat_coreg", "anat_mask"]
    ]
    t2_coreg = anat.with_name(anat.name.replace("T1", "T2"))
    if not t2_coreg.exists():
        cmd = ants_at(t2, anat, anat_coreg, t2_coreg)
        print(cmd.cmdline)
        cmd.run()
    out_file = anat.with_name("5TT_nocoreg.mif")
    if not out_file.exists():
        cmd = (
            f"5ttgen fsl {anat} {out_file} -t2 {t2_coreg} -mask {mask} -force"
        )
        os.system(cmd)
    subj_files["5TT"] = out_file
    return subj_files


def mask_dwi(subj_files: dict):
    dwi = subj_files.get("dwi")
    out_file = subj_files.get("out_dir") / "dwi_mask.mif"
    if not out_file.exists():
        cmd = f"dwi2mask {dwi} {out_file} -force"
        os.system(cmd)
    subj_files["dwi_mask"] = out_file
    return subj_files


def dwi2response(subj_files: dict):
    dwi, mask, out_dir = [
        subj_files.get(key) for key in ["dwi", "dwi_mask", "out_dir"]
    ]
    out_wm, out_gm, out_csf = [
        out_dir / f"{tissue}_RF.txt" for tissue in ["WM", "GM", "CSF"]
    ]
    out_voxels = out_dir / "RF_voxels.mif"
    flags = [fname.exists() for fname in [out_wm, out_gm, out_csf, out_voxels]]
    if not all(flags):
        cmd = f"dwi2response dhollander {dwi} {out_wm} {out_gm} {out_csf} -voxels {out_voxels} -mask {mask} -force"
        os.system(cmd)
    subj_files["responses"] = {}
    for tissue, out_file in zip(
        ["WM", "GM", "CSF"], [out_wm, out_gm, out_csf]
    ):
        subj_files["responses"][tissue] = out_file
    return subj_files


def dwi2fod(subj_files: dict):
    dwi, mask, out_dir, responses = [
        subj_files.get(key)
        for key in ["dwi", "dwi_mask", "out_dir", "responses"]
    ]
    out_wm, out_gm, out_csf = [
        out_dir / f"{tissue}_FODs.mif" for tissue in ["WM", "GM", "CSF"]
    ]
    flags = [fname.exists() for fname in [out_wm, out_gm, out_csf]]
    if not all(flags):
        cmd = f"dwi2fod msmt_csd {dwi} {responses.get('WM')} {out_wm} {responses.get('GM')} {out_gm} {responses.get('CSF')} {out_csf} -mask {mask} -force"
        os.system(cmd)
    subj_files["FODs"] = {}
    for tissue, out_file in zip(
        ["WM", "GM", "CSF"], [out_wm, out_gm, out_csf]
    ):
        subj_files["FODs"][tissue] = out_file
    return subj_files


def normalise_intensity(subj_files: dict):
    out_dir, mask = [subj_files.get(key) for key in ["out_dir", "dwi_mask"]]
    wm_fod, gm_fod, csf_fod = [
        subj_files.get("FODs").get(tissue) for tissue in ["WM", "GM", "CSF"]
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
        subj_files["FODs"][f"{tissue}_norm"] = out_file
    return subj_files


def coregister_anat(subj_files: dict):
    anat, mean_b0, t1w2epi, five_tissue, out_dir = [
        subj_files.get(key)
        for key in ["anat", "mean_bzero", "t1w2epi", "5TT", "out_dir"]
    ]
    t1w2epi_mrtrix = out_dir / "struct2dwi_mrtrix.txt"
    if not t1w2epi_mrtrix.exists():
        cmd = f"transformconvert {t1w2epi} {anat} {mean_b0} flirt_import {t1w2epi_mrtrix} -force"
        os.system(cmd)
    for in_file, ftype in zip([anat, five_tissue], ["T1w_coreg", "5TT_coreg"]):
        out_file = out_dir / f"{ftype}.mif"
        if not out_file.exists():
            cmd = f"mrtransform {in_file} -linear {t1w2epi_mrtrix} {out_file} -force"
            os.system(cmd)
        subj_files[ftype] = out_file
    return subj_files


def perform_tractography(subj_files: dict, num_streamlines: str):
    dwi, mask, five_tissue, out_dir, fods = [
        subj_files.get(key)
        for key in ["dwi", "dwi_mask", "5TT_coreg", "out_dir", "FODs"]
    ]
    tractogram = out_dir / f"{num_streamlines}.tck"
    if not tractogram.exists():
        cmd = f"tckgen {fods.get('WM_norm')} {tractogram} -algorithm SD_Stream -act {five_tissue} -minlength 30 -maxlength 500 -crop_at_gmwmi -seed_dynamic {fods.get('WM')} -select {num_streamlines} -force"
        os.system(cmd)
    subj_files["tractogram"] = tractogram
    return subj_files


def sift_cleanup(subj_files: dict, num_streamlines: str):
    tractogram, mask, five_tissue, out_dir, fods = [
        subj_files.get(key)
        for key in ["tractogram", "dwi_mask", "5TT_coreg", "out_dir", "FODs"]
    ]
    sift_tractogram = out_dir / f"SIFT_{num_streamlines}.tck"
    if not sift_tractogram.exists():
        cmd = f"tcksift {tractogram} {fods.get('WM_norm')} {sift_tractogram} -act {five_tissue} -term_number {num_streamlines} -nthreads 16 -force"
        os.system(cmd)
    subj_files["sift_tractogram"] = sift_tractogram
    return subj_files


def ants_at(
    in_file, ref, xfm, out_file, nn: bool = False, invert_xfm: bool = False
):
    at = ants.ApplyTransforms()
    at.inputs.input_image = in_file
    at.inputs.reference_image = ref
    at.inputs.transforms = xfm
    at.inputs.output_image = str(out_file)
    if nn:
        at.inputs.interpolation = "NearestNeighbor"
    if invert_xfm:
        at.inputs.invert_transform_flags = True
    return at


def apply_xfm(in_file: Path, aff: Path, out_file: Path, ref: Path, nn=True):
    applyxfm = fsl.preprocess.ApplyXFM()
    applyxfm.inputs.in_file = in_file
    applyxfm.inputs.in_matrix_file = aff
    applyxfm.inputs.out_file = out_file
    applyxfm.inputs.reference = ref
    applyxfm.inputs.apply_xfm = True
    if nn:
        applyxfm.inputs.interp = "nearestneighbour"
    return applyxfm


def convert_xfm(subj_files: dict):
    epi2anat = subj_files.get("epi2t1w")
    anat2epi = epi2anat.with_name("anatomical2epi.mat")
    if not anat2epi.exists():
        cmd = f"convert_xfm -omat {anat2epi} -inverse {epi2anat}"
        os.system(cmd)
    subj_files["anat2epi"] = anat2epi
    return subj_files


def atlas_to_dwi(subj_files: dict, atlas_name: str = "Brainnetome"):
    atlas, anat2epi, mean_bzero, out_dir = [
        subj_files.get(key)
        for key in ["parcellation", "t1w2epi", "mean_bzero", "out_dir"]
    ]
    out_file = out_dir / f"{atlas_name}_in_DWI_space.nii.gz"
    if not out_file.exists():
        cmd = apply_xfm(atlas, anat2epi, out_file, mean_bzero)
        cmd.run()
    subj_files["dwi_parcellation"] = out_file
    out_file = out_dir / f"{atlas_name}_nodes.mif"
    if not out_file.exists():
        cmd = f"mrconvert {subj_files.get('dwi_parcellation')} {out_file} -datatype int64 -force"
        os.system(cmd)
    subj_files["dwi_nodes"] = out_file
    return subj_files


def generate_connectome(
    subj_files: dict,
    num_streamlines: int,
    atlas_name: str = "Brainnetome",
    scaled: str = False,
):
    atlas, sift_tracts, out_dir = [
        subj_files.get(key)
        for key in ["dwi_nodes", "sift_tractogram", "out_dir"]
    ]
    if scaled == "none":
        out_conn = out_dir / f"{atlas_name}_{num_streamlines}_connectome.csv"
        out_assignments = (
            out_dir / f"{atlas_name}_{num_streamlines}_assignments.txt"
        )
        if (not out_conn.exists()) or (not out_assignments.exists()):
            cmd = f"tck2connectome {sift_tracts} {atlas} {out_conn} -out_assignments {out_assignments} -symmetric -force"
            os.system(cmd)
    elif scaled == "vol":
        out_conn = (
            out_dir / f"{atlas_name}_{num_streamlines}_connectome_scaled.csv"
        )
        out_assignments = (
            out_dir / f"{atlas_name}_{num_streamlines}_assignments_scaled.txt"
        )
        if (not out_conn.exists()) or (not out_assignments.exists()):
            cmd = f"tck2connectome {sift_tracts} {atlas} {out_conn} -out_assignments {out_assignments} -symmetric -scale_invnodevol -force"
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
            cmd = f"tck2connectome {sift_tracts} {atlas} {out_conn} -out_assignments {out_assignments} -symmetric -scale_length -stat_edge mean -force"
            os.system(cmd)


if __name__ == "__main__":
    mother_dir = Path("/media/groot/Yalla/media/MRI")
    bids_dir = mother_dir / "NIfTI"
    derivatives_dir = mother_dir / "derivatives" / "dwiprep"
    atlas_name = "Brainnetome"
    total_streamlines = "5M"
    clean_streamlines = "1M"
    # for atlas_name in get_available_parcellations(derivatives_dir):
    #     print(atlas_name)
    for subj in derivatives_dir.glob("sub-*"):
        for ses in subj.glob("ses*"):
            print(subj.name)
            subj_files, to_process = gather_subject_files(
                mother_dir,
                bids_dir,
                subj.name,
                anat_method="FS",
                atlas_name=atlas_name,
                ses=ses.name,
            )
            to_process = True
            print(to_process)
            if not to_process:
                continue
            if "348" not in subj.name:
                subj_files = gen_5tt(subj_files)
            subj_files = mask_dwi(subj_files)
            subj_files = dwi2response(subj_files)
            subj_files = dwi2fod(subj_files)
            subj_files = normalise_intensity(subj_files)
            subj_files = coregister_anat(subj_files)
            subj_files = perform_tractography(subj_files, total_streamlines)
            subj_files = sift_cleanup(subj_files, clean_streamlines)
            # subj_files = convert_xfm(subj_files)
            subj_files = atlas_to_dwi(subj_files, atlas_name)
            for scale in ["none", "vol", "length"]:
                generate_connectome(
                    subj_files,
                    num_streamlines=clean_streamlines,
                    atlas_name=atlas_name,
                    scaled=scale,
                )
        #     break
        # break
