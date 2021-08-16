from brainprint.feature_generation.subject_results import SubjectResults
from brainprint.utils.parcellations import parcellations
from nipype.interfaces.ants import ApplyTransforms

from pathlib import Path


def locate_files(subj_results: SubjectResults):
    struct_dir = subj_results.structural_derivatives_path
    diffusion_dir = subj_results.diffusion_derivatives_path
    fs_transform = [
        f
        for f in struct_dir.glob(
            "*_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
        )
    ][0]
    gm_prob = [
        f
        for f in struct_dir.glob("*_label-GM_probseg.nii.gz")
        if "MNI152" not in f.name
    ][0]
    native_parcellation = subj_results.get_subject_parcellation()
    register_atlas = True
    register_tensors = False
    if native_parcellation.exists():
        register_atlas = False
    if diffusion_dir.exists():
        register_tensors = True
    return (
        struct_dir,
        fs_transform,
        diffusion_dir,
        native_parcellation,
        gm_prob,
        register_atlas,
        register_tensors,
    )


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


def parcellation_registration(
    struct_dir: Path,
    fs_transform: Path,
    native_parcellation: Path,
    gm_prob: Path,
    atlas_name: str = "Brainnetome",
):
    standard_img = parcellations.get(atlas_name).get("atlas")
    native_parcellation_full = (
        native_parcellation.parent
        / native_parcellation.name.replace("_GM", "")
    )


if __name__ == "__main__":
    base_dir = Path("/media/groot/Yalla/media/MRI")
    subj_ids = [
        sub.name for sub in base_dir.glob("derivatives/fmriprep/sub-*")
    ]
    for subj_id in subj_ids:
        if subj_id.endswith("html"):
            continue
        res = SubjectResults(base_dir, subj_id)
        (
            struct_dir,
            fs_transform,
            diffusion_dir,
            native_parcellation,
            gm_prob,
            register_atlas,
            register_tensors,
        ) = locate_files(res)
