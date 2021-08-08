from pathlib import Path
from freesurfer_registers import longitudinal, single_session


def clean_workspace():
    workspace = Path(".")
    for fname in workspace.glob("*_flirt.mat"):
        fname.unlink()


if __name__ == "__main__":
    mother_dir = Path("/media/groot/Yalla/media/MRI")
    bids_dir = mother_dir / "NIfTI"
    dwi_derivatives = mother_dir / "derivatives" / "dwiprep"
    func_derivatives = mother_dir / "derivatives" / "fmriprep"
    atlas_file = (
        "/media/groot/Data/Parcellations/MNI/BN_Atlas_274_combined.nii.gz"
    )
    atlas_name = "Brainnetome"

    for subj in dwi_derivatives.glob("sub-*"):
        sessions = [
            s.name for s in func_derivatives.glob(f"{subj.name}/ses-*")
        ]
        if len(sessions) > 1:

            longitudinal.atlas_to_subject_space(
                func_derivatives, atlas_file, atlas_name, subj
            )
            longitudinal.coreg_to_freesurfer(func_derivatives, subj)
        elif len(sessions) == 1:
            session = sessions[0]
            print(subj, "---", session)
            single_session.atlas_to_subject_space(
                func_derivatives, atlas_file, atlas_name, subj, session
            )
            single_session.coreg_to_freesurfer(func_derivatives, subj, session)
        break
    clean_workspace()
