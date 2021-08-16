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
    subjects = [s.name for s in dwi_derivatives.glob("sub-*")]
    subjects += [s.name for s in func_derivatives.glob("sub-*")]
    subjects = sorted(set(subjects))
    for subj in sorted(dwi_derivatives.glob("sub-*")):
        print(subj)
        sessions = [
            s.name for s in func_derivatives.glob(f"{subj.name}/ses-*")
        ]
        try:
            if len(sessions) > 1:
                print("Longitudinal registerations")
                longitudinal.atlas_to_subject_space(
                    func_derivatives, atlas_file, atlas_name, subj
                )
                longitudinal.coreg_to_freesurfer(func_derivatives, subj)
            elif len(sessions) == 1:
                session = sessions[0]
                print("Single-session registerations")
                single_session.atlas_to_subject_space(
                    func_derivatives, atlas_file, atlas_name, subj, session
                )
                single_session.coreg_to_freesurfer(
                    func_derivatives, subj, session
                )
        except:
            continue
    clean_workspace()
