from pathlib import Path
from dwiprep.preprocessing.preprocess import PreprocessPipeline


input_dir = Path("/media/groot/Yalla/media/MRI/NIfTI")
output_dir = Path("/media/groot/Yalla/media/MRI/derivatives/dwiprep")
atlas = {
    "path": Path(
        "/media/groot/Data/Parcellations/MNI/BN_Atlas_274_combined_1mm.nii.gz",
    ),
    "name": "Brainnetome",
}
bad_subjects = []
for subj in sorted(input_dir.glob("sub-*")):
    print("Working on ", subj.name)
    try:
        try:
            anats = sorted(
                [f for f in subj.glob("ses*/anat/*acq-corrected*.nii*")]
            )[0]
        except IndexError:
            anats = sorted([f for f in subj.glob("ses*/anat/*T1w*.nii*")])[0]
        subj_output = output_dir / subj.name
        try:
            aps = sorted([f for f in subj.glob("ses*/dwi/*.nii*")])[0]
            pas = sorted(
                [
                    f
                    for f in subj.glob("ses*/fmap/*.nii*")
                    if "func" not in f.name
                ]
            )[0]
            input_dict = {"anatomical": anats, "ap": aps, "pa": pas}
            print(input_dict)
            t = PreprocessPipeline(input_dict, subj_output)
            t.run_corrections()
        except:
            continue
    except:
        continue
