from pathlib import Path
import os

# fmriprep_cmd = "singularity run --cleanenv /my_images/fmriprep-latest.simg {bids_dir} {out_dir} participant --participant-label {subj_id} --output-spaces anat MNI152NLin2009cAsym --fs-license-file /media/groot/Yalla/misc/freesurfer/license.txt --anat-only"
# fmriprep_cmd = "singularity run --cleanenv -B /media/groot/Yalla/:/work /my_images/fmriprep-latest.simg /work/Aging/{bids_dir} /work/Aging/{out_dir} participant --participant-label {subj_id} --output-spaces anat --nthreads 16 --omp-nthreads 16 --mem_mb 10000 --use-aroma --fs-license-file /work/misc/freesurfer/license.txt"
fmriprep_cmd = "singularity run --cleanenv -B {bids_dir.parent} /my_images/fmriprep-latest.simg {bids_dir} {out_dir} participant -w {bids_dir.parent}/work --participant-label {subj_id} --output-spaces anat MNI152NLin2009cAsym --use-aroma --fs-license-file {bids_dir.parent}/license.txt"
# recon_all_cmd = "recon-all -subjid {subj_id} -i {anat} -sd {out_dir} -all"
if __name__ == "__main__":
    input_dir = Path("/media/groot/Yalla/media/MRI")
    bids_dir = input_dir / "NIfTI"
    out_dir = input_dir / "derivatives"
    out_dir = input_dir / f"derivatives" / "fmriprep"
    out_dir.mkdir(exist_ok=True, parents=True)
    # out_fname = out_dir.parent.name + "/" + out_dir.name
    for subj_dir in sorted(bids_dir.glob("sub-*"), reverse=True):
        subj_id = subj_dir.name.split("-")[-1]
        cmd = fmriprep_cmd.format(
            bids_dir=bids_dir,
            out_dir=out_dir.parent,
            subj_id=subj_id,
        )
        # print(cmd)
        flag = out_dir / f"{subj_dir.name}.html"
        if flag.exists():
            continue
        # if Path(subj_dir / "ses-1" / "func").exists():
        #     cmd = cmd.replace("--anat-only ", "")
        if not Path(flag.parent / flag.stem).exists():
            print(cmd)
            os.system(cmd)
