import argparse
import tqdm
from pathlib import Path
from brainprint.feature_generation.subject_results import SubjectResults

parser = argparse.ArgumentParser(description="Generate features per subject.")
parser.add_argument("base_dir", type=str, nargs=1)
parser.add_argument("-out_dir", type=str, required=False)
parser.add_argument("-subject_id", type=str, nargs="?", required=False)
parser.add_argument(
    "-atlas", default="Brainnetome", type=str, nargs="?", required=False
)
args = vars(parser.parse_args())
base_dir, out_dir, subj_id, atlas_name = [
    args.get(key) for key in ["base_dir", "out_dir", "subject_id", "atlas"]
]
base_dir = Path(base_dir[0])

if not out_dir:
    out_dir = base_dir / "derivatives" / "data_processing" / atlas_name

out_dir.mkdir(exist_ok=True, parents=True)

if not subj_id:
    subj_ids = [
        s.name
        for s in sorted(base_dir.glob(f"{SubjectResults.BIDS_DIR_NAME}/sub-*"))
    ]
else:
    subj_ids = [subj_id]
for atlas in [atlas_name]:
    for subj_id in tqdm.tqdm(subj_ids):
        try:
            results = SubjectResults(base_dir, subj_id)
            flag = out_dir / f"{subj_id}_{results.FIRST_SESSION}.csv"
            if flag.exists():
                continue
            metrics = results.summarize_subject_metrics(atlas_name)
            for ses, df in metrics.items():
                df.to_csv(out_dir / f"{subj_id}_{ses}.csv")
        except:
            if results.diffusion_derivatives_path.exists():
                try:
                    results.coregister_tensors()
                except:
                    continue
            # continue
        # break
