import pandas as pd
from pathlib import Path
from brainprint.utils.parcellations import parcellations

base_dir = Path("/media/groot/Yalla/media/MRI")
data_dir = base_dir / "derivatives" / "data_processing"


def get_data(
    parameter: str, atlas_name: str = None, data_dir: Path = data_dir
):
    atlas_name = atlas_name or "Brainnetome"
    atlas_labels = pd.read_csv(
        parcellations.get(atlas_name).get("labels"), index_col=0
    )
    subjects = sorted(
        set(
            [
                int(s.stem.split("_")[0].split("-")[1])
                for s in data_dir.glob(f"{atlas_name}/*.csv")
            ]
        )
    )
    sessions = sorted(
        set(
            [
                int(s.stem.split("_")[1].split("-")[1])
                for s in data_dir.glob(f"{atlas_name}/*.csv")
            ]
        )
    )
    multi_index = pd.MultiIndex.from_product(
        [subjects, sessions], names=["Subject", "Session"]
    )
    data = pd.DataFrame(index=multi_index, columns=atlas_labels.index)
    for subj in subjects:
        for ses in sessions:
            fname = data_dir / atlas_name / f"sub-{subj}_ses-{ses}.csv"
            if fname.exists():
                addition = pd.read_csv(fname, index_col=0)
                data.loc[(subj, ses)] = addition.loc[
                    atlas_labels.index, parameter
                ]
    return data
