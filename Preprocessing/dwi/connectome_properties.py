import bct
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib


def get_available_parcellations(mother_dir: Path):
    parcellations = []
    for f in mother_dir.rglob("registrations/*_FS/*_native.nii.gz"):
        parcellations.append("_".join(f.name.split("_")[:-1]))
    return list(set(parcellations))


def generate_parcellations_dict(
    parcellations_dir: Path, avalilable_parcellations: list
):
    parcellations_dict = {}
    for parcellation in sorted(avalilable_parcellations):
        parcellations_dict[parcellation] = {}
        if "MMP" in parcellation:
            atlas_parcels = pd.read_csv(
                parcellations_dir / "MMP" / "MMP_parcels.csv", index_col=0
            )
            atlas_img = nib.load(
                parcellations_dir / "MMP" / "MMP_in_MNI_corr.nii.gz"
            )
        elif "Brainnetome" in parcellation:
            atlas_parcels = pd.read_csv(
                parcellations_dir / "BNA_with_cerebellum.csv", index_col=0
            )
            atlas_parcels.index = atlas_parcels.Label
            atlas_img = nib.load(
                parcellations_dir / "BN_Atlas_274_combined.nii.gz"
            )
        elif "Schaefer" in parcellation:
            n_networks, n_parcels = parcellation.split("_")[1:]
            atlas_parcels = pd.DataFrame(
                pd.read_csv(
                    parcellations_dir
                    / f"Schaefer2018_{n_parcels}_{n_networks}_order.txt",
                    index_col=0,
                    sep="\t",
                    header=None,
                ).iloc[:, 0]
            )
            atlas_parcels.columns = ["ROIname"]
            atlas_parcels["Label"] = atlas_parcels.index
            atlas_img = nib.load(
                parcellations_dir
                / f"Schaefer2018_{n_parcels}_{n_networks}_order_FSLMNI152_1mm.nii.gz"
            )
        parcellations_dict[parcellation]["atlas_parcels"] = atlas_parcels
        parcellations_dict[parcellation]["atlas_img"] = atlas_img
    return parcellations_dict


def plot_property(
    img: nib.Nifti1Image, data: np.ndarray, df: pd.DataFrame, col: str
):
    new_data = np.zeros_like(data)
    for i in df.index:
        cur_mask = data == df.loc[i, "Label"]
        new_data[cur_mask] = df.loc[i, col]
    # new_data[new_data == 0] = np.NaN
    new_img = nib.Nifti1Image(new_data, img.affine)
    return new_img


def edit_atlas_label(atlas_labels: pd.DataFrame) -> pd.DataFrame:
    atlas_labels["Hemi"] = ""
    for i in atlas_labels.index:
        atlas_labels.loc[i, "Hemi"] = atlas_labels.loc[i, "ROIname"].split(
            "_"
        )[-1]
        try:
            t = int(atlas_labels.loc[i, "Hemi"])
            atlas_labels.loc[i, "Hemi"] = None
        except:
            continue
    atlas_labels["community"] = [
        f"{atlas_labels.loc[i,'Gyrus'][:-1]}{atlas_labels.loc[i,'Hemi']}"
        for i in atlas_labels.index
    ]
    atlas_labels["CommunityCode"] = pd.Categorical(atlas_labels["community"])
    atlas_labels["CommunityCode"] = atlas_labels["CommunityCode"].cat.codes
    return atlas_labels


def calc_connectome_properties(
    atlas_labels: pd.DataFrame,
    img: nib.Nifti1Image,
    data: np.ndarray,
    subjects: list,
    connectomes: list,
    atlas_name: str,
    threshold: int = 10,
) -> dict:
    data_dict = {}
    num_rois = atlas_labels.shape[0]
    for subj, conn in zip(subjects, connectomes):
        print(subj)
        df = atlas_labels.copy()
        regions = df["Label"] - 1
        regions = regions.values.tolist()
        out_file = conn.with_name(f"{atlas_name}_properties.csv")
        if out_file.exists():
            continue
        cur_conn = pd.read_csv(conn, header=None)
        cur_conn = cur_conn.loc[regions, regions].values
        cur_conn[cur_conn < threshold] = 0
        # cur_conn_bin = cur_conn > threshold
        degree = bct.degrees_und(cur_conn)
        strength = bct.strengths_und(cur_conn)
        centrality = bct.betweenness_wei(cur_conn)
        rich_club = bct.rich_club_wu(cur_conn, num_rois)

        for col, val in zip(
            ["Degree", "Strength", "Centrality", "RichClubness"],
            [degree, strength, centrality, rich_club],
        ):
            print(f"\t{col}")
            df[col] = val
            val_img = plot_property(img, data, df, col)
            nib.save(val_img, out_file.with_name(f"{atlas_name}_{col}.nii.gz"))
        df.to_csv(out_file)
        data_dict[subj] = df
    return data_dict


if __name__ == "__main__":
    mother_dir = Path("/media/groot/Yalla/media/MRI")
    parcellations_dir = Path("/media/groot/Data/Parcellations/MNI")
    parcellations = get_available_parcellations(
        mother_dir / "derivatives" / "dwiprep"
    )
    parcellations_dict = generate_parcellations_dict(
        parcellations_dir, parcellations
    )
    for atlas_name, atlas_files in parcellations_dict.items():
        print(atlas_name)
        img = atlas_files.get("atlas_img")
        data = img.get_fdata()
        atlas_labels = atlas_files.get("atlas_parcels")
        subjects = []
        connectomes = []
        for subj in mother_dir.glob("derivatives/dwiprep/sub-*"):
            for ses in subj.glob("ses-*"):
                subj_connectome = (
                    ses / "tractography" / f"{atlas_name}_1M_connectome.csv"
                )
                if subj_connectome.exists():
                    subjects.append(subj.name)
                    connectomes.append(subj_connectome)
        # connectomes = [f for f in ]
        print("Found", len(subjects), "structural connectomes")

        data_dict = calc_connectome_properties(
            atlas_labels, img, data, subjects, connectomes, atlas_name
        )
