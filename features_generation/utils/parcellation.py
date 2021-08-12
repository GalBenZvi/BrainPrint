from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
import tqdm

parcellation_fname = Path(
    "/media/groot/Data/Parcellations/MNI/BN_Atlas_274_combined_1mm.nii.gz"
)
parcellation_labels = pd.read_csv(
    "/media/groot/Data/Parcellations/MNI/BNA_with_cerebellum.csv", index_col=0
)


def parcellate_metric(
    metric_path: Path, atlas_path: Path, atlas_df: pd.DataFrame
):
    """[summary]

    Parameters
    ----------
    metric_path : Path
        Path to metric's image
    atlas_img : Path
        Path to subject's parcellation image
    atlas_df : pd.DataFrame
        Path to subject's template parcellation pd.DataFrame
    """
    metric_img = nib.load(metric_path)
    atlas_data = nib.load(atlas_path).get_fdata()
    metric_data = metric_img.get_fdata()
    temp = np.zeros(atlas_df.shape[0])
    for i, parcel in tqdm.tqdm(enumerate(atlas_df.index)):
        label = atlas_df.loc[parcel, "Label"]
        mask = atlas_data == label
        temp[i] = np.nanmean(metric_data[mask.astype(bool)])
    return temp


def parcellate_hemisphere(roiname: str):
    """
    Parcellation Region's name into its related hemisphere
    Parameters
    ----------
    roiname : str
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if roiname.endswith("_L"):
        return "Left"
    elif roiname.endswith("_R"):
        return "Right"
    else:
        return "Cerebellum"


parcellation_labels["Hemi"] = parcellation_labels.ROIname.apply(
    parcellate_hemisphere
)
parcellation_labels["ROIname"] = (
    parcellation_labels.ROIname.str.replace("_L", "")
    .str.replace("_R", "")
    .str.replace("Cereb_", "")
)
