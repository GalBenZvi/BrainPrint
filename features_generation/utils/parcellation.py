from pathlib import Path
import pandas as pd

parcellation_fname = Path(
    "/media/groot/Data/Parcellations/MNI/BN_Atlas_274_combined_1mm.nii.gz"
)
parcellation_labels = pd.read_csv(
    "/media/groot/Data/Parcellations/MNI/BNA_with_cerebellum.csv", index_col=0
)


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
