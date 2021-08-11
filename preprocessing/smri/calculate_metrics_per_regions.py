from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import tqdm
from nipype.interfaces import fsl
from utils.parcellation import (
    parcellation_labels,
    parcellation_fname,
)


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


def read_atlas_parcels(atlas_parcels: Path) -> pd.DataFrame:
    df = pd.read_csv(atlas_parcels, sep=";", index_col=0)
    return df.iloc[:, :-1]


def read_parcels(atlas_parcels: Path) -> pd.DataFrame:
    return pd.read_csv(atlas_parcels, index_col=0)


def read_bna(atlas_parcels: Path) -> pd.DataFrame:
    df = pd.read_csv(atlas_parcels, sep=" ", header=None, index_col=0)
    df = pd.DataFrame(
        df.iloc[:, 0].values, columns=["ROIname"], index=df.index.values
    )
    return df


def read_aal_parcels(atlas_txt: Path) -> pd.DataFrame:
    df = pd.read_csv(atlas_txt, sep=" ", header=None)
    df.columns = ["Label", "ROIname", "2"]
    return df.iloc[:, :-1]


def gather_subject_data(
    derivatives_dir: Path, subj: str, atlas_name: str
) -> dict:
    fmriprep_dir = derivatives_dir / subj
    sessions = [f.name for f in fmriprep_dir.glob("ses-*")]
    if len(sessions) > 1:
        native_parcels = fmriprep_dir / "anat" / f"{atlas_name}_native.nii.gz"
        gm_probseg = fmriprep_dir / "anat" / f"{subj}_label-GM_probseg.nii.gz"
    else:
        native_parcels = (
            fmriprep_dir / sessions[0] / "anat" / f"{atlas_name}_native.nii.gz"
        )
        gm_probseg = (
            fmriprep_dir
            / sessions[0]
            / "anat"
            / f"{subj}_{sessions[0]}_label-GM_probseg.nii.gz"
        )
    return native_parcels, gm_probseg


def crop_to_gm(native_parcels: Path, gm_probseg: Path):
    cropped_parcels = (
        native_parcels.parent
        / f"{native_parcels.name.split('.')[0]}_GM.nii.gz"
    )
    if not cropped_parcels.exists():
        gm_mask = nib.load(gm_probseg).get_fdata().astype(bool)
        orig_img = nib.load(native_parcels)
        gm_parcels = orig_img.get_fdata()
        gm_parcels[~gm_mask] = np.nan
        gm_img = nib.Nifti1Image(gm_parcels, orig_img.affine)
        nib.save(gm_img, cropped_parcels)
    return cropped_parcels


def calc_average(parcel, atlas_data: np.ndarray, subj_data: np.ndarray, temp):
    # temp = []
    # for parcel in parcels:
    mask = atlas_data == parcel
    temp[parcel] = np.nanmean(subj_data[mask.astype(bool)])
    return temp


# @jit(parallel=True)
def average_parcels(
    atlas_data: np.ndarray, subj_img: Path, temp_df: pd.DataFrame
):
    subj_data = nib.load(subj_img).get_fdata()
    temp = np.zeros(temp_df.index.shape)
    for i, parcel in enumerate(temp_df.index):
        # print(temp_df.ROIname[parcel])
        roi = temp_df.loc[parcel, "Label"]
        mask = atlas_data == roi
        temp[i] = np.nanmean(subj_data[mask.astype(bool)])
    return temp


def parcellate_subjects_data(
    derivatives_dir: Path,
    atlas_parcels: pd.DataFrame,
    atlas_name: str = "Brainnetome",
    features: list = ["Thickness", "Volume", "Sulc"],
):
    subjects_dict = {}
    for subj in sorted(derivatives_dir.glob("sub-*")):
        if not subj.is_dir():
            continue
        print(subj.name)
        try:
            native_parcels_full, gm_mask = gather_subject_data(
                derivatives_dir, subj.name, atlas_name
            )
            native_parcels = crop_to_gm(native_parcels_full, gm_mask)
            atlas_data = nib.load(native_parcels).get_fdata()
        except FileNotFoundError:
            print(
                f"No {atlas_name} native parcellation found for {subj.name}!"
            )
            continue
        sessions = [ses.name for ses in subj.glob("ses-*")]
        if len(sessions) > 1:
            anat_dir = subj / "anat"
        else:
            anat_dir = subj / sessions[0] / "anat"
        temp_df = atlas_parcels.copy()
        out_file = anat_dir / f"{atlas_name}_parcels.csv"
        # if out_file.exists():
        #     continue
        for param in tqdm.tqdm(features):
            param_file = anat_dir / f"{subj.name}_{param.lower()}.nii.gz"
            if not param_file.exists():
                continue
            temp_df[param] = average_parcels(atlas_data, param_file, temp_df)
        temp_df.to_csv(out_file)


if __name__ == "__main__":
    mother_dir = Path("/media/groot/Yalla/media/MRI/")
    derivatives_dir = mother_dir / "derivatives" / "fmriprep"
    parcellations_dir = Path("/media/groot/Data/Parcellations/MNI")
    atlas_name = "Brainnetome"
    print("###", atlas_name, "###")
    # try:
    # statistics_dir = derivatives_dir / "statistics" / f"{atlas_name}_FS"
    # statistics_dir.mkdir(exist_ok=True, parents=True)
    subjects, parameters = parcellate_subjects_data(
        derivatives_dir,
        parcellation_labels,
    )
    # # print(subjects)
    # statistics_dict = generate_statistics(
    #     subjects, statistics_dir, parameters, atlas_parcels
    # )
