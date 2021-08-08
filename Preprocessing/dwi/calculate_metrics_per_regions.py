from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import tqdm


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
    # print(df)
    # df.index = df.Label.astype(int)
    df = pd.DataFrame(
        df.iloc[:, 0].values, columns=["ROIname"], index=df.index.values
    )
    return df


def read_aal_parcels(atlas_txt: Path) -> pd.DataFrame:
    df = pd.read_csv(atlas_txt, sep=" ", header=None)
    df.columns = ["Label", "ROIname", "2"]
    return df.iloc[:, :-1]


def gather_subject_data(
    derivatives_dir: Path, subj: str, atlas_name: str, norm_method: str = "CAT"
) -> dict:
    native_parcels = [
        f
        for f in derivatives_dir.glob(
            f"{subj}/registrations/preprocessed_{norm_method}*/{atlas_name}_native*"
        )
    ][0]
    return native_parcels


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
    norm_method: str = "CAT",
    coreg_dirname: str = "coregistered",
):
    subjects_dict = {}
    for subj in derivatives_dir.glob("sub-*"):
        print(subj.name)
        try:
            subj_data = {}
            native_parcels = gather_subject_data(
                derivatives_dir, subj.name, atlas_name, norm_method
            )
            atlas_data = nib.load(native_parcels).get_fdata()
        except FileNotFoundError:
            print(
                f"No {atlas_name} native parcellation found for {subj.name}!"
            )
            continue
        for session in subj.glob("ses-*"):
            print(session.name)
            session_df = atlas_parcels.copy()
            tensor_dir = session / "tensors_parameters" / coreg_dirname
            out_fname = tensor_dir / f"{atlas_name}_parcels.csv"
            subj_data[session.name] = out_fname
            # if not out_fname.exists():
            for param in tqdm.tqdm(tensor_dir.glob("*.nii.gz")):
                # print(param.name.split(".")[0])
                session_df[param.name.split(".")[0]] = average_parcels(
                    atlas_data, param, session_df
                )
            session_df.to_csv(out_fname)
        subjects_dict[subj.name] = subj_data
    return (
        subjects_dict,
        [f.name.split(".")[0] for f in tensor_dir.glob("*.nii.gz")],
    )


def generate_statistics(
    subjects: dict,
    out_dir: Path,
    parameters: list,
    atlas_parcels: pd.DataFrame,
):
    out_dict = {}
    for param in tqdm.tqdm(parameters):
        out_dict[param] = {}
        # print(param)
        param_dir = out_dir / param
        param_dir.mkdir(exist_ok=True)
        param_df = pd.DataFrame(
            index=subjects.keys(), columns=atlas_parcels.ROIname.values
        )
        for session in ["ses-1", "ses-2"]:
            out_fname = param_dir / f"{session}.csv"
            # print(out_fname)
            out_dict[param][session] = out_fname
            # if out_fname.exists():
            #     continue
            session_df = param_df.copy()
            for subj, subj_data in subjects.items():
                # print(subj)
                session_df.loc[subj] = pd.read_csv(
                    subj_data.get(session), index_col=0
                )[param].values
            session_df.to_csv(out_fname)
    return out_dict


def calculate_statistics(out_dir: Path, inputs: dict):
    for param, sessions in inputs.items():
        # print(Path(out_dir / param).exists())
        out_fname = out_dir / param / "statistics.csv"
        # if out_fname.exists():
        #     continue
        before, after = [
            pd.read_csv(session, index_col=0) for session in sessions.values()
        ]
        t, p = ttest_rel(before, after, axis=0, nan_policy="omit")
        df = pd.DataFrame(columns=["t", "p"], index=before.columns)
        df["t"] = t
        df["p"] = p
        df.to_csv(out_fname)
        inputs[param]["statistics"] = out_fname
    return inputs


def df_to_nii(
    df: pd.DataFrame,
    template_data: nib.Nifti1Image,
    affine,
    atlas_parcels: pd.DataFrame,
    out_dir: Path,
):
    p_data = np.zeros_like(template_data)
    p_vis_data = np.zeros_like(template_data)
    t_data = np.zeros_like(template_data)
    for i in df.index:
        # print(i)
        val = atlas_parcels.index[atlas_parcels.ROIname == i][0].astype(float)
        print(i, "---", val)
        mask = template_data == val
        for data, col in zip([p_data, t_data, p_vis_data], ["p", "t", "1-p"]):
            data[mask] = df[col][i]
    for data, fname in zip(
        [p_data, t_data, p_vis_data], ["p_values", "t_scores", "1-p_values"]
    ):
        img = nib.Nifti1Image(data, affine)
        nib.save(img, out_dir / f"{fname}.nii.gz")


def statistics_to_img(
    out_dir: Path, inputs: dict, template: nib.Nifti1Image, atlas_parcels
):
    temp_data = template.get_fdata()
    affine = template.affine
    for param in inputs.keys():
        param_dir = out_dir / param
        statistics = pd.read_csv(
            inputs.get(param).get("statistics"), index_col=0
        )
        statistics["1-p"] = 1 - statistics["p"]
        df_to_nii(statistics, temp_data, affine, atlas_parcels, param_dir)


if __name__ == "__main__":
    mother_dir = Path("/media/groot/Yalla/media/MRI/")
    derivatives_dir = mother_dir / "derivatives" / "dwiprep"
    parcellations_dir = Path("/media/groot/Data/Parcellations/MNI")
    parcellations = get_available_parcellations(derivatives_dir)
    parcellations_dict = generate_parcellations_dict(
        parcellations_dir, parcellations
    )
    atlas_name = "Brainnetome"
    atlas_data = parcellations_dict.get(atlas_name)
    print("###", atlas_name, "###")
    atlas_parcels = atlas_data.get("atlas_parcels")
    # try:
    statistics_dir = derivatives_dir / "statistics" / f"{atlas_name}_FS"
    statistics_dir.mkdir(exist_ok=True, parents=True)
    subjects, parameters = parcellate_subjects_data(
        derivatives_dir, atlas_parcels, "FS", "coreg_FS"
    )
    # # print(subjects)
    # statistics_dict = generate_statistics(
    #     subjects, statistics_dir, parameters, atlas_parcels
    # )
