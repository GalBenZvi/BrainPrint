from typing import List

import pandas as pd
from brainprint.utils import Modality

DWI_FEATURES: List[str] = [
    "MD",
    "FA",
    "AD",
    "RD",
    "EigenValue",
    "EigenVector",
    "CS",
    "CP",
    "CL",
]
SMRI_FEATURES: List[str] = [
    "Thickness",
    "Volume",
    "Sulc",
]

FEATURES = {
    Modality.DIFFUSION: DWI_FEATURES,
    Modality.STRUCTURAL: SMRI_FEATURES,
    Modality.FUNCTIONAL: [],
}


def df_to_series(
    tmp_df: pd.DataFrame,
    mindex: pd.MultiIndex,
    modality: Modality,
    features: dict = FEATURES,
):
    """
    Converts single subject's region-wise parameters DataFrame into a
    multi-indexed Series.

    Parameters
    ----------
    tmp_df : pd.DataFrame
        Subject's region-wise parameters DataFrame
    mindex : pd.MultiIndex
        MultiIndex to transform the DataFrame into
    features : list, optional
        A list of features existing as columns within tmp_df, by default
        FEATURES

    Returns
    -------
    pd.Series
        subject's data in a multi-indexed Series form.
    """
    series = pd.Series(index=mindex)
    features = features.get(modality)
    for feature in features:
        for i in tmp_df.index:
            roi = tmp_df.loc[i, "ROIname"]
            hemi = tmp_df.loc[i, "Hemi"]
            series.loc[(hemi, roi, modality, feature)] = tmp_df.loc[i, feature]
    return series
