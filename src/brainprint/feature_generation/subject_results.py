import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List
from brainprint.feature_generation.utils.parcellation import parcellate_metric

import tqdm
import pandas as pd
import nibabel as nib
import numpy as np
from brainprint.feature_generation.utils.features import FEATURES
from brainprint.utils import Modality, parcellations


class SubjectResults:
    DEFAULT_ATLAS_NAME: str = "Brainnetome"
    BIDS_DIR_NAME: str = "NIfTI"
    DIFFUSION_RELATIVE_PATH: str = "derivatives/dwiprep"
    FUNCTIONAL_RELATIVE_PATH: str = "derivatives/fmriprep"
    STRUCTURAL_DERIVATIVE_DIR: str = "anat"
    SESSION_DIRECTORY_PATTERN: str = "ses-*"
    FIRST_SESSION: str = "ses-1"
    SUBJECT_DIRECTORY_PATTERN: str = "sub-*"
    TENSOR_DIRECTORY_PATH: str = "tensors_parameters/coreg_FS"
    DERIVATIVES_FROM_MODALITY: Dict[Modality, Callable] = {
        Modality.DIFFUSION: "get_dwi_paths",
        Modality.STRUCTURAL: "get_smri_paths",
    }
    PARAMETERS: Dict[Modality, List[str]] = FEATURES

    _results_dict: dict = None

    def __init__(self, base_dir: Path, subject_id: str) -> None:
        self.base_dir = base_dir
        self.subject_id = subject_id
        self.native_parcellation = self.get_subject_parcellation()

    def get_functional_sessions(self) -> list:
        return [
            ses.name
            for ses in self.functional_derivatives_path.glob(
                self.SESSION_DIRECTORY_PATTERN
            )
        ]

    def get_diffusion_derivatives_path(self) -> Path:
        return self.base_dir / self.DIFFUSION_RELATIVE_PATH / self.subject_id

    def get_functional_derivatives_path(self) -> Path:
        return self.base_dir / self.FUNCTIONAL_RELATIVE_PATH / self.subject_id

    def get_structural_derivatives_path(self) -> Path:
        functional = self.get_functional_derivatives_path()
        sessions = list(functional.glob(self.SESSION_DIRECTORY_PATTERN))
        if not sessions:
            return
        longitudinal = len(sessions) > 1
        buffer_dir = "" if longitudinal else sessions[0]
        return functional / buffer_dir / self.STRUCTURAL_DERIVATIVE_DIR

    def get_subject_parcellation(self, atlas_name: str = None) -> Path:
        atlas_name = atlas_name or self.DEFAULT_ATLAS_NAME
        atlas_file = f"{atlas_name}_native_GM.nii.gz"
        path = self.structural_derivatives_path / atlas_file
        if not path.exists():
            warnings.warn(
                f"Subject {self.subject_id} does not have a native GM parcellation."
            )
        else:
            return self.structural_derivatives_path / atlas_file

    def get_dwi_paths(self) -> dict:
        """
        Locates tensor-derived metrics files within subject's dwiprep outputs.

        Returns
        -------
        dict
            A dictionary comprised of tensor-derived metrics files' paths for each
            of subject's sessions
        """
        derivatives_dir = self.diffusion_derivatives_path
        subject_derivatives = defaultdict(dict)
        session_dirs = derivatives_dir.glob(self.SESSION_DIRECTORY_PATTERN)
        for session_dir in session_dirs:
            session_id = session_dir.name
            tensor_dir = (
                derivatives_dir / session_id / self.TENSOR_DIRECTORY_PATH
            )
            for parameter in self.PARAMETERS.get(Modality.DIFFUSION):
                derivative_path = tensor_dir / f"{parameter}.nii.gz"
                if derivative_path.exists():
                    subject_derivatives[session_id][
                        parameter
                    ] = derivative_path
        return subject_derivatives

    def get_smri_paths(
        self,
        atlas_name: str = DEFAULT_ATLAS_NAME,
    ) -> dict:
        """
        Returns the paths of structural preprocessing derivatives of the first
        session.

        Parameters
        ----------
        atlas_name : str
            Atlas name

        Returns
        -------
        dict
            Derivated file path by parameter ID
        """
        subject_derivatives = defaultdict(dict)
        for parameter in self.PARAMETERS.get(Modality.STRUCTURAL):
            derivative_file = f"{self.subject_id}_{parameter.lower()}.nii.gz"
            derivative_path = (
                self.structural_derivatives_path / derivative_file
            )
            if derivative_path.exists():
                subject_derivatives[self.FIRST_SESSION][
                    parameter
                ] = derivative_path
        return subject_derivatives

    def get_derivative_dict(self, atlas_name: str = None) -> dict:
        """
        Locate subject's needed files for extraction of features.

        Parameters
        ----------
        atlas_name : str
            Atlas name

        Returns
        -------
        dict
            Dictionary comprised of paths to needed files and their corresponding
            keys
        """
        atlas_name = atlas_name or self.DEFAULT_ATLAS_NAME
        subject_dict = defaultdict(dict)
        for modality, getter_name in self.DERIVATIVES_FROM_MODALITY.items():
            getter = getattr(self, getter_name, None)
            if getter is not None:
                subject_dict[modality] = getter()
        return subject_dict

    def parcellate_metric(
        self,
        path: Path,
        atlas_data: np.ndarray,
        atlas_template_df: pd.DataFrame,
    ):
        """
        Returns the summarized metric information.

        Parameters
        ----------
        path : Path
            Path to metric's image
        atlas_name : Path
            Atlas name
        atlas_df : pd.DataFrame
            Path to subject's template parcellation pd.DataFrame

        Parameters
        ----------
        path : Path
            Path to metric's image
        atlas_data : np.ndarray
            A np.ndarray comprised of parcellation atlas' labels
        atlas_template_df : pd.DataFrame
            A pd.DataFrame to be filled with labels' summarized values.

        Returns
        -------
        np.ndarray comprised of summarized values of the image given as input.
        """
        metric_img = nib.load(path)
        metric_data = metric_img.get_fdata()
        temp = np.zeros(atlas_template_df.shape[0])
        for i, parcel in enumerate(atlas_template_df.index):
            label = atlas_template_df.loc[parcel, "Label"]
            mask = atlas_data == label
            temp[i] = np.nanmean(metric_data[mask.astype(bool)])
        return temp

    def summarize_subject_metrics(self, atlas_name: str = None):
        atlas_name = atlas_name or self.DEFAULT_ATLAS_NAME
        atlas_img = nib.load(self.get_subject_parcellation())
        atlas_data = atlas_img.get_fdata()
        subject_metrics = {}
        for ses in [self.FIRST_SESSION]:  ### WE NEED TO TALK ABOUT THIS
            template_df = pd.read_csv(
                parcellations.get("Brainnetome").get("labels"), index_col=0
            ).copy()
            for key in self.derivative_dict.keys():
                for metric, path in (
                    self.derivative_dict.get(key).get(ses).items()
                ):
                    template_df[metric] = self.parcellate_metric(
                        path, atlas_data, template_df
                    )
            subject_metrics[ses] = template_df
        return subject_metrics

    @property
    def diffusion_derivatives_path(self) -> Path:
        return self.get_diffusion_derivatives_path()

    @property
    def functional_derivatives_path(self) -> Path:
        return self.get_functional_derivatives_path()

    @property
    def structural_derivatives_path(self) -> Path:
        return self.get_structural_derivatives_path()

    @property
    def derivative_dict(self) -> dict:
        if self._results_dict is None:
            self._results_dict = self.get_derivative_dict()
        return self._results_dict

    @property
    def sessions(self) -> list:
        return self.get_functional_sessions()


if __name__ == "__main__":
    base_dir = Path("/media/groot/Yalla/media/MRI")
    subj_id = "sub-233"
    res = SubjectResults(base_dir, subj_id)
    print(res.summarize_subject_metrics())
