"""
Definition of the :class:`SubjectResults` class.
"""
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

import nibabel as nib
import numpy as np
import pandas as pd
from brainprint.feature_generation import messages
from brainprint.feature_generation.utils.features import FEATURES
from brainprint.feature_generation.utils.parcellation import parcellate_metric
from brainprint.utils import Modality, parcellations


class SubjectResults:
    """
    Facilitates navigation and extraction of BIDS-compatible preprocessing
    results.
    """

    #: Default atlas name to use for calculating regional statistics.
    DEFAULT_ATLAS_NAME: str = "Brainnetome"

    #: Name of the BIDS directory containing NIfTI-formatted scans and
    #: metadata.
    BIDS_DIR_NAME: str = "NIfTI"

    #: Diffusion-weighted imaging preprocessing results directory.
    DIFFUSION_RELATIVE_PATH: str = "derivatives/dwiprep"
    TENSOR_DIRECTORY_PATH: str = "tensors_parameters/coreg_FS"

    #: Functional imaging preprocessing results.
    FUNCTIONAL_RELATIVE_PATH: str = "derivatives/fmriprep"

    #: Name of the structural derivatives directory within fMRIPrep's results.
    STRUCTURAL_DERIVATIVE_DIR: str = "anat"

    #: Pattern to use for sessions in the BIDS directory hierarchy.
    SESSION_DIRECTORY_PATTERN: str = "ses-*"
    FIRST_SESSION: str = "ses-1"

    #: Pattern to use for subjects in the BIDS directory hierarchy.
    SUBJECT_DIRECTORY_PATTERN: str = "sub-*"

    #: Dispatch table from modality to derivative paths generation function.
    DERIVATIVES_FROM_MODALITY: Dict[Modality, Callable] = {
        Modality.DIFFUSION: "get_dwi_paths",
        Modality.STRUCTURAL: "get_smri_paths",
    }

    #: Parameters to be extracted by modality.
    PARAMETERS: Dict[Modality, List[str]] = FEATURES

    #: Native grey matter parcellation file template.
    NATIVE_GM_PATTERN: str = "{atlas_name}_native_GM.nii.gz"

    #: File name pattern to use to locate sMRI derivatives.
    SMRI_DERIVATIVE_PATTERN: str = "{subject_id}_{parameter}.nii.gz"

    # Cached reference of the derivates dictionary.
    _results_dict: dict = None

    def __init__(self, base_dir: Path, subject_id: str) -> None:
        """
        Initialize a new :class:`SubjectResults` instance.

        Parameters
        ----------
        base_dir : Path
            Base project directory
        subject_id : str
            Subject ID to extract parameters for
        """
        self.base_dir: Path = Path(base_dir)
        self.subject_id: str = subject_id
        self.native_parcellation: Path = self.get_subject_parcellation()

    def get_diffusion_derivatives_path(self) -> Path:
        """
        Returns the path of the diffusion-weighted imaging preprocessing
        derivatives.

        Returns
        -------
        Path
            DWI preprocessing derivatives
        """
        return self.base_dir / self.DIFFUSION_RELATIVE_PATH / self.subject_id

    def get_functional_derivatives_path(self) -> Path:
        """
        Returns the path of the functional imaging preprocessing derivatives.

        Returns
        -------
        Path
            fMRI preprocessing derivatives
        """
        return self.base_dir / self.FUNCTIONAL_RELATIVE_PATH / self.subject_id

    def get_structural_derivatives_path(self) -> Path:
        """
        Returns the path of the sMRI preprocessing derivatives.

        Returns
        -------
        Path
            sMRI preprocessing derivatives
        """
        functional = self.get_functional_derivatives_path()
        sessions = list(functional.glob(self.SESSION_DIRECTORY_PATTERN))
        if not sessions:
            return
        longitudinal = len(sessions) > 1
        buffer_dir = "" if longitudinal else sessions[0]
        return functional / buffer_dir / self.STRUCTURAL_DERIVATIVE_DIR

    def get_subject_parcellation(self, atlas_name: str = None) -> Path:
        """
        Returns the path of the native grey matter parcellation file for this
        subject.

        Parameters
        ----------
        atlas_name : str, optional
            Atlas name to find the parcellation file by, by default None

        Returns
        -------
        Path
            Native grey matter parcellation
        """
        atlas_name = atlas_name or self.DEFAULT_ATLAS_NAME
        atlas_file = self.NATIVE_GM_PATTERN.format(atlas_name=atlas_name)
        path = self.structural_derivatives_path / atlas_file
        if not path.exists():
            message = messages.MISSING_PARCELLATION.format(
                subject_id=self.subject_id
            )
            warnings.warn(message)
        return path

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
    ) -> dict:
        """
        Returns the paths of structural preprocessing derivatives of the first
        session.

        Returns
        -------
        dict
            Derivated file path by parameter ID
        """
        subject_derivatives = defaultdict(dict)
        for parameter in self.PARAMETERS.get(Modality.STRUCTURAL):
            name = self.SMRI_DERIVATIVE_PATTERN.format(
                subject_id=self.subject_id, parameter=parameter.lower()
            )
            derivative_path = self.structural_derivatives_path / name
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

    def summarize_subject_metrics(
        self, atlas_name: str = None
    ) -> pd.DataFrame:
        """
        Returns subject parameters across modalities summarized by region.

        Parameters
        ----------
        atlas_name : str, optional
            Atlas to use for regional statistics calculation, by default None

        Returns
        -------
        pd.DataFrame
            Summarized subject parameters
        """
        atlas_name = atlas_name or self.DEFAULT_ATLAS_NAME
        atlas_data = nib.load(self.native_parcellation).get_fdata()
        atlas_labels = parcellations.get(atlas_name).get("labels")
        template_df = pd.read_csv(atlas_labels, index_col=0)
        subject_metrics = {}
        for sessions_dict in self.derivative_dict.values():
            for session_id, metric_derivatives in sessions_dict.items():
                df = template_df.copy()
                for metric, path in metric_derivatives.items():
                    df[metric] = self.parcellate_metric(path, atlas_data, df)
                subject_metrics[session_id] = df
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


if __name__ == "__main__":
    base_dir = Path("/media/groot/Yalla/media/MRI")
    subj_id = "sub-233"
    res = SubjectResults(base_dir, subj_id)
    print(res.summarize_subject_metrics())
