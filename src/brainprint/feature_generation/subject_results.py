from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

from brainprint.feature_generation.utils.features import FEATURES
from brainprint.utils import Modality


class SubjectResults:
    DEFAULT_ATLAS_NAME: str = "Brainnetome"
    BIDS_DIR_NAME: str = "NIfTI"
    DIFFUSION_RELATIVE_PATH: str = "derivatives/dwiprep"
    FUNCTIONAL_RELATIVE_PATH: str = "derivatives/fmriprep"
    STRUCTURAL_DERIVATIVE_DIR: str = "anat"
    SESSION_DIRECTORY_PATTERN: str = "ses-*"
    FIRST_SESSION: str = SESSION_DIRECTORY_PATTERN.replace("*", "1")
    SUBJECT_DIRECTORY_PATTERN: str = "sub-*"
    TENSOR_DIRECTORY_PATH: str = "tensors_parameters/coreg_FS"
    DERIVATIVES_FROM_MODALITY: Dict[Modality, Callable] = {
        Modality.DIFFUSION: "get_dwi_paths",
        Modality.STRUCTURAL: "get_smri_paths",
    }
    PARAMETERS: Dict[Modality, List[str]] = FEATURES

    def __init__(self, base_dir: Path, subject_id: str) -> None:
        self.base_dir = base_dir
        self.subject_id = subject_id

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
        buffer_dir = sessions[0] if longitudinal else ""
        return functional / buffer_dir / self.STRUCTURAL_DERIVATIVE_DIR

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
        derivatives_dir = self.structural_derivatives_path
        subject_derivatives = defaultdict(dict)
        atlas_file = f"{atlas_name}_native_GM.nii.gz"
        subject_derivatives["native_parcellation"] = (
            derivatives_dir / atlas_file
        )
        for parameter in self.PARAMETERS.get(Modality.STRUCTURAL):
            derivative_file = f"{self.subject_id}_{parameter.lower()}.nii.gz"
            derivative_path = derivatives_dir / derivative_file
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
                subject_dict["Features"][modality] = getter()
        return subject_dict

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
        return self.get_derivative_dict()

    # def check_subject_derivatives(base_dir: Path, subject_id: str) -> bool:
    #     """
    #     Checks whether a given subject has the required derivatives within
    #     *base_dir*.

    #     Parameters
    #     ----------
    #     base_dir : Path
    #         Base project directory
    #     subject_id : str
    #         Subject ID (and expected directory name)

    #     Returns
    #     -------
    #     bool
    #         Whether this subject has the required derivates or not
    #     """
    #     return all(
    #         getter(base_dir, subject_id).exists()
    #         for getter in DERIVATIVE_PATH_GETTER.values()
    #     )

    # def generate_preprocessed_subjects(base_dir: Path) -> list:
    #     """
    #     Iterate over the main directory to find subjects that have all necessary
    #     files.

    #     Parameters
    #     ----------
    #     base_dir : Path
    #         Path to project's main directory

    #     Returns
    #     -------
    #     list
    #         List of subjects' identifiers comprised only of "valid" subjects
    #     """
    #     bids_dir = Path(base_dir) / BIDS_DIR_NAME
    #     subject_dirs = bids_dir.glob(SUBJECT_DIRECTORY_PATTERN)
    #     for subject_dir in subject_dirs:
    #         has_derivates = check_subject_derivatives(
    #             base_dir, subject_dir.name
    #         )
    #         if has_derivates:
    #             yield subject_dir.name
