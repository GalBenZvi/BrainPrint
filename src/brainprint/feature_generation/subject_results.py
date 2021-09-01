import warnings
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List
from brainprint.feature_generation.utils.parcellation import parcellate_metric
from brainprint.feature_generation.utils.functions import (
    at_ants,
    coregister_tensors_single_session,
    crop_to_mask,
    epi_reg,
    coregister_tensors_longitudinal,
    tractography_pipeline,
)
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
    TENSOR_COREG_DIRECTORY_PATH: str = "tensors_parameters/coreg_FS"
    TENSOR_NATIVE_DIRECTORY_PATH: str = "tensors_parameters/native"

    #: Diffusion imaging preprocessing results.
    REGISTERATION_DIRECTORY_PATH: str = "registrations"
    FREESURFER_REGISTERATION_PATH: str = "registrations/preprocessed_FS"
    BIAS_CORRECTED_FNAME: str = "bias_corrected.mif"
    TRACTOGRAPHY_DIRECTORY_PATH: str = "tractography"
    #: Functional imaging preprocessing results.
    FUNCTIONAL_RELATIVE_PATH: str = "derivatives/fmriprep"
    STRUCTURAL_DERIVATIVE_DIR: str = "anat"
    SESSION_DIRECTORY_PATTERN: str = "ses-*"
    FIRST_SESSION: str = "ses-1"
    SUBJECT_DIRECTORY_PATTERN: str = "sub-*"
    TENSOR_DIRECTORY_PATH: str = "tensors_parameters/coreg_FS"
    DERIVATIVES_FROM_MODALITY: Dict[Modality, Callable] = {
        Modality.DIFFUSION: "get_coregistered_dwi_paths",
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

    def get_bids_path(self) -> Path:
        return self.base_dir / self.BIDS_DIR_NAME / self.subject_id

    def get_diffusion_derivatives_path(self) -> Path:
        return self.base_dir / self.DIFFUSION_RELATIVE_PATH / self.subject_id

    def get_functional_derivatives_path(self) -> Path:
        return self.base_dir / self.FUNCTIONAL_RELATIVE_PATH / self.subject_id

    def get_structural_derivatives_path(self) -> Path:
        functional = self.get_functional_derivatives_path()
        sessions = list(functional.glob(self.SESSION_DIRECTORY_PATTERN))

        if not sessions:
            return
        self.longitudinal = len(sessions) > 1
        buffer_dir = "" if self.longitudinal else sessions[0]
        return functional / buffer_dir / self.STRUCTURAL_DERIVATIVE_DIR

    def get_subject_parcellation(self, atlas_name: str = None) -> Path:
        atlas_name = atlas_name or self.DEFAULT_ATLAS_NAME
        atlas_file = f"{atlas_name}_native_GM.nii.gz"
        path = self.structural_derivatives_path / atlas_file
        if not path.exists():
            warnings.warn(
                f"Subject {self.subject_id} does not have a native GM parcellation.\nWill try to generate one."
            )
            try:
                self.coregister_atlas_to_native(atlas_name)
                return path
            except Exception:
                warnings.warn(f"Native GM parcellation was unsuccessful")
        else:
            return self.structural_derivatives_path / atlas_file

    def coregister_atlas_to_native(self, atlas_name: str = None):
        """
        Coregister atlas from standard space to subject's native space
        Parameters
        ----------
        atlas_name : str, optional
            [description], by default None
        """
        atlas_name = atlas_name or self.DEFAULT_ATLAS_NAME
        coregistered_atlas = (
            self.structural_derivatives_path / f"{atlas_name}_native.nii.gz"
        )
        if not coregistered_atlas.exists():
            at_ants(
                parcellations.get(atlas_name).get("atlas"),
                self.t1w_brain,
                self.standard_to_native_transform,
                coregistered_atlas,
                nn=True,
            )
        cropped_to_gm = (
            coregistered_atlas.parent
            / coregistered_atlas.name.replace(".nii.gz", "_GM.nii.gz")
        )
        crop_to_mask(coregistered_atlas, self.gm_mask, cropped_to_gm)

    def get_standard_to_native_xfm(self) -> Path:
        """
        Returns the path of the nonlinear transformation from standard to subject's native space.
        Returns
        -------
        Path
            Standard to native space transformation h5 file.
        """
        buffer = (
            f"_{self.structural_derivatives_path.parent.name}"
            if not self.longitudinal
            else ""
        )
        return (
            self.structural_derivatives_path
            / f"{self.subject_id}{buffer}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
        )

    def get_native_to_structural_preprocessed_xfm(self) -> Path:
        """
        Return the path of the linear transformation between subject's native anatomical space and its preprocessed one

        Returns
        -------
        Path
            The linear transformation between subject's native anatomical space and its preprocessed one
        """
        buffer = (
            f"_{self.structural_derivatives_path.parent.name}"
            if not self.longitudinal
            else ""
        )
        xfm_dict = {}
        for ses in self.get_functional_sessions():
            try:
                xfm_dict[ses] = [
                    f
                    for f in self.structural_derivatives_path.glob(
                        f"{self.subject_id}_{ses}_*_from-orig_to-T1w_mode-image_xfm.txt"
                    )
                    if "uncorrected" not in f.name
                ][0]
            except IndexError:
                xfm_dict[ses] = None

        return xfm_dict

    def get_native_gm_mask(self) -> Path:
        """
        Returns the path of the native gray matter probabalistic mask.
        Returns
        -------
        Path
            Native gray matter probabalistic mask.
        """
        buffer = (
            f"_{self.structural_derivatives_path.parent.name}"
            if not self.longitudinal
            else ""
        )
        return (
            self.structural_derivatives_path
            / f"{self.subject_id}{buffer}_label-GM_probseg.nii.gz"
        )

    def get_preprocessed_T1w(self) -> Path:
        """
        Returns the path of the native preprocessed T1w image.
        Returns
        -------
        Path
            Native preprocessed T1w image.
        """
        buffer = (
            f"_{self.structural_derivatives_path.parent.name}"
            if not self.longitudinal
            else ""
        )
        return (
            self.structural_derivatives_path
            / f"{self.subject_id}{buffer}_desc-preproc_T1w.nii.gz"
        )

    def get_t2w(self) -> dict:
        t2w_dict = {}
        for ses in self.get_functional_sessions():
            try:
                t2w_dict[ses] = [
                    f
                    for f in self.bids_path.glob(
                        f"{ses}/anat/{self.subject_id}_{ses}*_T2w.nii.gz"
                    )
                    if "uncorrected" not in f.name
                ][0]
            except IndexError:
                t2w_dict[ses] = None
        return t2w_dict

    def get_brain_mask(self) -> Path:
        """
        Returns the path of the native preprocessed T1w brain mask.
        Returns
        -------
        Path
            Native preprocessed T1w brain mask
        """
        buffer = (
            f"_{self.structural_derivatives_path.parent.name}"
            if not self.longitudinal
            else ""
        )
        return (
            self.structural_derivatives_path
            / f"{self.subject_id}{buffer}_desc-brain_mask.nii.gz"
        )

    def get_T1w_brain(self) -> Path:
        """
        Returns the path of the native preprocessed T1w brain.
        Returns
        -------
        Path
            Native preprocessed T1w brain
        """
        buffer = (
            f"_{self.structural_derivatives_path.parent.name}"
            if not self.longitudinal
            else ""
        )
        t1w_brain = (
            self.structural_derivatives_path
            / f"{self.subject_id}{buffer}_desc-brain.nii.gz"
        )
        if not t1w_brain.exists():
            cmd = f"fslmaths {self.preprocessed_t1w} -mul {self.brain_mask} {t1w_brain}"
            os.system(cmd)
        return t1w_brain

    def get_mean_b0(self) -> Path:
        """
        Returns the path of subject's native mean B0 image.
        Returns
        -------
        Path
            Subject's native mean B0 image.
        """
        dwi_derivatives = self.diffusion_derivatives_path
        longitudinal = len([ses for ses in dwi_derivatives.glob("ses-*")]) > 1
        registrations = dwi_derivatives / self.REGISTERATION_DIRECTORY_PATH
        if longitudinal:
            return (
                registrations / "mean_b0" / "mean_coregistered_mean_b0.nii.gz"
            )
        else:
            return (
                registrations
                / "mean_b0"
                / f"mean_b0_{self.FIRST_SESSION}.nii.gz"
            )

    def get_epi_to_t1w_transform(self) -> Path:
        """
        Returns the path of subject's coregistered mean B0 image.
        Returns
        -------
        Path
            Subject's coregistered mean B0 image.
        """
        coreg_epi = (
            self.diffusion_derivatives_path
            / self.FREESURFER_REGISTERATION_PATH
            / "mean_epi2anatomical.nii.gz"
        )
        transform = epi_reg(
            self.mean_b0, self.preprocessed_t1w, self.t1w_brain, coreg_epi
        )
        return transform

    def coregister_tensors(self):
        """
        Apply either longitudinal or single-session coregisterations between tensor-derived parameters and preprocessed T1w image
        """
        dwi_derivatives = self.diffusion_derivatives_path
        longitudinal = len([ses for ses in dwi_derivatives.glob("ses-*")]) > 1
        registerations_dir = (
            self.diffusion_derivatives_path
            / self.REGISTERATION_DIRECTORY_PATH
            / "mean_b0"
        )
        fs_dir = (
            self.diffusion_derivatives_path
            / self.FREESURFER_REGISTERATION_PATH
        )
        epi_to_anatomical = self.get_epi_to_t1w_transform()
        if longitudinal:
            anat2epi = coregister_tensors_longitudinal(
                registerations_dir,
                fs_dir,
                self.t1w_brain,
                epi_to_anatomical,
                self.get_dwi_paths(),
            )
        else:
            anat2epi = coregister_tensors_single_session(
                self.t1w_brain,
                epi_to_anatomical,
                self.get_dwi_paths(),
            )
        return anat2epi

    def get_dwi_paths(self) -> dict:
        """
        Locates native-space tensor-derived metrics files within subject's dwiprep outputs.

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
                derivatives_dir
                / session_id
                / self.TENSOR_NATIVE_DIRECTORY_PATH
            )
            for parameter in self.PARAMETERS.get(Modality.DIFFUSION):
                derivative_path = tensor_dir / f"{parameter}.mif"
                if derivative_path.exists():
                    subject_derivatives[session_id][
                        parameter
                    ] = derivative_path
        return subject_derivatives

    def get_coregistered_dwi_paths(self) -> dict:
        """
        Locates coregistered tensor-derived metrics files within subject's dwiprep outputs.

        Returns
        -------
        dict
            A dictionary comprised of tensor-derived metrics files' paths for each
            of subject's sessions
        """
        derivatives_dir = self.diffusion_derivatives_path
        subject_derivatives = defaultdict(dict)
        session_dirs = derivatives_dir.glob(self.SESSION_DIRECTORY_PATTERN)
        flags = []
        for session_dir in session_dirs:
            session_id = session_dir.name
            tensor_dir = (
                derivatives_dir / session_id / self.TENSOR_COREG_DIRECTORY_PATH
            )
            for parameter in self.PARAMETERS.get(Modality.DIFFUSION):
                derivative_path = tensor_dir / f"{parameter}.nii.gz"

                subject_derivatives[session_id][parameter] = derivative_path
                if not derivative_path.exists():
                    flags.append(derivative_path)
        if any(flags):
            self.coregister_tensors()

        return subject_derivatives

    def get_original_dwis(self) -> dict:
        """
        Returns subject's original (4D) DWI files
        Returns
        -------
        dict
            Subject's original (4D) DWI files
        """
        derivatives_dir = self.diffusion_derivatives_path
        session_dirs = derivatives_dir.glob(self.SESSION_DIRECTORY_PATTERN)
        subject_derivatives = {}
        for session in session_dirs:
            session_id = session.name
            subject_derivatives[session_id] = (
                session / self.BIAS_CORRECTED_FNAME
            )
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
            else:
                subject_derivatives[self.FIRST_SESSION][parameter] = None
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
                for metric, path in tqdm.tqdm(
                    self.derivative_dict.get(key).get(ses).items()
                ):
                    try:
                        template_df[metric] = self.parcellate_metric(
                            path, atlas_data, template_df
                        )
                    except AttributeError:
                        continue
                break
            subject_metrics[ses] = template_df
        return subject_metrics

    def generate_connectome(
        self,
        streamlines_init: str = "5M",
        streamlines_post_cleanup: str = "1M",
        atlas_name: str = None,
    ):
        atlas_name = atlas_name or self.DEFAULT_ATLAS_NAME
        anat_preproc = self.preprocessed_t1w
        anat_mask = self.brain_mask
        t1w2epi_dict = self.coregister_tensors()
        anat_coreg_dict = self.get_native_to_structural_preprocessed_xfm()
        t2w_dict = self.get_t2w()
        dwi_dict = self.get_original_dwis()
        native_parcellation = self.native_parcellation
        connectomes_dict = {}
        for dwi_session, functional_session in zip(
            dwi_dict.keys(), anat_coreg_dict.keys()
        ):
            dwi = dwi_dict.get(dwi_session)
            t1w2epi = t1w2epi_dict.get(dwi_session)
            anat_coreg = anat_coreg_dict.get(functional_session)
            t2w = t2w_dict.get(functional_session)
            mean_bzero = (
                self.diffusion_derivatives_path
                / self.REGISTERATION_DIRECTORY_PATH
                / "mean_b0"
                / f"mean_b0_{dwi_session}.nii.gz"
            )
            out_dir = (
                self.diffusion_derivatives_path
                / dwi_session
                / self.TRACTOGRAPHY_DIRECTORY_PATH
            )
            connectomes_dict[dwi_session] = tractography_pipeline(
                anat_preproc,
                anat_mask,
                t1w2epi,
                anat_coreg,
                t2w,
                dwi,
                native_parcellation,
                mean_bzero,
                out_dir,
                streamlines_init,
                streamlines_post_cleanup,
                atlas_name,
            )
        return connectomes_dict

    @property
    def bids_path(self) -> Path:
        return self.get_bids_path()

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
    def standard_to_native_transform(self) -> Path:
        return self.get_standard_to_native_xfm()

    @property
    def gm_mask(self) -> Path:
        return self.get_native_gm_mask()

    @property
    def preprocessed_t1w(self) -> Path:
        return self.get_preprocessed_T1w()

    @property
    def t1w_brain(self) -> Path:
        return self.get_T1w_brain()

    @property
    def brain_mask(self) -> Path:
        return self.get_brain_mask()

    @property
    def mean_b0(self) -> Path:
        return self.get_mean_b0()


if __name__ == "__main__":
    base_dir = Path("/media/groot/Yalla/media/MRI")
    subj_id = "sub-233"
    res = SubjectResults(base_dir, subj_id)
    print(res.generate_connectome())
