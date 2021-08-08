"""
Definition of the :class:`ReconAllResults` class.
"""
from pathlib import Path
from typing import List, Union

import pandas as pd
from preprocessing.smri.recon_all.stats import ReconAllStats


class ReconAllResults:
    """
    Facilitates access to a recon-all results directory.
    """
    STATS_DIR: str = "stats"

    def __init__(self, path: Path):
        """
        Initializes a new :class:`ReconAllResults` instance.

        Parameters
        ----------
        path : Path
            Results directory path
        """
        self.path = Path(path)
        self.stats = ReconAllStats(path / self.STATS_DIR)

    @classmethod
    def extract_stats(cls, path: Union[Path, List[Path]]) -> pd.DataFrame:
        """
        Extracts anatomical statistics from one or many result directories.

        Parameters
        ----------
        path : Union[Path, List[Path]]
            Results directory path or a list of such

        Returns
        -------
        pd.DataFrame
            Anatomical statistics
        """
        if isinstance(path, Path):
            return ReconAllStats(path).to_dataframe()
        else:
            all_stats = None
            for run_path in path:
                stats_path = run_path / cls.STATS_DIR
                stats = ReconAllStats(stats_path).to_dataframe()
                if not stats.empty:
                    stats["Subject ID"] = stats_path.parent.name
                    if all_stats is None:
                        all_stats = stats.copy()
                    else:
                        all_stats = all_stats.append(stats)
            indices = ["Subject ID"] + ReconAllStats.INDICES
            return all_stats.reset_index().set_index(indices)
