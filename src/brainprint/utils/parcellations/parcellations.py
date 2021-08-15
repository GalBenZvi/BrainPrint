from enum import Enum
from pathlib import Path

current_dir = Path(__file__)
brainnetome_atlas = (
    current_dir / "Brainnetome" / "BN_Atlas_274_combined_1mm.nii.gz"
)
brainnetome_labels = current_dir / "Brainnetome" / "BNA_with_cerebellum.csv"
parcellations = {
    "Brainnetome": {"atlas": brainnetome_atlas, "labels": brainnetome_labels}
}
