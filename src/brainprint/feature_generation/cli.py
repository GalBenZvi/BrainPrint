import argparse

parser = argparse.ArgumentParser(description="Generate features per subject.")
parser.add_argument("base_dir", type=str, nargs=1, required=True)
parser.add_argument("--subject-id", type=str, nargs="?", required=False)
parser.add_argument(
    "--atlas", default="Brainnetome", type=str, nargs="?", required=False
)
