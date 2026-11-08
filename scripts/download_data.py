"""Download the competition files into data/.

Needs Kaggle API credentials (~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY)
and an account that has joined the competition.

    python scripts/download_data.py
"""
import sys
import zipfile
from pathlib import Path

COMPETITION = "workstation-rental-price-tier-prediction"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Install the Kaggle client first:  pip install kaggle")
        return 1

    api = KaggleApi()
    api.authenticate()

    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading {COMPETITION} into {DATA_DIR} ...")
    api.competition_download_files(COMPETITION, path=str(DATA_DIR), quiet=False)

    for zip_path in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()

    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        path = DATA_DIR / name
        status = f"{path.stat().st_size / 1e6:8.1f} MB" if path.exists() else "MISSING"
        print(f"  {name:<24}{status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
