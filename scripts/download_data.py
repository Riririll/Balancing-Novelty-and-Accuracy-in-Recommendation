"""Download and extract MovieLens 100K (ml-latest-small) dataset."""
import os
import zipfile
import urllib.request
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download(url: str, dest: Path):
    print(f"Downloading {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  -> saved to {dest}")


def extract(zip_path: Path, out_dir: Path):
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"  -> extracted into {out_dir}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ml-latest-small.zip"
    target = DATA_DIR / "ml-latest-small"

    if target.exists():
        print(f"Dataset already at {target}. Skipping.")
        return

    if not zip_path.exists():
        download(URL, zip_path)

    extract(zip_path, DATA_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
