"""
Download EEG datasets from OpenNeuro.

Usage:
    python download_data.py                # Download all
    python download_data.py --dataset UC   # Download one
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATASET_INFO = {
    'UC': {
        'openneuro_id': 'ds003490',
        'description': 'UC San Diego EEG PD Dataset',
        'subjects': '15 PD + 16 HC',
    },
    'UNM': {
        'openneuro_id': 'ds002778',
        'description': 'UNM EEG PD Dataset',
        'subjects': '14 PD + 14 HC',
    },
    'Iowa': {
        'openneuro_id': 'ds004584',
        'description': 'Iowa EEG PD Dataset',
        'subjects': '14 PD + 14 HC',
    },
}

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_dataset(name):
    """Download one dataset from OpenNeuro."""
    info = DATASET_INFO[name]
    ds_id = info['openneuro_id']
    target = DATA_DIR / name

    print(f"\nDownloading {info['description']} ({ds_id})")
    print(f"  Target: {target}")

    if target.exists() and any(target.rglob("sub-*")):
        print(f"  Already exists. Skipping.")
        return True

    target.mkdir(parents=True, exist_ok=True)

    try:
        import openneuro
        openneuro.download(dataset=ds_id, target_dir=str(target))
        print(f"  Done.")
        return True
    except ImportError:
        print("  openneuro-py not installed. Install: pip install openneuro-py")
    except Exception as e:
        print(f"  openneuro failed: {e}")

    # Fallback: AWS
    try:
        s3_url = f"s3://openneuro.org/{ds_id}"
        subprocess.run(
            ["aws", "s3", "sync", "--no-sign-request", s3_url, str(target)],
            check=True,
        )
        return True
    except Exception:
        pass

    print(f"  Manual download: https://openneuro.org/datasets/{ds_id}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['UC', 'UNM', 'Iowa', 'all'], default='all')
    args = parser.parse_args()

    datasets = list(DATASET_INFO.keys()) if args.dataset == 'all' else [args.dataset]

    for name in datasets:
        download_dataset(name)

    print(f"\nNext: cd src && python main.py --real --k_shot 5")


if __name__ == "__main__":
    main()
