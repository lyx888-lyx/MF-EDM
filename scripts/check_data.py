#!/usr/bin/env python3
"""Check the exact feature files expected by MF-EDM, without loading pickles.

Usage:
    python scripts/check_data.py --dataset IEMOCAP
    python scripts/check_data.py --dataset all --root /path/to/MF-EDM

The check is intentionally dependency-free. A FOUND status verifies existence
and nonzero size only; it does not verify pickle schema or provenance.
"""
import argparse
from pathlib import Path

EXPECTED_FILES = {
    "IEMOCAP": (
        "IEMOCAP_features/IEMOCAP_features.pkl",
        "IEMOCAP_features/iemocap_features_roberta.pkl",
    ),
    "MELD": (
        "MELD_features/MELD_features_raw1.pkl",
        "MELD_features/meld_features_roberta.pkl",
    ),
    "KMIND": ("KMIND_features/KMIND_features_normalized.pkl",),
    "M3ED": ("M3ED_features/M3ED_features.pkl",),
}


def check_files(root, dataset):
    """Return (checked, missing) without unpickling untrusted data."""
    root = Path(root)
    selected = EXPECTED_FILES if dataset == "all" else {dataset: EXPECTED_FILES[dataset]}
    checked = []
    missing = []
    for name, paths in selected.items():
        for relative_path in paths:
            full_path = root / relative_path
            ok = full_path.is_file() and full_path.stat().st_size > 0
            checked.append((name, relative_path, ok))
            if not ok:
                missing.append(relative_path)
    return checked, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=("all", *EXPECTED_FILES))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)

    checked, missing = check_files(args.root, args.dataset)
    print("MF-EDM feature preflight (existence and nonzero size only)")
    for name, path, ok in checked:
        print(f"[{name}] {'FOUND  ' if ok else 'MISSING'}  {path}")

    if missing:
        print(f"\nMissing/empty: {len(missing)} file(s). See REPRODUCE.md for sources.")
        return 2
    print(f"\nAll {len(checked)} required file(s) are present. Schema not yet validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
