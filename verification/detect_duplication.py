#!/usr/bin/env python3
"""
Detect train/test case duplication in a BreaDM release.

The BreaDM archive is distributed as eight derived trees. In two of them
(cls/img9Se and seg) the test folder contains byte-identical copies of cases
that also appear in the training folder. This script checks every tree in a
given copy of the dataset and reports any overlap.

Usage:
    python detect_duplication.py --root /path/to/BreaDM
    python detect_duplication.py --root /path/to/BreaDM --hash   # also verify byte-identity

Expected output on the archive as distributed (sha256 64d8e621...97ce6e5):

    cls/img9Se     train 166  test  53   OVERLAP 6    <- defect
    cls/img17Se    train 166  test  47   clean
    seg            train 166  test  50   OVERLAP 3    <- defect
    seg3D          train 166  test  47   clean
    ...

Reference:
    Replication and Reproducibility Analysis of Breast Tumor Classification and
    Segmentation Benchmarks on the BreastDM Dataset. MICCAI 2026 / MSB EMERGE.
"""

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict

CASE_RE = re.compile(r"BreaDM-(?:Be|Ma)-\d+")
SPLITS = ("train", "val", "test")


def find_trees(root):
    """Return {tree_name: path} for every subtree that has train/ and test/ folders."""
    trees = {}
    for dirpath, dirnames, _ in os.walk(root):
        names = set(dirnames)
        if "train" in names and "test" in names:
            rel = os.path.relpath(dirpath, root)
            trees[rel if rel != "." else os.path.basename(root)] = dirpath
            dirnames[:] = [d for d in dirnames if d in SPLITS]
    return dict(sorted(trees.items()))


def cases_in(split_dir):
    """Map case id -> list of file paths, for every file under a split directory."""
    out = defaultdict(list)
    for dirpath, _, filenames in os.walk(split_dir):
        m = CASE_RE.search(dirpath)
        if not m:
            continue
        case = m.group(0)
        for fn in filenames:
            out[case].append(os.path.join(dirpath, fn))
    return out


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def identical(files_a, files_b):
    """True if the two case folders hold the same filenames with the same contents."""
    by_name_a = {os.path.basename(p): p for p in files_a}
    by_name_b = {os.path.basename(p): p for p in files_b}
    if set(by_name_a) != set(by_name_b):
        return False
    return all(sha256(by_name_a[n]) == sha256(by_name_b[n]) for n in by_name_a)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="path to the extracted BreaDM directory")
    ap.add_argument("--hash", action="store_true",
                    help="verify that overlapping cases are byte-identical (slower)")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"not a directory: {args.root}")

    trees = find_trees(args.root)
    if not trees:
        sys.exit(f"no train/test subtrees found under {args.root}")

    print(f"root: {os.path.abspath(args.root)}")
    print(f"found {len(trees)} derived tree(s)\n")
    print(f"{'tree':<28}{'train':>7}{'val':>6}{'test':>6}{'overlap':>9}   status")
    print("-" * 74)

    defects = []
    for name, path in trees.items():
        counts, cases = {}, {}
        for sp in SPLITS:
            d = os.path.join(path, sp)
            cases[sp] = cases_in(d) if os.path.isdir(d) else {}
            counts[sp] = len(cases[sp])

        overlap = sorted(set(cases["train"]) & set(cases["test"]))
        status = "clean" if not overlap else "DUPLICATED"
        print(f"{name:<28}{counts['train']:>7}{counts['val']:>6}{counts['test']:>6}"
              f"{len(overlap):>9}   {status}")

        if overlap:
            defects.append((name, path, overlap, cases))

    if not defects:
        print("\nNo train/test case overlap found. This copy is clean.")
        return 0

    print("\n" + "=" * 74)
    for name, path, overlap, cases in defects:
        n_files = sum(len(cases["test"][c]) for c in overlap)
        total = sum(len(v) for v in cases["test"].values())
        pct = 100.0 * n_files / total if total else 0.0
        print(f"\n{name}")
        print(f"  {len(overlap)} case(s) appear in BOTH train and test:")
        for c in overlap:
            print(f"    {c}")
        print(f"  affecting {n_files} of {total} test files ({pct:.1f}%)")

        train_sorted = sorted(cases["train"])
        if overlap == train_sorted[:len(overlap)]:
            print(f"  note: these are exactly the first {len(overlap)} entries of the "
                  f"sorted training list")

        print(f"  test cases after removing them: "
              f"{len(cases['test']) - len(overlap)}")

        if args.hash:
            print("  verifying byte-identity...")
            for c in overlap:
                same = identical(cases["train"][c], cases["test"][c])
                print(f"    {c}: {'byte-identical' if same else 'DIFFERENT CONTENT'}")

    print("\n" + "=" * 74)
    print("To reproduce the split described in the original paper, remove the "
          "duplicated cases\nfrom the test folders of the affected trees. Build the "
          "de-duplicated copy with HARDLINKS,\nnot symlinks — the segmentation loader "
          "uses os.walk without followlinks and will read\na symlink tree as empty:")
    print("\n    cp -al BreaDM/seg BreaDM/seg_clean && rm -rf BreaDM/seg_clean/test/<case>\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
