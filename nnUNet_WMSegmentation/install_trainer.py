#!/usr/bin/env python3
"""
install_trainer.py
==================
Copy ``custom_trainer.py`` (containing ``nnUNetTrainerHistoAug``) into the
nnU-Net v2 trainer variants directory so that the ``-tr`` flag can discover it.

Usage
-----
    python install_trainer.py          # auto-detect nnunetv2 location
    python install_trainer.py --check  # verify installation without copying
"""

import argparse
import importlib
import os
import shutil
import sys
import importlib.util

def find_nnunet_trainers_dir() -> str:
    """Return the variants/data_augmentation directory inside nnunetv2."""
    spec = importlib.util.find_spec("nnunetv2")
    if spec is None or spec.origin is None:
        raise ImportError(
            "nnunetv2 is not installed. Install it first:\n"
            "  pip install nnunetv2"
        )
    pkg_dir = os.path.dirname(spec.origin)
    target = os.path.join(
        pkg_dir,
        "training",
        "nnUNetTrainer",
        "variants",
        "data_augmentation",
    )
    if not os.path.isdir(target):
        raise FileNotFoundError(
            f"Expected directory not found: {target}\n"
            "Your nnunetv2 installation may have a different layout."
        )
    return target


def install(check_only: bool = False):
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_trainer.py")
    if not os.path.isfile(src):
        print(f"ERROR: Source file not found: {src}")
        sys.exit(1)

    target_dir = find_nnunet_trainers_dir()
    dest = os.path.join(target_dir, "nnUNetTrainerHistoAug.py")

    if check_only:
        if os.path.isfile(dest):
            print(f"OK: Custom trainer is installed at {dest}")
        else:
            print(f"NOT INSTALLED: Expected file at {dest}")
        return

    shutil.copy2(src, dest)
    print(f"Installed custom trainer:")
    print(f"  Source      : {src}")
    print(f"  Destination : {dest}")
    print()
    print("You can now train with:  nnUNetv2_train DATASET 2d FOLD -tr nnUNetTrainerHistoAug")


def main():
    parser = argparse.ArgumentParser(description="Install nnUNetTrainerHistoAug into nnunetv2.")
    parser.add_argument("--check", action="store_true", help="Only check if already installed.")
    args = parser.parse_args()
    install(check_only=args.check)


if __name__ == "__main__":
    main()
