"""Lightweight structural smoke check.

Run: python src/common/smoke_check.py
Outputs which dependencies are installed and tests basic module imports.
Does NOT require the large HDF5 dataset.
"""
import importlib
import sys
import os
import argparse

def _check(pkg):
    try:
        importlib.import_module(pkg)
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Structural smoke check")
    parser.add_argument("--verbose", action="store_true", help="Show TensorFlow info logs")
    args = parser.parse_args()

    # Suppress TF excessive logging unless verbose
    if not args.verbose:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide INFO & WARNING

    required = ["numpy", "tensorflow", "h5py"]
    print("Dependency presence:")
    for r in required:
        print(f"  {r}: {'OK' if _check(r) else 'MISSING'}")

    # Add src to path
    SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    print("\nImport tests:")
    to_test = [
        "common.config",
        "adaptive_sampling.sampler",
        "adaptive_sampling.callbacks_confusion_snr",
        "baseline_peng.train_inceptionv3",
        "baseline_chahil.train_squeezenet",
    ]
    for mod in to_test:
        try:
            importlib.import_module(mod)
            print(f"  {mod}: OK")
        except Exception as e:
            print(f"  {mod}: FAIL ({e.__class__.__name__}: {e})")

    print("\nSmoke check complete. If core deps are MISSING, install with: pip install -r requirements.txt")
    if not args.verbose:
        print("(Run with --verbose to see TensorFlow backend initialization logs)")

if __name__ == "__main__":
    main()
