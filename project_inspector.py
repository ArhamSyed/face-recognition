"""
Project inspector utility

- Lists Python files in the repository
- Attempts to import `face_utils` and `face_dataset` and reports whether import succeeded

This script is intentionally conservative: imports are wrapped in try/except so it won't crash if optional dependencies (like cv2) are missing.
"""

from pathlib import Path
import importlib
import sys


def list_py_files(base_path):
    p = Path(base_path)
    return sorted([str(x.name) for x in p.glob('*.py')])


def try_import(module_name):
    try:
        module = importlib.import_module(module_name)
        return True, module
    except Exception as e:
        return False, e


def main():
    base = Path(__file__).parent
    print(f"Project directory: {base}")

    py_files = list_py_files(base)
    print("Python files in project:")
    for f in py_files:
        print(" -", f)

    modules_to_check = ['face_utils', 'face_dataset']
    print("\nImport checks:")

    for mod in modules_to_check:
        ok, result = try_import(mod)
        if ok:
            print(f" - {mod}: OK (module: {getattr(result, '__name__', str(result))})")
        else:
            print(f" - {mod}: FAIL -> {result}")

    print("\nDone.")


if __name__ == '__main__':
    main()

