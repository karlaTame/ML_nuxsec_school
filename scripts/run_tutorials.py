"""Execute the tutorials in fresh kernels and save outputs outside the sources."""

import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="tutorial and/or tutorial_more_leptons (default: both)")
    parser.add_argument("--profile", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per cell")
    args = parser.parse_args()
    if any(name not in {"tutorial", "tutorial_more_leptons"} for name in args.notebooks):
        parser.error("Notebook names must be tutorial or tutorial_more_leptons")
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "executed" / args.profile
    out.mkdir(parents=True, exist_ok=True)
    os.environ["ML4NU_PROFILE"] = args.profile
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    for name in args.notebooks or ["tutorial", "tutorial_more_leptons"]:
        notebook = nbformat.read(root / f"{name}.ipynb", as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(notebook, kernel_name="python3", timeout=args.timeout,
                                resources={"metadata": {"path": str(root)}})
        client.create_kernel_manager().kernel_spec.argv[0] = sys.executable
        def progress(cell, cell_index, **kwargs):
            print(f"{name}: cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
        client.on_cell_start = progress
        try:
            client.execute()
        finally:
            nbformat.write(notebook, out / f"{name}.ipynb")
        print(f"Saved {out / (name + '.ipynb')}", flush=True)


if __name__ == "__main__":
    main()
