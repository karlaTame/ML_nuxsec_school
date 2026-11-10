# Machine learning in neutrino cross sections: theory

School tutorials for learning neutrino–nucleus cross sections, with their physics library and required input data.
Both run on **CPU** with Python **3.11 or 3.12**, locally or in Google Colab.
No GPU, LHAPDF, pre-trained weights, or pre-generated event samples are required.

| Tutorial | Paper | Google Colab |
|---|---|---|
| [Muon-neutrino disappearance](tutorial.ipynb) | [2412.16303](https://arxiv.org/abs/2412.16303) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karlaTame/ML_nuxsec_school/blob/main/tutorial.ipynb) |
| [More Leptons](tutorial_more_leptons.ipynb) | [2610.30338](https://arxiv.org/abs/2610.30338) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karlaTame/ML_nuxsec_school/blob/main/tutorial_more_leptons.ipynb) |

## Local CPU

Clone the repository and create an environment:

```bash
git clone --depth 1 --branch main https://github.com/karlaTame/ML_nuxsec_school.git
cd ML_nuxsec_school
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead. Install CPU
PyTorch on Linux/Windows:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On macOS use `python -m pip install torch` instead. Then install and open Jupyter:

```bash
python -m pip install -e '.[notebook]'
python -m jupyter lab
```

Open either notebook and select **Restart Kernel and Run All**. The first cell
locates the checkout; CPU selection is explicit even on a machine with a GPU.
Source notebooks use a standard Python kernel, so no particular Conda environment
is needed.

For a short check from fresh kernels, without opening Jupyter:

```bash
python scripts/run_tutorials.py --profile smoke
python scripts/run_tutorials.py --profile quick
```

`smoke` checks execution with small budgets. `quick` teaches the full nominal
workflow: 5,000 training steps in Tutorial 1 and 2,000 in More Leptons. Allow
several minutes on CPU. `full` is much more expensive and is intended for
convergence studies; a standard free Colab runtime may not have sufficient
resources for its larger settings. Tutorial plots alone do not establish
quantitative reproduction of a paper. See the [tutorial guide](docs/tutorials.md)
and [first-paper reproduction audit](docs/tutorial1-reproduction.md).

## Google Colab

1. Open a notebook with its Colab button above and choose a Python runtime with
   **no accelerator**.
2. Run the first setup cell. It clones `main`, installs `.[colab]`, and
   makes the supplied data available in the running kernel.
3. Choose `PROFILE="smoke"` for a small check or `"quick"` for the tutorial,
   then run the remaining cells.

The Colab buttons use the public `main` branch. Change `REPO_URL` and
`REPO_BRANCH` in the setup cell when using a fork or another branch.

For a **private repository**, use the ZIP path without putting credentials in
notebook code:

1. Download the ZIP of this branch from GitHub, or export a local committed
   checkout with `git archive --format=zip --prefix=ML_nuxsec_school/ --output=../ML_nuxsec_school-colab.zip HEAD`.
2. In Colab, use **File → Upload notebook** to open one of the local tutorial
   notebooks. Set `COLAB_SOURCE="upload"` in its first setup cell.
3. Run that cell and select the repository ZIP when prompted. Upload the whole
   repository ZIP, since the notebook needs the library and data as well.

The ZIP must contain the current tutorial branch, including `data/` and `src/`.
Only notebook dependencies are installed in Colab; a Jupyter server is unnecessary.
Download any generated results you want to keep before the runtime disconnects.
Colab's available resources and session duration are variable; see its
[official FAQ](https://research.google.com/colaboratory/faq.html).
