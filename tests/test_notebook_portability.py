"""Check portable notebook metadata and the actual Colab checkout paths."""

import json
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile

import pytest
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("tutorial.ipynb", "tutorial_more_leptons.ipynb")


def _bootstrap():
    notebook = json.loads((ROOT / NOTEBOOKS[0]).read_text())
    cell = next(c for c in notebook["cells"] if c.get("id") == "portable-setup")
    namespace = {}
    exec("".join(cell["source"]), namespace)
    return namespace["prepare_colab"]


def test_notebooks_ship_without_outputs_or_machine_specific_kernels():
    bootstraps = []
    for name in NOTEBOOKS:
        notebook = json.loads((ROOT / name).read_text())
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert not cell["outputs"]
        bootstraps.append(next(c["source"] for c in notebook["cells"]
                               if c.get("id") == "portable-setup"))
    assert bootstraps[0] == bootstraps[1]


def test_colab_extra_preserves_the_hosted_kernel_stack():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = {req.name: req for req in map(Requirement, project["optional-dependencies"]["colab"])}
    # google-colab currently pins IPython 7.34.0. Requiring IPython 8+ here
    # would replace a component of the running hosted kernel.
    assert extras["ipython"].specifier.contains("7.34.0")
    assert not {"ipykernel", "jupyterlab", "jupyter-server"}.intersection(extras)


@pytest.fixture
def portable_setup(monkeypatch):
    prepare = _bootstrap()
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "path", list(sys.path))
    real_run = subprocess.run
    installs = []

    def run(args, **kwargs):
        if args[1:4] == ["-m", "pip", "install"]:
            installs.append(args)
            return subprocess.CompletedProcess(args, 0)
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    return prepare, installs


def _minimal_repository(path):
    (path / "src" / "ml4nu").mkdir(parents=True)
    (path / "src" / "ml4nu" / "__init__.py").write_text('VALUE = "from checkout"\n')
    (path / "pyproject.toml").write_text('[project]\nname = "ml4nu"\nversion = "0.1.0"\n')


def test_colab_can_extract_a_github_style_private_repository_zip(tmp_path, portable_setup):
    prepare, installs = portable_setup
    archive = tmp_path / "private.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ML_nuxsec_school-main/pyproject.toml", '[project]\nname="ml4nu"\n')
        bundle.writestr("ML_nuxsec_school-main/src/ml4nu/__init__.py", "VALUE = 42\n")
        bundle.writestr("ML_nuxsec_school-main/data/input.txt", "bundled input\n")
    target = tmp_path / "colab"
    prepare("unused", "unused", target, source="upload", archive=archive)
    assert (target / "data" / "input.txt").read_text() == "bundled input\n"
    assert Path.cwd() == target
    assert sys.path[0] == str(target / "src")
    assert installs[0][-1] == str(target) + "[colab]"


def test_colab_clones_requested_branch_and_preserves_rerun_edits(tmp_path, portable_setup):
    prepare, installs = portable_setup
    origin = tmp_path / "origin"
    _minimal_repository(origin)
    subprocess.run(["git", "init", "-b", "main", str(origin)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(origin), "add", "."], check=True)
    subprocess.run(["git", "-C", str(origin), "-c", "user.name=Test",
                    "-c", "user.email=test@local.invalid", "commit", "-m", "Fixture"],
                   check=True, capture_output=True)
    target = tmp_path / "colab"
    prepare(origin.as_uri(), "main", target)
    marker = target / "local_results.txt"
    marker.write_text("keep my runtime results")
    prepare("unused on rerun", "main", target)
    assert marker.read_text() == "keep my runtime results"
    assert len(installs) == 2


def test_colab_does_not_overwrite_an_unrelated_folder(tmp_path, portable_setup):
    prepare, _ = portable_setup
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "user_file.txt"
    marker.write_text("keep")
    with pytest.raises(RuntimeError, match="choose another folder"):
        prepare("unused", "unused", target)
    assert marker.read_text() == "keep"
