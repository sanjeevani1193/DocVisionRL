# DocVisionRL
Vision-Language Models for Structured Document Extraction using Supervised Fine-Tuning and Verifiable Reinforcement Learning

This research project investigates structured extraction from receipt images using
small vision-language models. Future experiments will compare a pretrained base
model, SFT with LoRA/QLoRA, and SFT followed by GRPO with verifiable rewards.
**Milestone 1 covers environment setup and a three-example CORD v2 audit only.**
No training frameworks or models are installed, and no extraction schema is fixed.

## Environment

Verified on macOS 14.5, Apple Silicon (`arm64`), with Conda 23.7.4,
Python 3.12.16, and pip 26.2.2. The project uses a Conda environment stored in
`.venv/`; it is not a standard-library `venv`. Base Conda and Homebrew Python
are left unchanged.

To reproduce from the repository root on macOS arm64:

```sh
export CONDA_PKGS_DIRS="$PWD/.cache/conda/pkgs"
conda create --prefix "$PWD/.venv" --file conda-osx-arm64.lock --yes
conda activate "$PWD/.venv"
python -m pip install --only-binary=:all: --cache-dir .cache/pip -r requirements.lock
python -c 'import sys, platform; print(sys.version); print(sys.executable); print(platform.machine())'
python -m pip --version
python -m pip check
```

For the environment already created here, skip creation and installation:

```sh
conda activate "$PWD/.venv"
```

If `conda activate` says the shell is not initialized, run
`source "$(conda info --base)/etc/profile.d/conda.sh"` in that terminal, then
activate again. This initializes only the current shell.

`conda-osx-arm64.lock` records exact Conda package URLs/builds, including Python,
pip, and native runtime libraries. `requirements.txt` lists the two direct
dependencies; `requirements.lock` pins their complete installed Python dependency
set (plus Conda-provided `packaging`). Install the lock to reproduce this audit.
These are version/build pins, not a hash-verified wheel archive, and require the
package sources to remain available. The Conda lock is specific to macOS arm64;
other platforms need a separately resolved environment. `--only-binary=:all:`
fails if a wheel is unavailable instead of starting a source build.

VS Code's workspace settings point to `${workspaceFolder}/.venv/bin/python`.
If an interpreter was previously selected, use **Python: Select Interpreter** in
the Command Palette and choose `.venv/bin/python`, then open a new terminal.
Check `python -m pip --version` there: its path should be inside this repository's
`.venv`. An explicit `.venv/bin/python` also works without activation and avoids
the shell's existing Homebrew `python3` selection.

## Inspect CORD v2

```sh
.venv/bin/python scripts/inspect_cord.py
```

The script uses a pinned dataset commit, prints the published metadata and actual
features, and inspects exactly the first three training examples in stored order.
It needs internet access. Hugging Face caches default to `.cache/huggingface/`
unless `HF_HOME` is already set. Data, caches, and the environment are ignored by
Git; `data/.gitkeep` preserves the otherwise empty data directory.

For an optional local transcript (also ignored):

```sh
set -o pipefail
.venv/bin/python -u scripts/inspect_cord.py 2>&1 | tee data/cord_inspection.txt
```

How the script works:

- `HfApi.dataset_info` resolves the pinned commit and reads card metadata;
  `hf_hub_download` reads only the small published `dataset_infos.json` file.
- `load_dataset(..., split="train", revision=..., streaming=True)` returns an
  iterable instead of preparing a complete local dataset. `take(3)` bounds row
  iteration. Parquet reads may transfer extra blocks or rows; three examples is
  not a byte-level network limit. There is no non-streaming fallback.
- `Image(decode=False)` exposes stored image bytes. Pillow decodes each image
  separately and `image.load()` verifies its pixels, allowing the annotation to
  be inspected even if image decoding fails. Dimensions are original stored
  width and height; no resize or other preprocessing is applied.
- `json.loads` turns `ground_truth` into Python dictionaries/lists. The script
  reports root fields, metadata, and a bounded pretty-printed `gt_parse` preview.
  Recursive traversal discovers keys rather than assuming receipt field names;
  `[]` in a printed path means a list element. Type counts count values, while
  row presence counts each path once per receipt.
- Missing fields, invalid JSON, unexpected containers, image failures, and image
  dimension mismatches produce explicit warnings. Available rows continue to be
  inspected; an incomplete three-row audit exits with an error. Network or
  streaming failures retain their traceback and never trigger a full download.

See [docs/DATA_AUDIT.md](docs/DATA_AUDIT.md) for findings, provenance, and limits.
Three consecutive examples are a smoke test, not a representative dataset audit.
