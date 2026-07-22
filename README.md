# PU-DCGP-COMSOL

PU-DCGP-COMSOL contains the Python workflow used to build, solve, audit, and post-process the PU-DCGP and COMSOL experiments.

## Repository status

The current repository is a research snapshot rather than a complete release package. The accessible source tree contains extensive COMSOL H11 simulation and audit scripts under `simulator_v2/phase_h`. A single verified model-training entry point, complete dataset manifest, pinned COMSOL model set, and reproducible environment lock file are not currently available. Missing large `.mph` models, generated audit files, experiment data, or local COMSOL resources must be supplied separately.

The existing `h11_*` prefixes are retained because they encode experiment phase and artifact lineage. New Python names must use lowercase `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`; commands and directories use lowercase names without spaces.

## Environment

Recommended platform:

- Windows 10 or 11
- Python 3.11
- COMSOL Multiphysics 6.3
- COMSOL Java API available to the active Python environment

Create the environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

COMSOL execution also requires the `mph` package and a valid local COMSOL installation. Deep-learning dependencies are intentionally not pinned because the repository does not yet expose a verified training implementation or hardware-specific PyTorch/CUDA target.

## Code normalization

Remove Python comments and docstrings while preserving executable strings:

```bash
python tools/clean_python.py --root .
```

Check without modifying files:

```bash
python tools/clean_python.py --root . --check
```

Format and validate:

```bash
python -m ruff format .
python -m ruff check .
python -m compileall simulator_v2 tools
```

## COMSOL execution

Each COMSOL script remains independently executable. Example:

```bash
python -m simulator_v2.phase_h.h11_conservative_free_jet_skeleton --cores 4 --version 6.3
```

A complete run must follow the contracts encoded by the selected H11 workflow and requires all referenced source models, JSON audits, and experimental summaries.

## Training

No verified model-training entry point could be established from the current repository snapshot. Do not label a COMSOL solve or audit command as neural-network training.

After the missing training module is added, use the standardized command:

```bash
python train.py --module package.path.to_training_module -- --config configs/train.yaml
```

Arguments after `--` are passed unchanged to the selected training module. Example:

```bash
python train.py --module experiments.pu_dcgp.train -- --config configs/pu_dcgp.yaml --device cuda
```

The command will fail clearly until the referenced module exists.

## Required completion items

A reproducible training release still needs:

- the final training module and configuration files
- a dataset manifest with train, validation, and test partitions
- dependency versions matched to the intended CUDA runtime
- required COMSOL `.mph` models or a documented acquisition path
- one end-to-end smoke test
- expected output locations and checkpoint naming rules
