# EEG report generator

This Flask application analyzes uploaded EEG data and creates a downloadable Word report with spectral plots and topographic maps.

Use Python 3.11. The scientific dependencies retain their existing major versions and the SciPy `simps` API, while Pillow 13 requires Python 3.11 or newer.

## Install

```sh
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
```

## Run

```sh
uv run --python .venv/bin/python python app.py
```

## Verify

```sh
MPLBACKEND=Agg uv run --python .venv/bin/python python -m unittest discover -s tests -v
```

The integration test uploads deterministic synthetic EEG data, downloads the generated Word report, and checks that it contains all 15 expected plots. It verifies dependency compatibility rather than the clinical validity of the analysis.
