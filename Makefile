install-python:
	uv python install

install:
	uv venv
	. .venv/bin/activate
	uv pip install --all-extras --requirement pyproject.toml

feature-engineering:
	uv run ipython notebooks/1_fp_computing_features.ipynb