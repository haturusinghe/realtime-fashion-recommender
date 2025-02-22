install-python:
	uv python install

install:
	uv venv
	. .venv/bin/activate
	uv pip install --all-extras --requirement pyproject.toml

feature-engineering:
	uv run ipython notebooks/1_fp_computing_features.ipynb

train-retrieval:
	uv run ipython notebooks/2_tp_training_retrieval_model.ipynb

train-ranking:
	uv run ipython notebooks/3_tp_training_ranking_model.ipynb

create-embeddings:
	uv run ipython notebooks/4_ip_computing_item_embeddings.ipynb