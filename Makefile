data: data/external/train-robust04.parquet

data/external/train-robust04.parquet:
	uv run topic-judges data robust04 --output=data/external/train-robust04.parquet
