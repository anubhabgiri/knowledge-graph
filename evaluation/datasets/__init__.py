"""Dataset loaders for benchmark evaluation."""
from evaluation.datasets.base import BaseDatasetLoader
from evaluation.datasets.bc5cdr import BC5CDRDatasetLoader
from evaluation.datasets.webnlg import WebNLGDatasetLoader

__all__ = ["BaseDatasetLoader", "BC5CDRDatasetLoader", "WebNLGDatasetLoader"]

