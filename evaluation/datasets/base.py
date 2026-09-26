"""Base dataset loader interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from evaluation.models import BenchmarkSample


class BaseDatasetLoader(ABC):
    """Abstract interface for dataset loaders."""

    @abstractmethod
    def load(self, limit: Optional[int] = None) -> list[BenchmarkSample]:
        """Load benchmark samples from dataset.

        Parameters
        ----------
        limit:
            Maximum number of samples to load (useful for development/testing).

        Returns
        -------
        list[BenchmarkSample]
            Loaded benchmark samples with text and ground-truth triplets.
        """
        pass

