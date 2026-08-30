from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InpaintEngine(ABC):
    @abstractmethod
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint a BGR uint8 image (H, W, 3) using a uint8 mask (H, W) {0, 255}."""
