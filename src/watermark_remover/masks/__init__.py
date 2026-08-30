from __future__ import annotations

from watermark_remover.masks.auto_detect import AutoDetectMaskProvider, load_template
from watermark_remover.masks.base import MaskCandidate, MaskProvider, validate_mask_coverage
from watermark_remover.masks.manual import ManualMaskProvider
from watermark_remover.masks.serialize import (
    export_mask_json,
    export_mask_png,
    load_mask_json,
    load_mask_png,
)

__all__ = [
    "AutoDetectMaskProvider",
    "ManualMaskProvider",
    "MaskCandidate",
    "MaskProvider",
    "load_template",
    "export_mask_json",
    "export_mask_png",
    "load_mask_json",
    "load_mask_png",
    "validate_mask_coverage",
]
