from __future__ import annotations


class InputValidationError(Exception):
    """Invalid path, format, or size — CLI exit 1."""


class MaskError(Exception):
    """Mask missing, wrong shape/values, empty/full without flags — CLI exit 1."""


class EngineError(Exception):
    """Inpaint engine failed or is unavailable — CLI exit 2."""


class ResourceLimitError(Exception):
    """RAM/VRAM/worker cap would be exceeded — CLI exit 3."""


class ProcessingCancelled(Exception):
    """In-flight job was cancelled. Not a CLI exit mapping."""
