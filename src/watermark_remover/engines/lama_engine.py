from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from watermark_remover.engines.base import InpaintEngine
from watermark_remover.engines.tiling import TiledInpaint
from watermark_remover.exceptions import EngineError, MaskError
from watermark_remover.masks.base import validate_mask_array

_SEED = 0
_MODULO = 8
_TILE_SIZE = 512
_TILE_OVERLAP = 32
_INFER_DILATE = 3
_SETUP_COMMAND = "python scripts/download_models.py"


def _pin_seeds(device: str) -> None:
    np.random.seed(_SEED)
    cv2.setRNGSeed(_SEED)
    try:
        import torch

        torch.manual_seed(_SEED)
        if device == "cuda" and bool(torch.cuda.is_available()):
            torch.cuda.manual_seed_all(_SEED)
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        if hasattr(ort, "set_seed"):
            ort.set_seed(_SEED)
    except ImportError:
        pass


def _ceil_modulo(value: int, mod: int) -> int:
    remainder = value % mod
    if remainder == 0:
        return value
    return value + (mod - remainder)


class _PaddedPass(InpaintEngine):
    """Single padded ONNX pass — used by TiledInpaint to avoid process() recursion."""

    def __init__(self, engine: LaMaInpaintEngine) -> None:
        self._engine = engine

    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._engine._infer_padded(image, mask)


class LaMaInpaintEngine(InpaintEngine):
    def __init__(self, weights_path: Path, device: str) -> None:
        if device not in {"cpu", "cuda"}:
            raise EngineError(f"unknown LaMa device: {device!r}")
        path = Path(weights_path)
        if not path.is_file():
            raise EngineError(f"LaMa weights not found ({path.name}). Run: {_SETUP_COMMAND}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EngineError(
                "onnxruntime is required for LaMa. "
                "Install with: pip install 'watermark-remover[lama]'"
            ) from exc

        _pin_seeds(device)
        providers = _execution_providers(ort, device)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(
                str(path), sess_options=session_options, providers=providers
            )
        except Exception as exc:
            raise EngineError(f"failed to load LaMa weights ({path.name})") from exc

        self._device = device
        self._image_input, self._mask_input = _resolve_input_names(self._session)
        self._output_name = self._session.get_outputs()[0].name
        self._fixed_hw = _fixed_spatial_hw(self._session, self._image_input)

    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise EngineError("image must be BGR uint8 with shape (H, W, 3)")
        binary = validate_mask_array(mask)
        if binary.shape != image.shape[:2]:
            raise MaskError(f"mask shape {binary.shape} does not match image {image.shape[:2]}")
        _pin_seeds(self._device)
        height, width = image.shape[:2]
        window_h, window_w = self._window_size(height, width)
        crop = _window_around_mask(binary, window_h, window_w, height, width)
        if crop is not None:
            y0, y1, x0, x1 = crop
            filled = self._infer_padded(image[y0:y1, x0:x1], binary[y0:y1, x0:x1])
            result = image.copy()
            local = binary[y0:y1, x0:x1] > 0
            result[y0:y1, x0:x1][local] = filled[local]
            return np.ascontiguousarray(result)
        return TiledInpaint(_TILE_SIZE, _TILE_OVERLAP).process(image, binary, _PaddedPass(self))

    def _window_size(self, height: int, width: int) -> tuple[int, int]:
        if self._fixed_hw is not None:
            return self._fixed_hw
        return min(_TILE_SIZE, height), min(_TILE_SIZE, width)

    def _infer_padded(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self._pad_target(orig_h, orig_w)
        infer_mask = _dilate_mask(mask)
        padded_image, top, left = _pad_hwc(image, target_h, target_w, is_mask=False)
        padded_mask, _, _ = _pad_hwc(infer_mask, target_h, target_w, is_mask=True)
        rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_nchw = np.transpose(rgb, (2, 0, 1))[None, ...]
        mask_nchw = (padded_mask > 0).astype(np.float32)[None, None, ...]
        try:
            outputs = self._session.run(
                [self._output_name],
                {
                    self._image_input: image_nchw,
                    self._mask_input: mask_nchw,
                },
            )
        except Exception as exc:
            raise EngineError("LaMa ONNX inference failed") from exc
        bgr = _output_to_bgr(outputs[0])[top : top + orig_h, left : left + orig_w]
        result = image.copy()
        hole = mask > 0
        result[hole] = bgr[hole]
        return np.ascontiguousarray(result)

    def _pad_target(self, height: int, width: int) -> tuple[int, int]:
        if self._fixed_hw is not None:
            fixed_h, fixed_w = self._fixed_hw
            if height > fixed_h or width > fixed_w:
                raise EngineError(
                    f"LaMa tile {height}x{width} exceeds fixed model size {fixed_h}x{fixed_w}"
                )
            return fixed_h, fixed_w
        return _ceil_modulo(height, _MODULO), _ceil_modulo(width, _MODULO)


def _execution_providers(ort: object, device: str) -> list[str]:
    available = set(ort.get_available_providers())  # type: ignore[attr-defined]
    if device == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _resolve_input_names(session: object) -> tuple[str, str]:
    names = [inp.name for inp in session.get_inputs()]  # type: ignore[attr-defined]
    image_name = "image" if "image" in names else names[0]
    remaining = [name for name in names if name != image_name]
    if not remaining:
        raise EngineError("LaMa ONNX model must expose image and mask inputs")
    mask_name = "mask" if "mask" in remaining else remaining[0]
    return image_name, mask_name


def _fixed_spatial_hw(session: object, image_input: str) -> tuple[int, int] | None:
    for inp in session.get_inputs():  # type: ignore[attr-defined]
        if inp.name != image_input:
            continue
        shape = list(inp.shape)
        if len(shape) != 4:
            return None
        height, width = shape[2], shape[3]
        if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
            return int(height), int(width)
        return None
    return None


def _output_to_bgr(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise EngineError("LaMa ONNX output must have 3 channels")
    if arr.shape[0] == 3:
        chw = arr
    elif arr.shape[2] == 3:
        chw = np.transpose(arr, (2, 0, 1))
    else:
        raise EngineError("LaMa ONNX output must have 3 channels")
    scale = 1.0 if float(np.max(chw)) > 2.0 else 255.0
    rgb = np.clip(chw * scale, 0, 255).astype(np.uint8)
    hwc = np.transpose(rgb, (1, 2, 0))
    return cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    if _INFER_DILATE < 1:
        return mask
    kernel = np.ones((_INFER_DILATE, _INFER_DILATE), np.uint8)
    return cv2.dilate(mask, kernel)


def _pad_hwc(
    array: np.ndarray, target_h: int, target_w: int, *, is_mask: bool
) -> tuple[np.ndarray, int, int]:
    height, width = array.shape[:2]
    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if is_mask:
        padded = np.pad(
            array,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0,
        )
        return padded, top, left
    can_reflect = top < height and bottom < height and left < width and right < width
    mode = "symmetric" if can_reflect else "edge"
    padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode=mode)
    return padded, top, left


def _window_around_mask(
    mask: np.ndarray,
    window_h: int,
    window_w: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, min(window_h, height), 0, min(window_w, width)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_h = min(window_h, height)
    crop_w = min(window_w, width)
    if (y1 - y0) > crop_h or (x1 - x0) > crop_w:
        return None
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    gy0 = max(0, min(cy - crop_h // 2, height - crop_h))
    gx0 = max(0, min(cx - crop_w // 2, width - crop_w))
    return gy0, gy0 + crop_h, gx0, gx0 + crop_w
