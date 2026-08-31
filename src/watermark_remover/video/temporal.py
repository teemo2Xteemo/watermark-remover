from __future__ import annotations

import cv2
import numpy as np

from watermark_remover.config import Settings, get_settings
from watermark_remover.exceptions import EngineError

_FARNEBACK_PYR_SCALE = 0.5
_FARNEBACK_LEVELS = 3
_FARNEBACK_WINSIZE = 15
_FARNEBACK_ITERATIONS = 3
_FARNEBACK_POLY_N = 5
_FARNEBACK_POLY_SIGMA = 1.2
_BLEND_PREV_WEIGHT = 0.5
_RAFT_ALIGN = 8


class TemporalSmoother:
    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings if settings is not None else get_settings()
        self._raft_enabled = bool(cfg.raft_enabled)

    def apply(
        self, prev: np.ndarray, current: np.ndarray, inpaint_mask: np.ndarray
    ) -> np.ndarray:
        """Smooth `current` toward flow-aligned `prev` inside `inpaint_mask` only.

        prev, current: BGR uint8 (H, W, 3). inpaint_mask: uint8 (H, W), {0, 255}.
        Pixels where the mask is 0 are copied from `current` unchanged.
        """
        _validate_frames(prev, current, inpaint_mask)
        if not np.any(inpaint_mask):
            return current

        if self._raft_enabled:
            warped_prev = self._warp_prev_raft(prev, current)
        else:
            warped_prev = _warp_prev_farneback(prev, current)
        return _composite_inpaint_region(current, warped_prev, inpaint_mask)

    def _warp_prev_raft(self, prev: np.ndarray, current: np.ndarray) -> np.ndarray:
        try:
            import torch
            import torch.nn.functional as F
            from torchvision.models.optical_flow import raft_small  # type: ignore[import-not-found]
        except ImportError as exc:
            raise EngineError(
                "RAFT optical flow requested (raft_enabled=True) but torchvision is not installed"
            ) from exc

        device = torch.device("cpu")
        model = raft_small(weights=None)
        model.to(device)
        model.eval()

        def to_tensor(image: np.ndarray) -> torch.Tensor:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
            return tensor.unsqueeze(0).to(device)

        img_current = to_tensor(current)
        img_prev = to_tensor(prev)
        orig_h, orig_w = int(img_current.shape[2]), int(img_current.shape[3])
        pad_h = (_RAFT_ALIGN - orig_h % _RAFT_ALIGN) % _RAFT_ALIGN
        pad_w = (_RAFT_ALIGN - orig_w % _RAFT_ALIGN) % _RAFT_ALIGN
        if pad_h or pad_w:
            img_current = F.pad(img_current, (0, pad_w, 0, pad_h))
            img_prev = F.pad(img_prev, (0, pad_w, 0, pad_h))
        try:
            with torch.inference_mode():
                flows = model(img_current, img_prev)
        except Exception as exc:
            raise EngineError("RAFT optical flow failed") from exc
        flow = flows[-1][0, :, :orig_h, :orig_w].permute(1, 2, 0).detach().cpu().numpy()
        return _remap_with_flow(prev, np.ascontiguousarray(flow, dtype=np.float32))


def _validate_frames(
    prev: np.ndarray, current: np.ndarray, inpaint_mask: np.ndarray
) -> None:
    if prev.shape != current.shape or prev.dtype != np.uint8 or current.dtype != np.uint8:
        raise EngineError("temporal smoother requires matching BGR uint8 prev/current frames")
    if current.ndim != 3 or current.shape[2] != 3:
        raise EngineError("temporal smoother requires BGR uint8 frames with shape (H, W, 3)")
    if inpaint_mask.shape != current.shape[:2] or inpaint_mask.dtype != np.uint8:
        raise EngineError("temporal smoother requires uint8 inpaint_mask with the same HxW")


def _warp_prev_farneback(prev: np.ndarray, current: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        prev_gray,
        None,
        _FARNEBACK_PYR_SCALE,
        _FARNEBACK_LEVELS,
        _FARNEBACK_WINSIZE,
        _FARNEBACK_ITERATIONS,
        _FARNEBACK_POLY_N,
        _FARNEBACK_POLY_SIGMA,
        0,
    )
    return _remap_with_flow(prev, flow)


def _remap_with_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = np.ascontiguousarray(grid_x + flow[:, :, 0])
    map_y = np.ascontiguousarray(grid_y + flow[:, :, 1])
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _composite_inpaint_region(
    current: np.ndarray, warped_prev: np.ndarray, inpaint_mask: np.ndarray
) -> np.ndarray:
    blended = cv2.addWeighted(
        warped_prev, _BLEND_PREV_WEIGHT, current, 1.0 - _BLEND_PREV_WEIGHT, 0.0
    )
    out = current.copy()
    out[inpaint_mask != 0] = blended[inpaint_mask != 0]
    return out
