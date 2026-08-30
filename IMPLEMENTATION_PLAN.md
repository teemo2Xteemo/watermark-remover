# Implementation Plan — Local Watermark / Object-Removal Tool

Source of truth for this document:

- Six `.cursor/rules/*.mdc` files (read in full before drafting).
- Stitch MCP fetch for project **Offline Inpaint Studio** (`14439947336896175162`), executed this session.

No code is written in this step.

---

## 0. Fetched UI Spec Summary

**Fetch method (this session):** Cursor’s native Stitch MCP tool catalog did not register (`toolCount: 0`; schema error `can't resolve reference #/$defs/ScreenInstance`). Using the API key already in `.cursor/mcp.json`, the same official MCP endpoint `https://stitch.googleapis.com/mcp` was called via JSON-RPC `tools/call` (stateless; no inventing screens).

**MCP tools that returned data:**

| Tool | Arguments | Result |
|------|-----------|--------|
| `tools/list` | `{}` | 15 tools available |
| `list_projects` | `{}` | Project present |
| `get_project` | `{name: "projects/14439947336896175162"}` | Title **Offline Inpaint Studio** |
| `list_screens` | `{projectId: "14439947336896175162"}` | **3 screens** (DESKTOP, 2560×2048) |
| `get_screen` | `name` + `projectId` + `screenId` for each of the 3 IDs below | Title, HTML, screenshot URLs |
| HTML + PNG download | `htmlCode.downloadUrl`, `screenshot.downloadUrl` | Saved locally and read |

**Project title from `get_project`:** Offline Inpaint Studio  
**Design theme from `list_projects` / `get_project`:** Carbon — Enterprise System (IBM Plex Sans, primary `#0f62fe`, light `colorMode` in theme payload; rendered screens are dark UI).

**Note:** Sending `X-Goog-User-Project: 14439947336896175162` (as currently set in `.cursor/mcp.json`) made `get_project` return “Project not found or deleted”. The same call **without** that header succeeded. That header is a GCP billing project, not a Stitch design-project id.

### Screens fetched (confirm this set)

Three screens. There is **no fourth screen**. There is **no populated Image Mode workspace** in this project — only Empty State (shared chrome + dropzone) and Video Mode (full workspace).

---

### Screen 1 — `Watermark Remover - Video Mode`

- **Stitch screen ID:** `466bc67d529a4f938f3aa32c244a27a7`
- **Resource:** `projects/14439947336896175162/screens/466bc67d529a4f938f3aa32c244a27a7`
- **Device:** DESKTOP 2560×2048
- **HTML comment sections:** `TopNavBar` → left `aside` (`Section A: Auto-Detect`, `Section B: Model Settings`, `Section C: Video Settings`, `Sticky Bottom Button`) → `Main Content Area` (canvas) → `Bottom Panel (Timeline)`

**Layout (left → center → bottom):**

1. **TopNavBar**
   - Product label: `Carbon Eraser Pro`
   - Nav links: `File`, `Edit`, `View`, `Export`, `Help`
   - Mode toggle: button `Image Mode` | button `Video Mode` (Video selected)
   - Icon buttons: `settings`, `notifications`
2. **Left sidebar — Section A: Detection Mode** (`<h3>Detection Mode</h3>`)
   - Toggle buttons: `Auto` | `Manual`
   - Dropzone copy: `Upload watermark template (PNG/JPG)`
   - Range slider label: `Sensitivity` (rendered value `50%`; `min=1` `max=100`)
   - Button: `Run Detection`
   - Candidate row: `Candidate 1` + confidence badge `87%` + reject icon button `close` (green `check_circle` on the row)
3. **Left sidebar — Section B: Model Selection** (`<h3>Model Selection</h3>`)
   - `<select>` options (exact strings):
     - `LaMa (High Quality)`
     - `Stable Diffusion Inpainting`
     - `Fast Inpaint (ProPainter)`
   - Helper text: `Best for large areas, GPU recommended`
   - Accordion: `Advanced Settings`
     - Row: `Use GPU (CUDA)` + status `Active`
4. **Left sidebar — Section C: Video Settings** (`<h3>Video Settings</h3>`)
   - Toggle: `Apply Temporal Smoothing`
   - Label + `<select>`: `Output Quality` → `Same as Source` | `1080p` | `720p`
   - Checkbox: `Keep Original Audio`
   - Label + number input: `Process Nth frame` (rendered `1`)
5. **Sticky Bottom Button**
   - Primary: `Apply Inpainting`
6. **Canvas / Preview**
   - Overlay label: `MASK 1`
   - Hover play: `play_arrow`
   - View controls (icon-only): `visibility`, `compare`, `zoom_in`
7. **Bottom Panel (Timeline)**
   - Transport: `skip_previous`, `fast_rewind`, `play_arrow`, `fast_forward`, `skip_next`
   - Timecode (rendered): `00:12:14 / 01:34:00`
   - Button: `Add Mask Keyframe`
   - Time ruler: `00:00` … `00:50`
   - Range overlay tooltip: `Mask Active`

---

### Screen 2 — `Watermark Remover - Empty State`

- **Stitch screen ID:** `55cfe8fc80104558be78c8dfcf378fbb`
- **Resource:** `projects/14439947336896175162/screens/55cfe8fc80104558be78c8dfcf378fbb`
- **Device:** DESKTOP 2560×2048
- **HTML comment sections:** `TopNavBar`, `SideNavBar`, `Main Canvas - Empty State`, `Canvas Dropzone`, `Status Bar`

**Layout:**

1. **TopNavBar**
   - Product label: `Carbon Eraser Pro`
   - Nav: `File` (active), `Edit`, `View`, `Export`, `Help`
   - Icon buttons: `settings` (`aria-label="settings"`), `notifications` (`aria-label="notifications"`)
   - `User profile` avatar (`<img alt="User profile">`)
   - **No** `Image Mode` / `Video Mode` toggle on this screen
2. **SideNavBar** (`Project Workspace`, version copy `V1.0.4-Alpha`)
   - Nav items (exact labels): `Mask Tools` (active), `Engine Settings`, `Object List`, `History`
   - Disabled primary: `Process All` + helper `Requires input media`
   - Footer links: `Shortcuts`, `Support`
3. **Canvas toolbar** (disabled / `pointer-events-none`): `undo`, `redo`, `zoom_in`, `zoom_out`
4. **Canvas Dropzone**
   - Heading: `Drag and drop a file here`
   - Body: `Supports MP4, MOV, PNG, and JPG formats. Maximum file size for unauthenticated sessions is 500MB.`
   - Button: `Open File`
5. **Status Bar**
   - `Status: Waiting for input`
   - `0 MB / 0 MB`

---

### Screen 3 — `Watermark Remover - Processing`

- **Stitch screen ID:** `65ac88f4a7e741ba96a96d761ddb5f6f`
- **Resource:** `projects/14439947336896175162/screens/65ac88f4a7e741ba96a96d761ddb5f6f`
- **Device:** DESKTOP 2560×2048
- **HTML comment sections:** `Top Navigation Bar`, `Side Navigation Bar`, blurred `Video Processing Dashboard`, `Processing Modal`

**Layout:**

1. **Top Navigation Bar** — same chrome: `Carbon Eraser Pro`, `File` `Edit` `View` `Export` `Help`, `settings`, `notifications`, `User profile`
2. **Side Navigation Bar** — `Project Workspace` / `V1.0.4-Alpha`; nav `Mask Tools`, `Engine Settings` (active), `Object List`, `History`; disabled `Process All`; `Shortcuts`, `Support`
3. **Blurred background dashboard** (not the 5-section rule layout)
   - Title: `Video Processing Dashboard`
   - Subtitle: `Engine status: Active | Source: local_drive_01`
   - Card `Preview Render`
   - Card `Detection Metrics`: `Confidence Score` (rendered `98.4%`), `Mask Complexity` (rendered `High`)
   - Card `Frame Analysis Timeline`
4. **Processing Modal**
   - Title: `Watermark Removal Engine`
   - Status line: `Processing frame 120/540...`
   - `Task ID: WM-R-089A4` (HTML; screenshot description may differ)
   - Percent: `22%`
   - ETA: `00:04 remaining`
   - Progress bar
   - Log panel lines (exact prefixes):
     - `>[SYS] Initiating spatial analysis... OK`
     - `>[SYS] Generating temporal masks for segment A... OK`
     - `>[ENG] Applying inpainting algorithm v2.4...`
     - `>[ENG] Frame 118 computed in 45ms.`
     - `>[ENG] Frame 119 computed in 42ms.`
     - `>[ENG] Analyzing frame 120...`
   - Button: `Cancel`
5. **Footer**
   - `© 2024 Carbon Enterprise Systems. All rights reserved.`
   - Links: `Privacy Policy`, `Terms of Service`, `API Documentation`

---

## 0.5. UI Spec vs Rule Conflicts — LOCKED

All C1–C16 decided. Implement these; do not re-open unless the user revisits.

The historical A/B table is replaced by the decision table below. Fetched Stitch labels remain in section 0 as visual reference only.

| # | Decision | What to implement |
|---|----------|-------------------|
| C1 | **B** | Five gr.Blocks sections in rule order: **Input → Mask → Preview → Engine → Run**. Stitch is visual/IA reference only (nav labels, left-column spirit). Stitch does not set information architecture. |
| C2 | **B** | Full Image 5-section flow in M2 even without a Stitch image-loaded frame. Derive Image controls from Video Mode sidebar (Detection Mode → Model Selection → Video Settings → Apply) and Empty State, mapped onto the five sections — do not invent a third IA. |
| C3 | **B** | JPG/PNG/WEBP and MP4/MOV/WEBM. Limit = max_input_bytes (2 GiB, Q3). No unauthenticated-sessions and no 500MB copy. |
| C4 | **B** | Engine dropdown: opencv / lama / auto only. No Stable Diffusion, no ProPainter. Extra engines = a future milestone + registry review. |
| C5 | **B** | Run enabled only after confirmed mask + mandatory preview overlay. Button labels may stay Stitch-flavored: Image Process All, Video Apply Inpainting — enable/disable still follows the rule (not has-media). C15: do not treat Process All as batch. |
| C6 | **B** | Explicit Accept and Reject on each candidate. Run stays off until accept/edit. Do not treat Stitch check_circle as implicit accept. |
| C7 | **B** | gr.ImageEditor (bbox + freehand, Q18). Video: toggle static mask (all frames) vs keyframe masks at timestamps. May keep Stitch MASK 1 overlay + Add Mask Keyframe as keyframe display, plus the rule toggle. |
| C8 | **B** | Explicit import/export {stem}.mask.png and {stem}.mask.json (schema_version 1, PNG is pixel source of truth). Generic nav Export alone is not enough. |
| C9 | **B** | No History on disk (Q13). Optional in-session undo via gr.State only. |
| C10 | **B** | Omit avatar, notifications, API docs, Privacy Policy, Terms, Support, Shortcuts (Q13). Local-only chrome. |
| C11 | **B** | Visible product name watermark-remover (Q11). No Carbon Eraser Pro / Carbon Enterprise Systems. |
| C12 | **B** | Visible LaMa CPU warning when CUDA is unavailable. Not Active-only. |
| C13 | **hybrid** | UI: Process Nth frame; Output Quality (Same as Source / 1080p / 720p) config-gated (maps to resolution cap derived from max_ram_mb/max_vram_mb). Apply Temporal Smoothing on/off; algorithm = Farneback default, RAFT only via hidden advanced/config flag. Keep Original Audio checkbox. No extra drop-audio control. |
| C14 | **B** (+ layout note) | Data: job_id = real uuid4 (never WM-R-089A4); log panel reads the same structlog/callback stream (job_id, engine, frame_idx, duration_ms) — Q20. Cancel = interrupt + temp cleanup. Layout: Stitch-like modal / progress bar / percent allowed as Gradio presentation. |
| C15 | **B** | No Process All batch control. Folder batch = CLI batch.py only (Q19). Single-file CTA uses C5 labels for one job. |
| C16 | **B** | Native Gradio Row/Column/Accordion; minimal custom CSS. Optional cheap Gradio theme tint. Do not rebuild Stitch pixel-perfect CSS. |

**Resolved vs other rules:** SD/ProPainter out (C4). No API/legal/auth chrome (C10). No fake unauthenticated/500MB copy (C3).

---

## 1. Rule Compliance Summary

### `00-core-architecture.mdc`

Local-only inpaint; watermark is just a mask type. Non-negotiables: no cloud AI in the pipeline, no telemetry, no C2PA/ownership checks, no platform-specific watermark brands, no steganographic marks, auto-detect returns `MaskCandidate`s only, never overwrite input without an explicit flag, fix RNG seeds. Directory tree and plugin ABCs (`MaskProvider`, `InpaintEngine`, `engines/registry.py`) are fixed. **Plan impact:** every milestone stays inside that tree; no `api/`, `orm/`, or extra product features from Stitch that violate non-negotiables unless you explicitly override in 0.5.

### `10-pipeline-engines.mdc`

Layering: `cli.py` / `ui/app.py` → `VideoProcessor` | `ImageProcessor` | `BatchRunner` → `MaskProvider` → `InpaintEngine` (+ optional `TiledInpaint`) → `TemporalSmoother` (video) → ffmpeg. Contracts: mask `uint8 {0,255}` same HxW; images BGR `uint8` at engine boundary. Detection allowed list only (template / static-region / edge-contrast). Engines: OpenCV Telea/NS and LaMa. CLI flags and exit codes are specified. **Plan impact:** M1–M6 implement this stack; UI must not import Gradio into `engines/` `masks/` `io/` `video/`.

### `20-frontend-components.mdc`

Gradio only, `share=False`, bind `127.0.0.1`, thin handlers, `gr.State`, RGB↔BGR at the UI edge, Run disabled without confirmed mask, video preview downsampled. **Plan impact:** M2/M7 follow **locked C1–C16** (five sections + rule behavior). Stitch supplies labels/visual spirit only.

### `30-data-artifacts.mdc`

No database. Artifacts: output media, `{stem}.mask.png`, `{stem}.mask.json` (`schema_version: 1`), `{stem}.keyframes.json`, `models/` weights, temp dirs. Atomic write (`*.tmp` → fsync → replace). `scripts/download_models.py` is the only downloader (SHA256, refuse overwrite without `--force`). **Plan impact:** M4 downloader + mask serialize; processors never download weights at runtime.

### `40-testing-quality.mdc`

`tests/{unit,integration,fixtures}/`; TDD: failing contract test first for new engine/provider; SSIM/PSNR vs committed baselines (example SSIM ≥ 0.95 in the rule); IoU for detector candidates; required edge-case table; `pytest -m "not slow and not gpu"` in default CI; ≥80% on `engines`/`masks`/`io`. **Plan impact:** section 4 — tests ride with each milestone, not only M8.

### `50-security-devops.mdc`

Secrets out of git; pydantic-settings/env for tunables; no HTTP clients in processing packages; two Dockerfiles (CUDA + CPU); CI ruff/mypy/pytest-socket; structlog/JSON logs with `job_id`; multiprocessing `spawn` on Windows; exit 130 on cancel. **Plan impact:** M9 + config module from M1; `.cursor/mcp.json` currently holds a live API key (must not be committed).

---

## 2. Project Structure (verbatim from `00-core-architecture.mdc`)

Copied **without changing the tree**. Descriptions under each node map contracts from `10-pipeline-engines.mdc` and naming from `00-core-architecture.mdc`. Files named in other rules / your milestone list are nested under the matching directory — they do not add new top-level packages.

```
src/watermark_remover/
  cli.py                 # typer entry
  config.py              # pydantic-settings only
  exceptions.py
  io/                    # image + video read/validate
  masks/                 # MaskProvider ABC + manual/auto + serialize
  engines/               # InpaintEngine ABC + opencv/lama + tiling
  video/                 # extract, temporal consistency, re-encode
  batch.py
  ui/app.py              # Gradio only
tests/{unit,integration,fixtures}/
models/                  # pretrained weights; gitignored
scripts/download_models.py
```

| Path | Main types / functions (planned) |
|------|----------------------------------|
| `src/watermark_remover/cli.py` | Typer app `watermark-remover`; flags `--input`, `--mask`, `--engine`, `--output`, `--overwrite`, `--allow-empty-mask`, `--allow-full-mask`, `--export-mask`; exit `0/1/2/3/130`. Thin: parse → processor. |
| `src/watermark_remover/config.py` | `pydantic-settings` settings class; env + `config.toml`; no hardcoded paths/caps. |
| `src/watermark_remover/exceptions.py` | `InputValidationError`, `MaskError`, `EngineError`, `ResourceLimitError`. |
| `src/watermark_remover/io/` | Image/video probe + read + size/resolution reject **before** heavy decode. |
| `src/watermark_remover/io/image.py` | `read_image(path) -> np.ndarray` BGR `uint8` `(H, W, 3)`; `write_image_atomic`. |
| `src/watermark_remover/io/video.py` | `probe_video(path) ->` fps/resolution/duration/codec; `open_capture` streaming only. |
| `src/watermark_remover/io/validate.py` | `validate_input_path`, `validate_size_limits`, `refuse_overwrite_unless_flag`. |
| `src/watermark_remover/masks/` | Plugin package. |
| `src/watermark_remover/masks/base.py` | `MaskProvider` ABC: `get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray` (`uint8` `{0,255}`, HxW = frame). `MaskCandidate` (`mask`, `confidence: float`, `method: str`, `bbox`). |
| `src/watermark_remover/masks/manual.py` | `ManualMaskProvider` — raster from PNG / JSON / UI editor. |
| `src/watermark_remover/masks/auto_detect.py` | `AutoDetectMaskProvider.detect_candidates(...) -> list[MaskCandidate]`; `get_mask` only after confirmation path. |
| `src/watermark_remover/masks/serialize.py` | PNG + JSON `schema_version: 1`; reject unknown major; JSON rasterize then nearest resize. |
| `src/watermark_remover/engines/` | Plugin package. |
| `src/watermark_remover/engines/base.py` | `InpaintEngine` ABC: `process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray` (BGR `uint8` in/out). |
| `src/watermark_remover/engines/opencv_engine.py` | `OpenCVInpaintEngine` — `radius`, `method` in `{telea, ns}`. |
| `src/watermark_remover/engines/lama_engine.py` | `LaMaInpaintEngine` — local **ONNX** via ONNX Runtime (locked Q15); `EngineError` if missing; no download. |
| `src/watermark_remover/engines/tiling.py` | `TiledInpaint` — tile **512px**, overlap **32px** (locked Q3); never unbounded tensors. |
| `src/watermark_remover/engines/registry.py` | Register engines; selector `opencv`/`lama`/`auto` (area vs `mask_area_threshold`); no `if engine ==` outside selector. |
| `src/watermark_remover/video/` | Stream extract → process → write temp → mux. |
| `src/watermark_remover/video/extract.py` | `extract_frames(...)` streaming iterator; not all frames in RAM. |
| `src/watermark_remover/video/encode.py` | ffmpeg re-encode; `-c:a copy`; fps preserved; CRF only (no bitrate field, locked Q17); default `crf=23`. |
| `src/watermark_remover/video/processor.py` | `VideoProcessor` — mask + engine + optional temporal; progress callback. |
| `src/watermark_remover/video/temporal.py` | `TemporalSmoother` — Farneback default; RAFT behind a flag; inpainted region only. |
| `src/watermark_remover/image_processor.py` | `ImageProcessor` — stills adapter (sibling of `batch.py` / `cli.py` / `config.py`). Locked Q1. |
| `src/watermark_remover/batch.py` | `BatchRunner` — deterministic folder sort; pair `foo.jpg` ↔ `foo.mask.png`. CLI-only in phase 1 (locked Q19). |
| `src/watermark_remover/ui/app.py` | Gradio `gr.Blocks` only; `gr.ImageEditor`; handlers call processors. |
| `tests/conftest.py` | Shared fixtures, seed pins, `has_ffmpeg`, clock/callback injection. |
| `tests/fixtures/` | Tiny checked-in PNG/MP4 + masks (few KB). |
| `tests/unit/` | `test_<module>.py` for engines, masks, io, selector. |
| `tests/integration/` | ~5s clip, CLI, optional Gradio smoke. |
| `models/` | Gitignored weights; runtime read-only. |
| `scripts/download_models.py` | Sole HTTP client for weights; SHA256; refuse overwrite without `--force`. |

**Not drawn in the `00` box but required by `10-pipeline-engines.mdc` layering:** `src/watermark_remover/image_processor.py` (`ImageProcessor`). Locked Q1: package-root sibling of `batch.py`, no `processors/` package. This plan does **not** invent `api/`, `routes/`, `orm/`.

**Also required by `50-security-devops.mdc` (not in the `00` tree box):** `pyproject.toml` and/or `requirements.txt`, `.env.example`, `Dockerfile`, `Dockerfile.cpu`, `.dockerignore`, compose file, CI workflow. Listed under M1/M9, not as new product architecture.

---

## 3. Milestone Breakdown

### Milestone order check (not silently changed)

M1 → M9 as given is technically sound (CLI core → UI stills → detect → LaMa/tiles → video I/O → temporal → video UI → leftover tests → Docker/CI).

**Do not reorder unless you agree.** Issues to acknowledge (not silently fixed):

1. **TDD vs M8:** `40-testing-quality.mdc` says write the failing contract test first. Dumping all tests into M8 would violate that. Section 4 proposes tests **with each milestone**; M8 is the remainder (integration video, coverage gate, CI markers).
2. **M2 Image IA (locked C2):** No Stitch “image loaded” frame. Map Video Mode sidebar + Empty State onto the five sections; do not invent a third IA.
3. **Processing screen** is not its own milestone. Run/progress (M2 stills + M7 video) uses C14: uuid4 + real log stream; Stitch-like modal layout allowed.
4. **M3 candidate list:** Accept + Reject (C6); Image section Mask; restyle in M7 if needed.

---

### M1 — CLI stills: `io/` + manual masks + OpenCV + `cli.py`

**Complexity:** Medium

**Rules:** `00-core-architecture.mdc`, `10-pipeline-engines.mdc`, `30-data-artifacts.mdc`, `40-testing-quality.mdc` (contract tests), `50-security-devops.mdc` (config/logging/no HTTP in lib)

**Files to create:**

- `src/watermark_remover/__init__.py`
- `src/watermark_remover/cli.py`
- `src/watermark_remover/config.py`
- `src/watermark_remover/exceptions.py`
- `src/watermark_remover/io/__init__.py`
- `src/watermark_remover/io/image.py`
- `src/watermark_remover/io/validate.py`
- `src/watermark_remover/masks/__init__.py`
- `src/watermark_remover/masks/base.py`
- `src/watermark_remover/masks/manual.py`
- `src/watermark_remover/masks/serialize.py`
- `src/watermark_remover/engines/__init__.py`
- `src/watermark_remover/engines/base.py`
- `src/watermark_remover/engines/opencv_engine.py`
- `src/watermark_remover/engines/registry.py`
- `src/watermark_remover/image_processor.py` (locked Q1)
- `pyproject.toml` (and/or `requirements.txt`) — entry `watermark-remover`
- `tests/conftest.py`
- `tests/unit/test_io.py`, `tests/unit/test_masks.py`, `tests/unit/test_opencv_engine.py`, `tests/unit/test_cli.py` (or CLI in integration)
- `tests/fixtures/` stills (section 4)

**Signatures (contracts):**

```python
class MaskProvider(ABC):
    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """uint8 {0, 255}, shape (H, W) == frame[:2]."""

class ManualMaskProvider(MaskProvider):
    def __init__(self, mask: np.ndarray) -> None: ...

class InpaintEngine(ABC):
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image BGR uint8 (H, W, 3); mask uint8 (H, W) {0, 255}; out BGR uint8 (H, W, 3)."""

class OpenCVInpaintEngine(InpaintEngine):
    def __init__(self, radius: int, method: Literal["telea", "ns"]) -> None: ...

def read_image(path: Path) -> np.ndarray: ...
def write_image_atomic(path: Path, image: np.ndarray) -> None: ...
def load_mask_png(path: Path) -> np.ndarray: ...
def load_mask_json(path: Path, frame_hw: tuple[int, int]) -> np.ndarray: ...
def export_mask_png(path: Path, mask: np.ndarray) -> None: ...
def export_mask_json(path: Path, payload: dict) -> None: ...
def get_engine(name: Literal["opencv", "lama", "auto"], mask: np.ndarray, settings: Settings) -> InpaintEngine: ...
```

`ImageProcessor` in `src/watermark_remover/image_processor.py`: `process(image: np.ndarray, mask: np.ndarray, engine_name: Literal["opencv", "lama", "auto"], config: Settings) -> np.ndarray`.

M1 `get_engine("lama")` → `EngineError` (not implemented yet). `auto` may select OpenCV only until M4.

**Definition of Done:**

- CLI: `watermark-remover --input PATH --mask PATH --engine opencv --output PATH` → `{stem}_inpainted{suffix}` if `--output` omitted.
- Edge cases (`40-testing-quality.mdc`): empty mask → `MaskError` unless `--allow-empty-mask`; full mask → `MaskError` unless `--allow-full-mask` (locked Q14); oversize → `InputValidationError` before heavy decode; output == input → refuse without `--overwrite`.
- OpenCV Telea on fixture: **byte-stable PNG** (rule).
- Same input+mask+config → same output (seeds).
- No network in library code.
- Exit codes 0/1/2/3/130 wired (130 if interrupt during run).

---

### M2 — Gradio Image mode (M1 pipeline)

**Complexity:** Large (C1–C16 locked; implement per 0.5)

**Rules:** `20-frontend-components.mdc`, `10-pipeline-engines.mdc` (thin UI), `00-core-architecture.mdc` (no overwrite default), `50-security-devops.mdc` (`127.0.0.1`, `share=False`)

**Files:** `src/watermark_remover/ui/__init__.py`, `src/watermark_remover/ui/app.py`; unit/smoke tests that do **not** chase 100% Gradio layout coverage.

**Five sections (C1-B), Image-complete (C2-B).** Map Video Mode / Empty State *labels* onto these sections — do not invent a third IA:

1. **Input** — `gr.File` / upload JPG, PNG, WEBP (C3-B). Title `watermark-remover` (C11-B). No 500MB/unauthenticated copy. No avatar/notifications/legal (C10-B).
2. **Mask** — `gr.ImageEditor` bbox + freehand (C7-B, Q18). Import/export `{stem}.mask.png` + `{stem}.mask.json` (C8-B). Sensitivity in `gr.State` default 50% (Q4) when auto-detect is present (M3).
3. **Preview** — overlay mandatory before run (C5-B).
4. **Engine** — `opencv` \| `lama` \| `auto` + Accordion (`radius`, `method`) (C4-B). LaMa CPU warning when no CUDA (C12-B).
5. **Run** — label may be `Process All` (C5-B) but it is **single-file**, not batch (C15-B). Disabled until confirmed mask. Progress: uuid4 `job_id`, structlog stream, Cancel = interrupt + cleanup; Stitch-like modal/percent OK (C14-B). Download result.

Native Gradio `Row`/`Column`/`Accordion`; optional cheap theme tint; no pixel-perfect Stitch CSS (C16-B).

**Signatures in `ui/app.py` (thin):**

```python
def ui_mask_to_uint8(mask: np.ndarray | dict | None) -> np.ndarray: ...
def on_run(image: np.ndarray | None, mask: np.ndarray | None, engine_name: Literal["opencv", "lama", "auto"], config: Settings) -> np.ndarray: ...
```

RGB→BGR before processors; BGR→RGB before display. `gr.State` for mask. Bind `server_name="127.0.0.1"`, `share=False`.

**Definition of Done:**

- Image-only happy path: upload → draw mask → preview overlay → run OpenCV → download; input path not overwritten.
- Run/`Process All` disabled without confirmed mask (C5-B).
- No inpaint math inside click callbacks.
- Gradio smoke optional; no pixel-perfect CSS (C16-B).

---

### M3 — Auto-detect + candidate accept/reject

**Complexity:** Medium–Large

**Rules:** `10-pipeline-engines.mdc` (Allowed detection only), `00-core-architecture.mdc` (never auto-apply), `20-frontend-components.mdc` (candidates + accept), `40-testing-quality.mdc` (IoU, never assert auto-apply)

**Files:** `src/watermark_remover/masks/auto_detect.py`; UI candidate list in `ui/app.py`; `tests/unit/test_auto_detect.py`; fixtures with watermark at varied position/scale.

**Allowed methods only:**

- `cv2.matchTemplate` with user-supplied PNG+alpha
- Static-region via frame differencing (video, ≥2 frames). On a single still: degrade to edge/contrast only; document in the detector docstring (locked Q8).
- Local edge/contrast heuristics
- Session-only threshold refine from yes/no; Stitch `Sensitivity` in `gr.State`, UI default 50% (locked Q4)

**Forbidden:** brand/platform detectors, cloud APIs, training a detector.

**Signatures:**

```python
class AutoDetectMaskProvider(MaskProvider):
    def detect_candidates(self, frame: np.ndarray, frame_idx: int) -> list[MaskCandidate]: ...
    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray: ...
```

`get_mask` must not invent a final mask from unconfirmed candidates.

**UI (C6-B):** `Detection Mode` Auto/Manual, template upload, `Sensitivity`, `Run Detection`, `Candidate N` + `%`, explicit **Accept** and **Reject**. Run stays off until accept/edit.

**Definition of Done:**

- `detect_candidates` returns `list[MaskCandidate]`; no file write of a final mask.
- Tests: precision/recall vs GT mask with **IoU ≥ 0.5** (Q5); never assert auto-apply.
- UI: Enable Run only after accept/edit (C6-B).

---

### M4 — LaMa + tiling + download script

**Complexity:** Large

**Rules:** `10-pipeline-engines.mdc`, `30-data-artifacts.mdc`, `40-testing-quality.mdc`, `50-security-devops.mdc`

**Files:** `src/watermark_remover/engines/lama_engine.py`, `src/watermark_remover/engines/tiling.py`, `scripts/download_models.py`, `tests/unit/test_lama_engine.py`, `tests/unit/test_tiling.py`, `tests/unit/test_download_models.py` (checksum / `--force` / no overwrite)

**Signatures:**

```python
class LaMaInpaintEngine(InpaintEngine):
    def __init__(self, weights_path: Path, device: str) -> None: ...
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

class TiledInpaint:
    def process(self, image: np.ndarray, mask: np.ndarray, engine: InpaintEngine) -> np.ndarray: ...

def download_models(*, force: bool) -> None: ...
```

Seeds: `torch.manual_seed`, numpy, CUDA when stochastic.

**Definition of Done:**

- Missing weights at runtime → `EngineError` with setup command; **no network**.
- Downloader: SHA256 required; write `models/`; refuse overwrite without `--force`.
- Unit: mock tiny stub ONNX/tensor; real weights `@pytest.mark.slow`.
- GPU tests `@pytest.mark.gpu`, skip without CUDA.
- Engine unit: SSIM ≥ **threshold stored in the test** and PSNR ≥ **threshold stored in the test** vs committed baseline (rule example SSIM `>= 0.95` — confirm Q6).
- `auto`: mask area `< mask_area_threshold` → OpenCV else LaMa; user `--engine` wins.
- Tiny/huge resolution: reject or tile; no crash (edge table).

---

### M5 — Video pipeline without temporal smoothing

**Complexity:** Large

**Rules:** `10-pipeline-engines.mdc`, `30-data-artifacts.mdc`, `40-testing-quality.mdc`

**Files:** `src/watermark_remover/io/video.py` (if not stubbed in M1), `src/watermark_remover/video/__init__.py`, `src/watermark_remover/video/extract.py`, `src/watermark_remover/video/encode.py`, `src/watermark_remover/video/processor.py`, `tests/unit/test_video_io.py` (fake ffmpeg), `tests/integration/test_video_cli.py`

**Signatures:**

```python
def extract_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]: ...
def encode_video(frames_dir: Path, audio_src: Path, output: Path, fps: float, crf: int) -> None: ...

class VideoProcessor:
    def process(self, input_path: Path, mask_provider: MaskProvider, engine: InpaintEngine, output_path: Path, progress: Callable[..., None] | None = None) -> Path: ...
```

Stream extract → process → write temp → mux. `max_workers <= cpu_count`. `ResourceLimitError` rather than OOM. Atomic final replace.

**Definition of Done:**

- Integration (~5s fixture): audio stream present, fps equals input (±epsilon), exit 0; skip if no ffmpeg.
- No full-video RAM decode.
- Default output `{stem}_inpainted.mp4` (or original suffix).

---

### M6 — Temporal smoothing

**Complexity:** Medium

**Rules:** `10-pipeline-engines.mdc` (Farneback default, RAFT flag, inpainted region only)

**Files:** `src/watermark_remover/video/temporal.py`, `tests/unit/test_temporal.py`

**Signatures:**

```python
class TemporalSmoother:
    def apply(self, prev: np.ndarray, current: np.ndarray, inpaint_mask: np.ndarray) -> np.ndarray: ...
```

**Definition of Done:** Unit tests on tiny frames; does not modify pixels outside `inpaint_mask`; RAFT not imported unless hidden flag on (C13). No extra flicker metric (Q7).

---

### M7 — Gradio Video mode (five sections + Stitch labels)

**Complexity:** Large (C1–C16 locked)

**Rules:** `20-frontend-components.mdc`, `10-pipeline-engines.mdc`, fetched Video Mode + Processing screens (labels only)

**Files:** extend `src/watermark_remover/ui/app.py` only (no second UI framework).

Same five sections as M2 (C1-B). Video-specific inside those sections (C13 hybrid, C7-B):

- Input: MP4/MOV/WEBM (C3-B); downsampled preview; full-res on Run.
- Mask: first shown frame; toggle **static (all frames)** vs **keyframes**; keep `MASK 1` overlay + `Add Mask Keyframe`; export `{stem}.keyframes.json`.
- Preview: overlay before run; optional compare/visibility as Gradio if cheap (C16-B).
- Engine: `opencv`/`lama`/`auto` (C4-B); CPU warning (C12-B); `Apply Temporal Smoothing` on/off (Farneback; RAFT only hidden flag); `Output Quality` config-gated; `Process Nth frame`; `Keep Original Audio`.
- Run: label `Apply Inpainting` (C5-B); enabled only with confirmed mask; C14 progress/Cancel; no batch `Process All` (C15-B).

**Definition of Done:**

- Video upload → static or keyframe mask → preview overlay → run → download.
- Integration: no crash when mask moves (keyframe fixture) — `40-testing-quality.mdc`.
- Do not decode entire video into UI memory.

---

### M8 — Remaining tests / coverage

**Complexity:** Medium if TDD was done per milestone; Large if not

**Rules:** `40-testing-quality.mdc`, `50-security-devops.mdc` (`pytest-socket`)

**Files:** fill gaps under `tests/unit/`, `tests/integration/`; import-graph guard (engines/masks/video/io must not import HTTP clients); network-open tests must fail.

**Definition of Done:**

- ≥80% on `src/watermark_remover/{engines,masks,io}`
- Default: `pytest -m "not slow and not gpu"`
- Entire edge-case table green
- Detector IoU tests present
- Video integration + keyframe fixture

---

### M9 — Docker + CI

**Complexity:** Medium

**Rules:** `50-security-devops.mdc`

**Files:** `Dockerfile`, `Dockerfile.cpu`, `.dockerignore`, compose (bind-mount in/out; optional `127.0.0.1:7860:7860` only), CI (ruff, mypy, `pytest -m "not slow and not gpu"`, pytest-socket), `.env.example` (empty values), `.gitignore` (`models/`, `.env`, **MCP keys**)

**Definition of Done:**

- Non-root user; copy only `src/`, `scripts/`, `pyproject.toml` / `requirements.txt`; weights volume `/models`
- Pin base digest or major.minor; ffmpeg in-image
- CI does **not** download LaMa weights
- Fail on `# noqa` abuse and on tests opening the network

---

## 4. Test Strategy per Milestone

**Proposal (matches `40-testing-quality.mdc`, not “all tests in M8”):**

Write the **failing contract test first** for every new `InpaintEngine` / `MaskProvider` (`process` shape/dtype; `get_mask` values `{0,255}`). Unit tests land in the same milestone as the code. M8 is leftover integration, coverage floor, lint-guard, and anything skipped as `@slow`/`@gpu`.

| Milestone | Tests written in that milestone |
|-----------|----------------------------------|
| M1 | Contract `get_mask` / `process`; OpenCV Telea byte-stable PNG; empty/full/overwrite/oversize; CLI exit codes |
| M2 | Handler types / RGB↔BGR helper; Run-disabled without mask (C5-B); no engine import of Gradio from `engines/` |
| M3 | Candidate IoU vs GT; never auto-apply; template + heuristic fixtures |
| M4 | LaMa stub shape/dtype + SSIM/PSNR; missing weights no network; tiling bounds; download checksum / `--force` |
| M5 | Fake ffmpeg in unit; real ffmpeg integration ~5s; audio + fps |
| M6 | Temporal only on masked pixels |
| M7 | Keyframe fixture no crash; UI smoke optional |
| M8 | Coverage ≥80% engines/masks/io; pytest-socket; import-graph HTTP guard |
| M9 | CI job definition, not extra product tests |

### Fixtures (`tests/fixtures/` — tiny, checked in, few KB)

Rule: “tiny PNG/MP4 + masks (checked in, few KB)” and integration “short ~5s clip”. Exact pixel sizes and SSIM/PSNR numbers stay in tests, not production. **Fixture list approved (locked Q9).** Tiebreaker: if a ~5s MP4 cannot stay few KB, shrink duration/resolution (e.g. 2s at 160×120); fixture size wins over realism.

| File (proposed names) | Role | Size constraint |
|----------------------|------|-----------------|
| `still_clean.png` | Base still | Few KB |
| `still_logo.png` | Still + synthetic overlay | Few KB |
| `still_logo.mask.png` | GT binary mask `{0,255}` | Few KB |
| `still_logo_inpainted_opencv_telea.png` | OpenCV baseline | Few KB |
| `still_empty.mask.png` | All-0 mask | Few KB |
| `still_full.mask.png` | All-255 mask | Few KB |
| `still_logo.mask.json` | Vector schema v1 | Few KB |
| `template_logo.png` | PNG+alpha for `matchTemplate` | Few KB |
| `detect_pos1.png`, `detect_pos2.png`, `detect_scale.png` | Detector position/scale set (3 stills) | Few KB each |
| `detect_pos1.mask.png`, `detect_pos2.mask.png`, `detect_scale.mask.png` | GT masks | Few KB each |
| `clip_5s.mp4` (or shorter/smaller if needed) | Integration video with audio | Few KB; shrink to e.g. 2s @ 160×120 if 5s cannot stay tiny (locked Q9) |
| `clip_5s.mask.png` | Static mask for clip | Few KB |
| `clip_5s.keyframes.json` | Moving-mask fixture | Few KB |
| `tiny.onnx` or stub tensor | LaMa unit mock | Few KB |

**Still count:** 1 clean + 1 watermarked + 3 detector variants = **5** unique stills (plus masks/baselines).  
**Clip count:** **1** tiny MP4 (target ~5s; shrink if needed). Detector tests: IoU ≥ **0.5** (locked Q5). Engine baselines: SSIM ≥ **0.95**, PSNR ≥ **30 dB** (locked Q6).  
Do not check in real LaMa weights.

---

## 5. Config Schema Draft

Defaults below are **locked Q3 / Q10 / Q15–Q17**. Config search order (locked Q16): (1) `WATERMARK_REMOVER_CONFIG` env path if set, (2) `./config.toml` in cwd, (3) pydantic-settings defaults. No `~/.config/...` in phase 1.

| Field (env / settings) | Type | Default (locked) | Required by |
|------------------------|------|------------------|-------------|
| `MAX_INPUT_BYTES` / `max_input_bytes` | `int` | `2 * 1024**3` (2 GiB) | `50`, `30`, `10` |
| `MODEL_DIR` | `Path` | (still from env / config file; no home-path hardcode) | `50`, `30` |
| `LAMA_WEIGHTS` | `Path` | under `MODEL_DIR` | `50`, `30` |
| `MAX_RAM_MB` / `max_ram_mb` | `int \| None` | `None` (unbounded); enforce when set | `50`, `30` |
| `MAX_VRAM_MB` / `max_vram_mb` | `int \| None` | `None` (unbounded); enforce when set | `50`, `30` |
| `CRF` / `crf` | `int` | `23` | `50`, `30`, `10` |
| `mask_area_threshold` | `float` | `0.03` (3%) | `30`, `10` |
| `LOG_LEVEL` | `str` | (env; rule uses `DEBUG` for per-frame) | `50` |
| OpenCV `radius` | `int` | `3` | `10` |
| OpenCV `method` | `Literal["telea", "ns"]` | (not locked; Telea used for byte-stable fixture tests) | `10` |
| `max_workers` | `int` | `os.cpu_count()`; hard-cap at `cpu_count` even if config asks for more | `10`, `50` |
| RAFT enable flag | `bool` | off unless flag | `10` |
| Tile size / overlap | `int` / `int` | `512` / `32` | `10` + locked Q3 |
| Resolution cap | derived | no fixed pixel cap; derive from `max_ram_mb` / `max_vram_mb` when set | `10` + locked Q3 |
| Output quality / resolution option | settings | config-gated (720p/1080p/etc.), not a hardcoded constant (locked Q12) | Q12 |
| Gradio `server_name` / `share` | `str` / `bool` | `127.0.0.1` / `False` | `20`, `50` |

**Not a settings field:** Sensitivity (UI `gr.State` only, default shown 50%, locked Q4). **No bitrate field** (locked Q17).

CLI-only: `--overwrite`, `--allow-empty-mask`, `--allow-full-mask`, `--export-mask`, `--engine`, `--input`, `--output`, `--mask`. Full-image mask → `MaskError` unless `--allow-full-mask` (locked Q14).

---

## 6. Dependency & Setup Risks

- **Stitch MCP in Cursor:** `tools/list` schema (`$defs/ScreenInstance`) does not register tools in Cursor. This plan used JSON-RPC to the same endpoint. Re-fetch later the same way if Cursor is still broken.
- **`.cursor/mcp.json` contains a live key.** `50-security-devops.mdc`: no secrets in git. Gitignore it; rotate if it was committed or pasted in chat logs.
- **`X-Goog-User-Project` set to the Stitch design id** breaks `get_project` (“not found”). Remove or set to a real GCP project id if you use that header.
- **Two Dockerfiles** (CUDA + CPU), ffmpeg in-image, weights via `/models` volume (`50-security-devops.mdc`).
- **LaMa weights:** SHA256 in downloader; default CI must not download; missing weights → `EngineError`.
- **`pytest-socket`:** any test that opens a socket fails; mock network for downloader tests.
- **Windows `spawn`:** you are on Windows; `ProcessPoolExecutor` / multiprocessing must use `spawn`; cap `max_workers`.
- **BGR vs RGB** at Gradio/Pillow edges — easy to regress.
- **Stitch lists SD and ProPainter** — implementing them is a new engine + weights problem and is out of the current engine table.
- **OneDrive path** (`Documents\projects\...`): file-lock / sync during atomic replace and ffmpeg temp files.
- **Pinning (locked Q10):** Python 3.10+; `opencv-python` >=4.9; `numpy` >=1.26,<2.0; `typer` >=0.12; `gradio` >=4.0; `pydantic-settings` >=2.0; `onnxruntime` latest stable (Q15); `ffmpeg-python` >=0.2; `pytest` >=8.0; `ruff` >=0.5; `mypy` >=1.10; `structlog` >=24.0. Verify latest patch at implementation time.

---

## 7. Open Questions

**None remaining for phase 1.** Q1–Q20 and C1–C16 are locked (section 0.5 + below).

### Locked

1. **Q1.** `src/watermark_remover/image_processor.py` — package-root sibling of `batch.py` / `cli.py` / `config.py`.
2. **Q2.** C1–C16 locked in section 0.5 (all **B** except C13 **hybrid**, C14 **B** with Stitch-like modal layout allowed).
3. **Q3.** `max_input_bytes=2GiB`; `mask_area_threshold=0.03`; `max_ram_mb`/`max_vram_mb` default `None` (enforce when set); `crf=23`; OpenCV `radius=3`; `max_workers=os.cpu_count()` hard-capped; tile 512 / overlap 32; no fixed pixel resolution cap (derive from RAM/VRAM).
4. **Q4.** Sensitivity is `gr.State` only; UI default 50%; not in pydantic-settings.
5. **Q5.** IoU min = 0.5.
6. **Q6.** SSIM ≥ 0.95; PSNR ≥ 30 dB.
7. **Q7.** No extra temporal metric; M6 as specified.
8. **Q8.** Single still: degrade auto-detect to edge/contrast; document in detector docstring.
9. **Q9.** Fixture list approved; shrink clip rather than exceed “few KB”.
10. **Q10.** Pins in section 6; verify latest patch at implementation.
11. **Q11.** Display/CLI name `watermark-remover` (C11-B).
12. **Q12 / C13.** Nth-frame + config-gated Output Quality; temporal on/off (Farneback; RAFT hidden); Keep Original Audio.
13. **Q13.** Omit Object List / History / Shortcuts / Support in phase 1.
14. **Q14.** Full-image mask → `MaskError` unless `--allow-full-mask`.
15. **Q15.** LaMa phase 1 = ONNX Runtime.
16. **Q16.** `WATERMARK_REMOVER_CONFIG` → `./config.toml` → settings defaults. No home-dir config.
17. **Q17.** CRF only; no bitrate field.
18. **Q18.** Latest stable Gradio; `gr.ImageEditor` (not Sketchpad).
19. **Q19 / C15.** Batch/folder = CLI `batch.py` only; no UI batch `Process All`.
20. **Q20 / C14.** UI log = same structlog/callback stream; `job_id` uuid4; Cancel = real interrupt + cleanup.

---

## Stop

Plan is complete for phase 1. **No code until you ask for a milestone.** First implementation step when you say so: **M1** (failing contract tests first).
