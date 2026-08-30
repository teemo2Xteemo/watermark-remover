# Requirements: Local Watermark/Object Removal Tool

## 1. Mục tiêu (Purpose)

Xây dựng một công cụ chạy hoàn toàn **local** (không gọi API/dịch vụ AI bên ngoài) để:
- Phát hiện (detect) vùng chứa watermark/logo/overlay trên ảnh và video
- Xóa vùng đó bằng kỹ thuật inpainting, tái tạo lại nội dung nền một cách hợp lý
- Hỗ trợ cả chế độ tự động (auto-detect) và chế độ người dùng tự chọn vùng (manual mask)

**Phạm vi kỹ thuật**: đây là bài toán object removal / inpainting tổng quát. Watermark chỉ là một loại "vùng cần xóa" trong số nhiều loại (logo, text overlay, vật cản, timestamp...).

**Ràng buộc quan trọng**: người dùng chịu trách nhiệm đảm bảo có quyền hợp pháp chỉnh sửa/tái sử dụng nội dung đầu vào. Tool không có nghĩa vụ (và không nên) xác minh quyền sở hữu — nhưng cũng không được thiết kế các tính năng chuyên biệt nhằm mục đích duy nhất là che giấu nguồn gốc AI-generated (ví dụ: không tự động phát hiện *loại* watermark cụ thể của từng platform AI để tối ưu xóa riêng cho platform đó).

---

## 2. Phạm vi (Scope)

### In-scope
- Xử lý ảnh tĩnh: JPG, PNG, WEBP
- Xử lý video: MP4, MOV, WEBM (codec H.264/H.265 phổ biến)
- 2 chế độ chọn vùng: manual (vẽ mask) và auto-detect (thuật toán, không phải AI service ngoài)
- 2+ engine inpainting: nhẹ (OpenCV) và chất lượng cao (LaMa local model)
- CLI tool + optional local web UI (Gradio/Streamlit)
- Xử lý batch (nhiều file cùng lúc)

### Out-of-scope (phase 1)
- Gọi API cloud (Adobe, Photoshop API, các AI inpainting SaaS...)
- Huấn luyện model detection riêng (dùng pretrained sẵn có)
- Watermark ẩn dạng steganographic/invisible (C2PA metadata, digital fingerprint) — đây là loại watermark khác hẳn về bản chất, nằm ngoài phạm vi image processing
- Real-time xử lý (livestream)

---

## 3. Kiến trúc tổng quan

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌─────────────┐
│   Input      │────▶│  Mask Acquisition │────▶│  Inpainting Engine │────▶│   Output    │
│ (image/video)│     │  (manual/auto)    │     │  (OpenCV/LaMa)     │     │             │
└─────────────┘     └──────────────────┘     └───────────────────┘     └─────────────┘
                                                       │
                                              ┌────────┴────────┐
                                              │  Video-only:     │
                                              │  Temporal        │
                                              │  Consistency     │
                                              │  (optical flow)  │
                                              └─────────────────┘
```

---

## 4. Functional Requirements

### FR1 — Input Handling
- FR1.1: Đọc ảnh qua OpenCV/Pillow, validate format & kích thước
- FR1.2: Đọc video qua ffmpeg-python hoặc OpenCV VideoCapture, extract metadata (fps, resolution, duration, codec)
- FR1.3: Reject/cảnh báo file quá lớn (config threshold, ví dụ >2GB video) trước khi xử lý

### FR2 — Manual Mask Selection
- FR2.1: UI cho phép người dùng vẽ **bounding box** hoặc **freehand polygon** lên frame/ảnh
- FR2.2: Với video: chọn mask ở frame đầu tiên, có tùy chọn:
  - Áp dụng mask cố định cho toàn bộ video (watermark tĩnh)
  - Cho phép thêm keyframe mask mới tại timestamp chỉ định (watermark di chuyển/đổi vị trí)
- FR2.3: Export/import mask dưới dạng file (PNG mask hoặc JSON polygon points) để tái sử dụng
- FR2.4: Preview mask overlay trước khi chạy inpainting

### FR3 — Auto-Detection
- FR3.1: **Template matching**: người dùng cung cấp ảnh mẫu watermark (logo riêng lẻ, dạng PNG có alpha) → dùng `cv2.matchTemplate` tìm vị trí xuất hiện trong ảnh/từng frame
- FR3.2: **Static-region detection cho video**: phân tích N frame đầu, tìm vùng có độ biến thiên pixel thấp bất thường so với nền động xung quanh (watermark tĩnh thường "đứng yên" trong khi nội dung nền chuyển động) — dùng frame differencing + threshold
- FR3.3: **Edge/contrast heuristic**: phát hiện vùng có pattern viền sắc nét cục bộ khác biệt với vùng lân cận (áp dụng cho overlay bán trong suốt)
- FR3.4: Auto-detect trả về **candidate mask(s)** kèm confidence score — không tự động áp dụng, luôn yêu cầu người dùng xác nhận/chỉnh sửa trước khi inpaint (tránh xóa nhầm vùng nội dung quan trọng)
- FR3.5: Cho phép người dùng feedback (đúng/sai) để refine threshold trong session hiện tại (không cần retrain model)

### FR4 — Inpainting Engine
- FR4.1: **Engine A — OpenCV classic** (`cv2.inpaint`, Telea hoặc Navier-Stokes algorithm)
  - Dùng cho: watermark nhỏ, nền đơn giản/ít texture, cần tốc độ cao
  - Config: radius, method
- FR4.2: **Engine B — LaMa (Large Mask Inpainting)**, chạy local qua ONNX Runtime hoặc PyTorch
  - Dùng cho: watermark lớn, nền phức tạp, cần chất lượng cao
  - Model weights tải sẵn về máy (pretrained, open-source license), không gọi API
  - Hỗ trợ chạy CPU (chậm) hoặc GPU/CUDA (khuyến nghị)
- FR4.3: Cho phép người dùng chọn engine thủ công, hoặc auto-select dựa trên kích thước mask (mask nhỏ hơn X% diện tích → OpenCV, lớn hơn → LaMa)
- FR4.4: Xử lý theo batch tile nếu ảnh/frame có độ phân giải rất lớn (tránh OOM)

### FR5 — Video-Specific Processing
- FR5.1: Extract frames qua ffmpeg → xử lý từng frame theo mask tương ứng
- FR5.2: **Temporal consistency**: áp dụng optical flow (`cv2.calcOpticalFlowFarneback` hoặc RAFT nếu cần chất lượng cao hơn) để làm mượt vùng đã inpaint giữa các frame liên tiếp, giảm flicker
- FR5.3: Re-encode video giữ nguyên audio track gốc, fps, và chất lượng tương đương input (config CRF/bitrate)
- FR5.4: Hỗ trợ xử lý song song nhiều frame (multiprocessing) để tăng tốc, giới hạn theo số CPU core available
- FR5.5: Progress tracking (% hoàn thành) cho video dài

### FR6 — Output
- FR6.1: Giữ nguyên format, resolution, metadata cơ bản (trừ vùng đã chỉnh sửa) so với input
- FR6.2: Tùy chọn export kèm file mask đã dùng (để tái sử dụng/audit)
- FR6.3: Naming convention rõ ràng cho output (tránh ghi đè input gốc mặc định)

### FR7 — Interfaces
- FR7.1: **CLI**: `tool.py --input <path> --mask <path|auto> --engine <opencv|lama> --output <path>`
- FR7.2: **Local Web UI** (Gradio/Streamlit): upload file → vẽ mask trực quan → chọn engine → preview → download
- FR7.3: Batch mode: xử lý folder với cùng 1 mask hoặc cùng 1 template detection

---

## 5. Non-Functional Requirements

| # | Yêu cầu | Chi tiết |
|---|---|---|
| NFR1 | **Local-only** | Không có network call nào tới AI service bên ngoài trong pipeline xử lý chính (trừ việc tải model weights pretrained 1 lần lúc setup) |
| NFR2 | **Performance** | Ảnh đơn: <5s với OpenCV, <15s với LaMa (CPU); GPU giảm đáng kể. Video: xử lý được tối thiểu ~5-10 fps trên GPU tầm trung |
| NFR3 | **Resource limit** | Có config giới hạn RAM/VRAM sử dụng, tránh crash trên máy cấu hình thấp |
| NFR4 | **Extensibility** | Kiến trúc plugin cho phép thêm engine inpainting mới (interface chung `InpaintEngine.process(image, mask) -> image`) |
| NFR5 | **Cross-platform** | Chạy được trên Windows/Linux/macOS (lưu ý: LaMa + CUDA cần setup riêng theo OS) |
| NFR6 | **Reproducibility** | Cùng input + cùng mask + cùng engine config → cùng output (seed cố định nếu engine có randomness) |
| NFR7 | **No hidden telemetry** | Không thu thập/gửi dữ liệu người dùng ra ngoài |

---

## 6. Tech Stack đề xuất

```
Ngôn ngữ:        Python 3.10+
Image/Video I/O: OpenCV, Pillow, ffmpeg-python
Inpainting:      
  - opencv-contrib-python (cv2.inpaint)
  - LaMa (ONNX Runtime hoặc PyTorch, model: big-lama pretrained)
Optical flow:    OpenCV (Farneback) / RAFT (optional, nếu cần chất lượng cao)
UI:              Gradio (khuyến nghị — nhanh, hỗ trợ vẽ mask có sẵn component)
CLI:             argparse hoặc typer
Parallelism:     multiprocessing / concurrent.futures cho video frame batch
Packaging:       requirements.txt + optional Docker image (đảm bảo môi trường CUDA đồng nhất)
```

---

## 7. Data Flow chi tiết (cho AI coding agent tham khảo khi implement)

```python
# Pseudocode kiến trúc chính

class MaskProvider(ABC):
    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Trả về binary mask (0/255) cùng kích thước frame"""

class ManualMaskProvider(MaskProvider):
    # Load từ file mask đã vẽ, hoặc giữ nguyên 1 mask cho toàn video

class AutoDetectMaskProvider(MaskProvider):
    # Template matching hoặc static-region analysis
    def detect_candidates(self, frames: list[np.ndarray]) -> list[MaskCandidate]:
        ...

class InpaintEngine(ABC):
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ...

class OpenCVInpaintEngine(InpaintEngine): ...
class LaMaInpaintEngine(InpaintEngine): ...

class VideoProcessor:
    def __init__(self, mask_provider: MaskProvider, engine: InpaintEngine):
        ...
    def process(self, input_path: str, output_path: str):
        # 1. extract frames (ffmpeg)
        # 2. for each frame: get_mask() -> engine.process()
        # 3. apply temporal smoothing (optical flow)
        # 4. re-encode with original audio (ffmpeg)
```

---

## 8. Testing Requirements

- Unit test cho từng `InpaintEngine` với ảnh mẫu + mask cố định, so sánh output với baseline (SSIM/PSNR threshold)
- Test auto-detect với bộ ảnh có watermark ở vị trí/kích thước khác nhau, đo precision/recall của mask candidate
- Test video pipeline với clip ngắn (~5s), kiểm tra: không mất audio, fps giữ nguyên, không crash với watermark di chuyển
- Test edge case: mask trùng hoàn toàn vùng ảnh, mask rỗng, input resolution cực lớn/cực nhỏ

---

## 9. Ghi chú giới hạn & rủi ro kỹ thuật

- Inpainting **không "phục dựng" nội dung gốc thật sự** — nó tái tạo nội dung hợp lý dựa trên ngữ cảnh xung quanh (plausible, không chính xác 100%). Với vùng watermark che chi tiết phức tạp (mặt người, text nhỏ), kết quả có thể bị mờ/artifact.
- Watermark dạng animated hoặc semi-transparent thay đổi opacity theo frame sẽ khó xử lý bằng static mask — cần xử lý theo alpha-blend estimation, phức tạp hơn nhiều so với mask nhị phân thông thường.
- LaMa model chạy CPU sẽ chậm đáng kể với video độ phân giải cao (1080p+) — nên khuyến nghị GPU hoặc hạ resolution khi preview.