# watermark-remover

Công cụ xóa logo, chữ chìm, timestamp hoặc một vùng nhỏ trên **ảnh**. Tool chạy trên máy bạn, không cần tài khoản.

Phiên bản hiện tại: **0.1.1**. Chỉ xử lý ảnh (JPG, PNG, WEBP). **Chưa xử lý video.**

---

## 1. Tool này dùng để làm gì

Bạn mở ảnh, khoanh vùng muốn xóa (ví dụ logo góc ảnh, chữ đè lên hình, hộp carton trong phòng), rồi bấm chạy. Tool tô lại vùng đó cho khớp với nền xung quanh và cho bạn tải ảnh kết quả.

Mọi việc xảy ra trên máy tính của bạn. Ảnh không được gửi lên internet khi xử lý.

---

## 2. Tool này **không** làm gì / lưu ý quan trọng

- **Chạy hoàn toàn trên máy bạn.** Không cần đăng nhập. Không thu thập dữ liệu sử dụng.
- Khi xử lý ảnh, tool **không** gọi dịch vụ AI trên mạng. Bước duy nhất cần internet là **tải sẵn “bộ não” xử lý ảnh về máy** (một lần, xem mục 5) nếu bạn muốn dùng chế độ LaMa.
- **Không** xóa watermark “vô hình” (dấu ẩn trong file, không nhìn thấy bằng mắt).
- **Không** tự xóa watermark khi bạn chưa xác nhận vùng cần xóa.
- Tool **không** ghi đè file ảnh gốc, trừ khi bạn chủ động bật tùy chọn đó (chỉ có ở lệnh nâng cao).
- Bạn cần **tự đảm bảo mình có quyền chỉnh sửa** ảnh đưa vào tool.

---

## 3. Yêu cầu máy tính

| Mục | Thực tế trong project |
|-----|------------------------|
| Hệ điều hành | Windows, macOS, Linux (theo yêu cầu sản phẩm) |
| Python | **3.10 trở lên** (khai báo trong `pyproject.toml`) |
| Card đồ họa (GPU) | **Không bắt buộc.** Không có GPU thì vẫn chạy được, nhưng chế độ LaMa **chậm hơn** (giao diện sẽ hiện dòng cảnh báo). Có GPU NVIDIA + CUDA thì LaMa có thể nhanh hơn — tool tự phát hiện, không cần bạn cấu hình tay. |
| RAM / VRAM tối thiểu | **Không ghi số cứng trong code.** Có thể đặt giới hạn tùy chọn; mặc định không giới hạn. |
| Dung lượng ổ đĩa | File “bộ não” LaMa đúng **208.044.816 byte** (khoảng **198 MiB**). Chưa có số đo dung lượng Python + thư viện trong repo. |
| Giới hạn file ảnh | Mặc định tối đa **2 GiB** mỗi ảnh (cấu hình `MAX_INPUT_BYTES`) |
| Định dạng ảnh | JPG, PNG, WEBP |
| ffmpeg | **Không cần** cho phiên bản này (chưa có xử lý video) |

---

## 4. Chuẩn bị trên máy (từng bước)

Làm lần lượt. Mỗi lệnh bên dưới: đọc dòng giải thích trước, rồi mới copy vào cửa sổ lệnh.

Cửa sổ lệnh là chương trình để gõ lệnh chữ:

- **Windows:** Command Prompt, PowerShell, hoặc Git Bash
- **macOS:** Terminal (ứng dụng Sẵn có)

### Bước A — Cài Python

Python là ngôn ngữ để chạy tool này. Cần bản **3.10 hoặc mới hơn**.

1. Tải bản cài đặt chính thức: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Windows: [trang tải Windows](https://www.python.org/downloads/windows/)
   - macOS: [trang tải macOS](https://www.python.org/downloads/macos/)
2. Chạy file vừa tải và làm theo hướng dẫn trên màn hình.
3. **Windows — rất quan trọng:** trong màn hình cài đặt, **tick ô “Add python.exe to PATH”** rồi mới bấm Install. Nếu quên, lệnh `python` sẽ không chạy được sau này.

Kiểm tra đã cài đúng chưa.

**Windows** — lệnh này hỏi máy: “Python đang là bản nào?”

```text
python --version
```

**macOS** (và Linux) — trên máy Apple thường dùng `python3`:

```text
python3 --version
```

Kết quả cần có dạng `Python 3.10...`, `3.11...`, `3.12...` hoặc cao hơn. Nếu máy báo không tìm thấy lệnh, xem mục 8.

Trên Windows, nếu `python` không chạy, thử:

```text
py --version
```

### Bước B — Tải project về máy

Cách đơn giản nhất (không cần biết lập trình): tải file nén từ GitHub (trang chứa mã nguồn của tool).

1. Mở: [https://github.com/teemo2Xteemo/watermark-remover](https://github.com/teemo2Xteemo/watermark-remover)
2. Bấm nút xanh **Code**, rồi **Download ZIP**.
3. Giải nén (Extract) ra một thư mục dễ nhớ, ví dụ `Documents`.
4. Vào trong thư mục vừa giải nén. Bạn sẽ thấy file `pyproject.toml` và thư mục `src`. Đó là **thư mục gốc của project** — mọi lệnh sau này phải chạy **từ đây**.

Cách khác: nếu bạn đã có sẵn thư mục project trên máy (đã giải nén hoặc đã copy), bỏ qua bước tải, chỉ cần mở cửa sổ lệnh **trong thư mục đó**.

*(Git là phần mềm chép nguyên một project từ internet và giữ lịch sử thay đổi. Người không chuyên không bắt buộc dùng git.)*

### Bước C — Mở cửa sổ lệnh đúng thư mục

**Windows (Command Prompt):**

```text
cd đường-dẫn-tới-thư-mục-project
```

Ví dụ, nếu giải nén ra `C:\Users\me\Documents\watermark-remover-main`:

```text
cd C:\Users\me\Documents\watermark-remover-main
```

**Windows (Git Bash) / macOS:**

```text
cd /đường/dẫn/tới/thư-mục-project
```

Nếu đường dẫn có khoảng trắng, bọc trong dấu ngoặc kép, ví dụ `"C:\Users\me\OneDrive\Documents\watermark-remover"`.

### Bước D — Tạo môi trường ảo rồi cài thư viện

**Môi trường ảo** = một thư mục riêng (tên `.venv`) chứa Python và thư viện của **riêng tool này**, không đụng phần mềm khác trên máy.

**Windows** — tạo môi trường ảo:

```text
python -m venv .venv
```

**macOS** — tạo môi trường ảo:

```text
python3 -m venv .venv
```

Bật môi trường ảo (cửa sổ lệnh sẽ hiện `(.venv)` ở đầu dòng khi thành công):

**Windows Command Prompt:**

```text
.venv\Scripts\activate.bat
```

**Windows PowerShell:**

```text
.venv\Scripts\Activate.ps1
```

**Windows Git Bash:**

```text
source .venv/Scripts/activate
```

**macOS** (và Linux):

```text
source .venv/bin/activate
```

Cài tool + giao diện web + bộ chạy LaMa. Dấu ngoặc kép **cần giữ** (nhất là trên PowerShell). Lệnh này đọc danh sách thư viện thật trong `pyproject.toml` (`ui` = Gradio, `lama` = onnxruntime) và cài vào `.venv`:

**Windows:**

```text
pip install -e ".[ui,lama]"
```

**macOS:**

```text
pip install -e ".[ui,lama]"
```

Đợi tới khi xong, không đóng cửa sổ. Lần đầu có thể hơi lâu vì phải tải thư viện.

Nếu bạn **chỉ** muốn chế độ nhanh `opencv` (không dùng LaMa), có thể cài nhẹ hơn — **không** có onnxruntime:

```text
pip install -e ".[ui]"
```

---

## 5. Tải “bộ não” xử lý ảnh (chỉ cần khi dùng LaMa)

Chế độ **opencv** (mặc định trên giao diện) **không** cần bước này — có thể nhảy sang mục 6.

Chế độ **lama** cần một file model trên máy. Script `scripts/download_models.py` tải file đó **một lần** từ Hugging Face về thư mục `models/` (mặc định tên `lama.onnx`, theo cấu hình `MODEL_DIR` / `LAMA_WEIGHTS`). Đây **không** phải bước xử lý ảnh; pipeline không tự tải khi bạn bấm chạy.

- Dung lượng file: **208.044.816 byte** (khoảng 198 MiB).
- Script kiểm tra mã SHA256; tải thiếu hoặc sai thì báo lỗi, không dùng file hỏng.
- Tối đa **3 lần thử**, mỗi lần chờ mạng tối đa **600 giây**.
- Nếu file đích **đã có**, script **không ghi đè** trừ khi bạn thêm `--force`.
- Cần internet **cho bước này**. Sau đó xử lý ảnh chạy offline.

Trong môi trường ảo đã bật, đứng ở thư mục gốc project:

```text
python scripts/download_models.py
```

(macOS: nếu `python` không có, dùng `python3 scripts/download_models.py`.)

Thành công thì trong log có dòng kiểu đã tải / đã kiểm tra SHA256. File nằm mặc định tại `models/lama.onnx` (trừ khi bạn đổi `LAMA_WEIGHTS` trong file `.env`).

---

## 6. Cách chạy tool (giao diện trên trình duyệt)

Giao diện là trang web **chỉ máy bạn xem được** (`127.0.0.1` = “chính máy này”; `share=False` nghĩa là không tạo link công khai).

Trong môi trường ảo đã bật, đứng ở thư mục gốc:

```text
watermark-remover ui
```

Giữ **mở** cửa sổ lệnh. Trong đó sẽ có địa chỉ local, thường giống:

```text
http://127.0.0.1:7860
```

(Cổng **không** bị ghim trong code project; Gradio thường dùng **7860** nếu cổng đó trống. Hãy mở đúng địa chỉ mà cửa sổ lệnh in ra.)

Mở trình duyệt (Chrome, Edge, Safari…) và dán địa chỉ đó. Đừng đóng cửa sổ lệnh khi đang dùng.

Dừng tool: quay lại cửa sổ lệnh, bấm `Ctrl+C`.

### Cho người quen dùng cửa sổ lệnh (có thể bỏ qua)

Cần sẵn **ảnh** và **file mask** (ảnh đen-trắng PNG, hoặc JSON mask của tool). Tool **từ chối** `--mask auto` — không tự xóa khi chưa có mask.

Lệnh đầy đủ (thay đường dẫn cho đúng máy bạn):

```text
watermark-remover --input anh.jpg --mask anh.mask.png --engine opencv
```

- `--input` — ảnh JPG / PNG / WEBP
- `--mask` — file `.png` hoặc `.json`
- `--engine` — `opencv`, `lama`, hoặc `auto` (mặc định `opencv`)
- `--output` — tùy chọn; nếu bỏ trống, file ra cạnh ảnh gốc, tên `{tên-gốc}_inpainted{đuôi}` (ví dụ `anh_inpainted.jpg`)
- `--overwrite` — cho phép ghi đè **đúng file gốc** (mặc định thì không)
- `--export-mask` — ghi thêm `{stem}.mask.png` và `{stem}.mask.json`

`auto` nghĩa là: nếu vùng mask **nhỏ hơn 3%** diện tích ảnh thì dùng opencv, còn lại dùng lama (ngưỡng `mask_area_threshold = 0.03`). Lệnh `--engine` / ô Engine trên giao diện **luôn thắng** lựa chọn tự động.

---

## 7. Cách dùng trên giao diện

Giao diện có 5 khối đúng như đã làm: **Input → Mask → Preview → Engine → Run**. Tiêu đề trang: `watermark-remover`, dòng phụ: *Image mode — local inpainting. No cloud calls.*

1. **Input** — bấm **Open File**, chọn JPG / PNG / WEBP. Ảnh hiện ở **Loaded image**. Dòng chữ ghi giới hạn dung lượng (mặc định **2 GiB**).
2. **Mask** — mặc định **Detection Mode = Manual**.
   - Tô vùng cần xóa bằng cọ đỏ trên **Mask editor (freehand)**. Sau khi vẽ, nếu có nút **Apply** trong khung vẽ thì bấm Apply (giao diện cũng cập nhật preview khi Apply).
   - Hoặc điền **BBox x / y / width / height** rồi bấm **Add bounding box** (hình chữ nhật).
3. **Preview** — bấm **Update preview**. Ảnh **Mask overlay** hiện vùng sẽ xóa (lớp màu xanh bán trong). Xem kỹ: chỉ những chỗ tô mới bị sửa.
4. Bấm **Confirm mask**. Nút **Process All** lúc này mới bật (trước đó nút bị khóa). Dòng trạng thái: *mask confirmed — Process All is enabled*.
5. **Engine** — chọn:
   - `opencv` — nhanh, không cần file model (mặc định)
   - `lama` — cần đã làm mục 5; hợp vùng lớn hơn; không GPU thì chậm hơn
   - `auto` — xem mục 6; vùng nhỏ thường vẫn ra opencv
6. **Run** — bấm **Process All**. Xem **Result**. Tải file ở **Download result** (tên dạng `{tên}_inpainted.png`). Dòng Status ghi engine thật sự đã dùng, ví dụ `engine: lama`.
7. Muốn dừng giữa chừng: **Cancel**.

**Chế độ Auto** (tùy chọn): chọn **Auto** trong Detection Mode. Có thể tải **Upload watermark template (PNG/JPG)** (cắt sát logo, PNG trong suốt thì tốt hơn), chỉnh **Sensitivity**, bấm **Run Detection**. Chọn một **Candidate**, bấm **Accept** (mới được dùng) hoặc **Reject**. Tool **không** tự xóa khi mới phát hiện — bắt buộc Accept rồi Confirm mask.

Import / export mask: **Import mask (.png / .json)** và **Export mask PNG + JSON**.

---

## 8. Xử lý lỗi thường gặp

**Python không chạy**  
Triệu chứng: `python is not recognized`, `command not found: python`.  
Cách xử lý: cài lại Python, Windows nhớ tick **Add to PATH**. Thử `py --version` (Windows) hoặc `python3 --version` (macOS). Đóng hết cửa sổ lệnh rồi mở lại sau khi cài.

**Quên bật môi trường ảo**  
Triệu chứng: `watermark-remover` không phải là lệnh; hoặc cài thư viện vào Python “toàn máy”.  
Cách xử lý: chạy lại lệnh `activate` ở mục 4 tới khi thấy `(.venv)`. Rồi mới `pip install` / `watermark-remover ui`.

**Thiếu giao diện Gradio**  
Triệu chứng: `Gradio is required for the UI. Install with: pip install 'watermark-remover[ui]'`.  
Cách xử lý: trong `.venv`, chạy `pip install -e ".[ui]"` (hoặc `".[ui,lama]"`).

**LaMa báo thiếu model**  
Triệu chứng: dòng lỗi dạng `LaMa weights not found (...). Run: python scripts/download_models.py`.  
Cách xử lý: chạy đúng lệnh đó từ thư mục gốc, trong `.venv`. Nếu bạn đặt `LAMA_WEIGHTS` trỏ vào file rất nhỏ (placeholder), tool sẽ bỏ qua file đó và thử `models/lama.onnx`.

**Tải model thất bại / hết đĩa**  
Triệu chứng: `incomplete download`, `download failed`, hoặc máy báo đầy ổ.  
Cách xử lý: cần trống **hơn** 208.044.816 byte. Xóa file `.tmp` nếu còn, chạy lại script. File cũ hợp lệ thì thêm `--force` mới ghi đè.

**Nút Process All bấm không được / xám**  
Triệu chứng: nút không bấm được; Status *confirm the preview overlay* hoặc *draw a mask*.  
Cách xử lý: phải có vùng tô (không để trống) → **Update preview** (hoặc Apply trên khung vẽ) → **Confirm mask**.

**PowerShell chặn activate**  
Triệu chứng: lỗi execution policy khi chạy `Activate.ps1`.  
Cách xử lý: dùng Command Prompt với `activate.bat`, hoặc Git Bash với `source .venv/Scripts/activate`.

**Không mở được trang web**  
Triệu chứng: trình duyệt không vào được, hoặc log báo cổng đã dùng.  
Cách xử lý: copy **đúng** URL in trong cửa sổ lệnh (có thể không phải 7860). Đừng dùng địa chỉ mạng khác; tool chỉ gắn `127.0.0.1`. Tắt tool cũ (`Ctrl+C`) rồi chạy lại `watermark-remover ui`.

---

## 9. Giới hạn hiện tại

Những gì **đang chạy được** trên codebase này (ảnh + giao diện + opencv + LaMa + tự gợi ý vùng, không tự áp):

- Ảnh tĩnh JPG / PNG / WEBP, tối đa 2 GiB (mặc định).
- Vẽ mask tay, hình chữ nhật, import/export mask, gợi ý vùng (phải Accept + Confirm).
- Hai cách tô lại nền: `opencv` và `lama` (cần model).
- Giao diện local và lệnh `watermark-remover` như trên.

Những gì **chưa có**:

- **Video** (không có pipeline video trong `src/`; extra `video` / ffmpeg trong `pyproject.toml` là chỗ dành cho sau).
- Xử lý hàng loạt thư mục (batch).
- Tài khoản, chia sẻ link công khai, chạy trên máy khác trong mạng (cố ý gắn `127.0.0.1`).
- Tự nhận diện watermark theo từng nền tảng (Midjourney, ChatGPT, v.v.).
- Watermark vô hình / dấu vân file.

Kết quả LaMa phụ thuộc mask: tô **sát** vật cần xóa. Tô cả khung hình dễ làm ảnh bị nhòe.

---

Video và tính năng khác có thể được thêm sau; README này chỉ mô tả những gì dùng được **ngay bây giờ**.

---

## 10. Cần xác nhận thêm

Các ý sau **không** có số liệu chắc trong code/rule, nên **không** đưa vào hướng dẫn như đã đo:

1. **RAM tối thiểu khuyến nghị** cho máy thường? (`MAX_RAM_MB` mặc định để trống = không giới hạn.)
2. **VRAM tối thiểu** khi chạy LaMa trên GPU?
3. **Dung lượng ổ đĩa** cho bản cài Python + `opencv-python` + Gradio + `onnxruntime` (ngoài 208.044.816 byte của model)?
4. **Thời gian tải model** trên mạng gia đình điển hình? (Code chỉ có timeout 600 giây × 3 lần.)
5. Thời gian xử lý 1 ảnh trên CPU/GPU có đạt mục tiêu trong `requirements.md` (OpenCV &lt; 5 giây, LaMa CPU &lt; 15 giây) hay không — **chưa verify trên máy sạch** trong tài liệu này.
6. Cả chuỗi “máy chưa cài Python → venv → pip → download → `watermark-remover ui`” trên Windows sạch và macOS sạch: **chưa verify trên máy sạch** khi viết README.
7. Cổng trình duyệt: project **không** ghi `server_port`; 7860 là mặc định phổ biến của Gradio, không phải số ghim trong repo.
8. Trang GitHub mặc định đang là nhánh `main`; README này được viết trên codebase đã có LaMa (M4). Nếu tải ZIP `main` trước khi merge, bước tải model / chọn `lama` có thể chưa có.
9. Có nên khuyến nghị một bản Python cụ thể (ví dụ 3.12) thay vì chỉ “3.10+” không? Repo không pin bản vá.
10. Hướng dẫn cài CUDA/driver GPU: **không có trong project** (tool chỉ tự dùng GPU nếu máy đã cài sẵn và onnxruntime thấy CUDA).
