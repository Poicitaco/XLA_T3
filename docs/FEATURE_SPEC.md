# XLA Face Privacy System — Đặc Tả Tính Năng & Ý Tưởng Dự Án

> **Phiên bản**: 1.0.0 | **Ngày cập nhật**: 14/03/2026  
> Tài liệu này mô tả toàn bộ ý tưởng, kiến trúc và tính năng của hệ thống để phục vụ thiết kế frontend.

---

## 1. Ý Tưởng & Bối Cảnh Dự Án

### 1.1 Vấn Đề Đặt Ra

Trong thời đại camera an ninh phủ khắp nơi, quyền riêng tư của khuôn mặt đang bị xâm phạm liên tục. Các hệ thống giám sát hiện tại có hai thái cực:

- **Ghi thô không che giấu** → lộ hoàn toàn danh tính mọi người
- **Xóa dữ liệu giám sát hoàn toàn** → mất bằng chứng khi cần điều tra

### 1.2 Giải Pháp: Privacy-First Surveillance

**XLA Face Privacy System** là hệ thống giám sát camera thông minh với triết lý: **"Bảo vệ quyền riêng tư theo mặc định, nhưng không mất đi bằng chứng"**.

Ý tưởng cốt lõi:

```
Camera → Phát hiện mặt → Ẩn danh hóa tự động → Lưu trữ
                                    ↓
                         Bản rõ (raw) → Mã hóa AES-256-GCM
                                    ↓
                         Chỉ Admin với mật khẩu riêng mới giải mã được
```

### 1.3 Hai Tầng Bảo Vệ

| Tầng                  | Mô tả                                         | Ai truy cập được            |
| --------------------- | --------------------------------------------- | --------------------------- |
| **Tầng 1: Blurred**   | Video đã làm mờ mặt, lưu dạng MP4 thường      | Tất cả (Member + Admin)     |
| **Tầng 2: Encrypted** | Video gốc mã hóa AES-256-GCM, lưu dạng `.enc` | Chỉ Admin với đúng mật khẩu |

### 1.4 Ứng Dụng Thực Tế

- 🏠 **Gia đình**: Camera nhà nhận ra người thân, che mặt khách/người lạ, lưu bằng chứng khi xảy ra sự cố
- 🏢 **Văn phòng**: Giám sát nhân viên theo policy, bảo vệ quyền riêng tư của khách thăm
- 🏥 **Cơ sở y tế**: Tuân thủ quy định HIPAA/GDPR về ẩn danh hóa bệnh nhân
- 🔬 **Nghiên cứu**: Thu thập dữ liệu hành vi mà không lưu danh tính

---

## 2. Kiến Trúc Hệ Thống

### 2.1 Tổng Quan

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND (SPA)                        │
│  Dashboard | Live View | Pipeline | Members | Clips | Admin  │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP REST + WebSocket
┌──────────────────────▼──────────────────────────────────────┐
│                    BACKEND (FastAPI)                          │
│                     Port 8000                                │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │   Pipeline  │  │  Image API  │  │   Admin & Clips    │  │
│  │   Session   │  │  (stateless │  │   (auth + storage) │  │
│  │ (background │  │   per-image)│  │                    │  │
│  │   thread)   │  └─────────────┘  └────────────────────┘  │
│  └──────┬──────┘                                            │
└─────────┼───────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│                    PROCESSING PIPELINE                        │
│                                                              │
│  Camera → FrameReader → YOLOv11 Detector → SimpleTracker    │
│               ↓                ↓                             │
│         InsightFace      anonymize_faces()                   │
│         Recognition      (blur/pixelate/...)                 │
│               ↓                ↓                             │
│         FamilyDatabase → ClipRecorder → data/clips/          │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Các File Chính

| File                        | Vai trò                                           |
| --------------------------- | ------------------------------------------------- |
| `src/app_api.py`            | FastAPI app — toàn bộ REST endpoints (~1700 dòng) |
| `src/run_phone_cam.py`      | FrameReader, SimpleTracker, RecognitionWorker     |
| `src/detector.py`           | YOLOv11 face detector wrapper                     |
| `src/face_recognition.py`   | InsightFace wrapper (buffalo_l, 512-dim)          |
| `src/family_database.py`    | FamilyDatabase — CRUD + vectorized matching       |
| `src/anonymizer_backend.py` | FaceAnonymizationEngine + PRESETS                 |
| `src/privacy.py`            | anonymize_faces() — 9 mode che mặt                |
| `src/face_lock.py`          | FaceRegionLocker — AES-GCM per-face (ảnh tĩnh)    |
| `src/admin_auth.py`         | PBKDF2-SHA256 admin credential management         |
| `src/clip_recorder.py`      | ClipRecorder — dual-storage blurred+encrypted     |
| `frontend/index.html`       | Dashboard SPA (7 tab)                             |
| `registration_tool.py`      | CLI tool đăng ký thành viên từ thư mục ảnh        |

### 2.3 Dữ Liệu Lưu Trữ

```
data/
├── family_members.json      # Face DB (member_id, name, threshold, embeddings[])
├── admin_credentials.json   # Mật khẩu admin (chỉ salt + hash, KHÔNG lưu plaintext)
└── clips/
    ├── blurred/             # *.mp4  — video đã làm mờ (công khai trong hệ thống)
    ├── secure/              # *.enc  — video gốc mã hóa AES-256-GCM (admin only)
    └── clips.json           # Metadata index: id, timestamp, fps, duration, frames
```

---

## 3. Tính Năng Chi Tiết

---

### 3.1 Nhóm: Thông Tin Hệ Thống

**Mục đích**: Cung cấp metadata cho frontend khởi tạo dropdown, options.

#### `GET /health`

- Kiểm tra server có hoạt động không
- Response: `{ ok, service, version }`
- **UI**: Indicator "🟢 Online / 🔴 Offline" ở header

#### `GET /info`

- Danh sách tất cả `anonymize_modes`, `selective_modes`, tên model mặc định, danh sách presets
- **UI**: Load 1 lần khi app khởi động để điền dropdown

#### `GET /presets`

- Config đầy đủ của 4 preset: `fast / balanced / strong / strict`
- **UI**: Khi user chọn preset → tự điền các tham số chi tiết vào form

#### `GET /pipeline/modes`

- Danh sách mode hợp lệ (giống /info nhưng gọn hơn)

---

### 3.2 Nhóm: Xử Lý Ảnh Tĩnh (Stateless)

**Mục đích**: Thử nghiệm ẩn danh hóa / nhận diện trên 1 ảnh upload, không cần camera.

#### `POST /anonymize`

Che mặt trong 1 ảnh.

| Tham số         | Kiểu   | Mô tả                                                  |
| --------------- | ------ | ------------------------------------------------------ |
| `image_b64`     | string | Ảnh base64 (JPEG/PNG)                                  |
| `preset`        | string | `fast \| balanced \| strong \| strict`                 |
| `config`        | object | Override tuỳ chọn (mode, blur_scale, pixel_block, ...) |
| `image_format`  | string | `jpg \| png`                                           |
| `image_quality` | int    | 50–100 (JPEG quality)                                  |

Response: `{ frame_b64, boxes[[x,y,w,h],...], scores, stats }`

**Anonymize modes** (`config.mode`):
| Mode | Hiệu ứng |
|------|----------|
| `blur` | Làm mờ Gaussian — phổ biến nhất |
| `pixelate` | Pixel hóa dạng mosaic |
| `solid` | Che bằng hình chữ nhật màu đặc |
| `obliterate` | Xóa vùng mặt, thay bằng màu nền |
| `headcloak` | Che toàn bộ đầu (kể cả phần trên tóc) |
| `silhouette` | Vẽ bóng đen dạng silhouette |
| `neckup` | Che từ cổ trở lên |
| `lock` | Mã hóa AES — dùng kết hợp với /lock endpoint |
| `none` | Không che, chỉ detect và trả về boxes |

#### `POST /lock` — Khóa mặt bằng AES-GCM

Mã hóa **reversible** từng vùng mặt riêng biệt.

| Tham số         | Mô tả                                                                          |
| --------------- | ------------------------------------------------------------------------------ |
| `lock_key`      | Passphrase người dùng nhập — dùng để tạo AES key                               |
| `lock_overlay`  | Hiệu ứng visual che mặt sau khi mã hóa: `solid \| noise \| ciphernoise \| rps` |
| `rps_tile_size` | Kích thước tile khi dùng overlay `rps` (Reversible Pixel Shuffling)            |

Response bao gồm `payloads[]` — **client PHẢI lưu lại** để unlock. Mỗi payload chứa ciphertext AES-GCM của 1 khuôn mặt.

**Lưu ý bảo mật**: Payloads không chứa key. Mất key = không khôi phục được.

#### `POST /unlock` — Giải khóa mặt

- Input: `image_b64` (ảnh đã khóa) + `payloads[]` (từ /lock) + `unlock_key`
- Output: `frame_b64` ảnh gốc được khôi phục hoàn toàn
- Nếu sai key → HTTP 400

#### `POST /recognize` — Nhận diện khuôn mặt trong ảnh tĩnh

Detect + nhận diện tất cả mặt trong 1 ảnh, so sánh với face database.

Response mỗi face:

```json
{
  "box": [x, y, w, h],
  "detection_score": 0.94,
  "member_id": "khoa",
  "name": "Khoa",
  "similarity": 0.72
}
```

- `member_id = null` + `name = null` → người lạ (không nhận ra)
- `similarity` là cosine similarity (0–1, càng cao càng chắc)

---

### 3.3 Nhóm: Camera

#### `GET /cameras`

Quét và liệt kê tất cả camera được kết nối với máy tính.

Response:

```json
{
  "cameras": [
    {
      "choice": 0,
      "name": "Integrated Webcam",
      "index": 0,
      "backend": "DSHOW"
    },
    {
      "choice": 1,
      "name": "OBS Virtual Camera",
      "index": 1,
      "backend": "DSHOW"
    }
  ],
  "total": 2
}
```

- `choice` — dùng làm `source_choice` khi start pipeline
- **UI**: Dropdown "Chọn camera" tự động populate từ endpoint này

---

### 3.4 Nhóm: Quản Lý Thành Viên (Face Database)

**Mục đích**: CRUD database khuôn mặt để pipeline nhận diện.

#### `GET /members`

Danh sách tất cả thành viên đã đăng ký.

```json
{
  "members": [
    {
      "member_id": "khoa",
      "name": "Khoa",
      "threshold": 0.35,
      "embedding_count": 25
    },
    {
      "member_id": "dat",
      "name": "Đạt",
      "threshold": 0.4,
      "embedding_count": 25
    }
  ],
  "total": 2,
  "model_name": "buffalo_l",
  "embedding_dim": 512
}
```

#### `GET /members/{member_id}`

Chi tiết 1 thành viên. HTTP 404 nếu không tồn tại.

#### `POST /members` — Đăng ký thành viên mới

**Content-Type**: `multipart/form-data`

| Field               | Bắt buộc | Mô tả                                     |
| ------------------- | -------- | ----------------------------------------- |
| `member_id`         | ✅       | ID cố định, không dấu cách (vd: `"khoa"`) |
| `name`              | ✅       | Tên hiển thị (vd: `"Khoa"`)               |
| `images[]`          | ✅       | 1 hoặc nhiều file ảnh JPEG/PNG            |
| `threshold`         | ❌       | Ngưỡng nhận diện riêng (mặc định 0.35)    |
| `max_samples`       | ❌       | Số embedding tối đa giữ lại (mặc định 25) |
| `recognition_model` | ❌       | `buffalo_l` (mặc định)                    |

Nếu `member_id` đã tồn tại → **thay thế embeddings** (upsert behavior).

**Lưu ý**: Sau khi đăng ký thành công, pipeline đang chạy sẽ **tự động reload DB ngay lập tức**.

#### `PATCH /members/{member_id}`

Sửa tên hoặc ngưỡng nhận diện:

```json
{ "name": "Nguyễn Văn Khoa", "threshold": 0.4 }
```

#### `DELETE /members/{member_id}`

Xóa thành viên. Pipeline tự reload sau khi xóa.

---

### 3.5 Nhóm: Live Pipeline

**Đây là tính năng cốt lõi** — pipeline camera chạy nền, xử lý liên tục.

**Nguyên tắc**: Chỉ **một pipeline** tồn tại tại một thời điểm. Đang chạy mà gọi start → HTTP 409.

#### `POST /pipeline/start`

Tham số đầy đủ:

**Camera**:
| Field | Mặc định | Mô tả |
|-------|----------|-------|
| `source_choice` | — | Index từ GET /cameras (khuyến nghị) |
| `backend` | `dshow` | Windows: `dshow \| msmf \| any` |
| `camera_width` | 1280 | Độ rộng capture |
| `camera_height` | 720 | Độ cao capture |
| `camera_fps` | 30 | FPS mong muốn |

**Detector**:
| Field | Mặc định | Mô tả |
|-------|----------|-------|
| `detect_every` | 4 | Chạy YOLO mỗi N frame (1=mỗi frame, tốn CPU; 4=cân bằng) |
| `conf_threshold` | 0.35 | Ngưỡng tin cậy detector |
| `proc_width` | 640 | Co frame về chiều rộng này trước khi detect (0=tắt) |

**Ẩn danh hóa**:
| Field | Mặc định | Mô tả |
|-------|----------|-------|
| `anonymize_mode` | `blur` | Xem bảng mode ở mục 3.2 |
| `selective_mode` | `disabled` | Xem bảng selective mode bên dưới |
| `face_padding` | 0.12 | Vùng đệm xung quanh mặt (12%) |

**Nhận diện & Tracking**:
| Field | Mặc định | Mô tả |
|-------|----------|-------|
| `recognition_threshold` | 0.35 | Ngưỡng cosine similarity |
| `reid_min_interval` | 12 | Min frames giữa 2 lần nhận diện lại (tránh spam) |
| `reid_max_interval` | 90 | Bắt buộc nhận diện lại sau N frames |

**Ghi clip**:
| Field | Mặc định | Mô tả |
|-------|----------|-------|
| `enable_clip_recording` | false | Bật ghi clip khi phát hiện mặt |
| `admin_password` | — | **Bắt buộc** khi `enable_clip_recording=true` |
| `clip_duration_s` | 8 | Độ dài mỗi clip (2–60 giây) |
| `clip_pre_buffer_s` | 3 | Buffer trước khi phát hiện (rolling buffer 0–15s) |
| `clip_cooldown_s` | 15 | Thời gian nghỉ tối thiểu giữa 2 clip liên tiếp |

**Selective Mode** — quyết định AI che mặt **ai**:

| Mode              | Hành vi                               | Use case                     |
| ----------------- | ------------------------------------- | ---------------------------- |
| `disabled`        | Che **TẤT CẢ** mặt phát hiện được     | Ẩn danh hoàn toàn            |
| `family`          | Chỉ che mặt **thành viên đã đăng ký** | Bảo vệ gia đình, lộ người lạ |
| `stranger`        | Chỉ che mặt **người lạ**              | Hiện gia đình, ẩn khách      |
| `registered_only` | Tương tự `family`                     | ―                            |

#### `DELETE /pipeline/stop`

Dừng pipeline, giải phóng camera, flush clip recorder nếu đang ghi.

#### `GET /pipeline/status`

Trạng thái real-time (nên poll mỗi 2–3 giây):

```json
{
  "running": true,
  "uptime_s": 183.4,
  "frame_idx": 5501,
  "fps": 27.8,
  "faces_detected": 2,
  "registered_count": 1,
  "unknown_count": 1,
  "anonymize_mode": "blur",
  "selective_mode": "family",
  "source_info": "choice=0 name=Integrated Webcam",
  "has_frame": true,
  "clip_recording": true,
  "clips_saved": 3
}
```

#### `PATCH /pipeline/config` — Hot-update (không cần restart)

Thay đổi có hiệu lực ngay frame tiếp theo:

```json
{
  "anonymize_mode": "pixelate",
  "selective_mode": "stranger",
  "detection_every": 2,
  "stream_jpeg_quality": 60,
  "stream_max_fps": 15
}
```

**Không thể hot-update**: camera source, YOLO model, InsightFace model → cần stop + start lại.

---

### 3.6 Nhóm: Live Streaming

Pipeline chạy → các giá trị frame được lưu trong `_FrameBuffer` → client lấy theo cách tuỳ chọn:

| Endpoint            | Giao thức          | Cách dùng                   | Đặc điểm                    |
| ------------------- | ------------------ | --------------------------- | --------------------------- |
| `GET /stream/frame` | HTTP polling       | Gọi lại định kỳ, nhận JSON  | Đơn giản, mọi client        |
| `GET /stream/mjpeg` | HTTP MJPEG         | `<img src="/stream/mjpeg">` | **Simplest** cho browser    |
| `GET /stream/sse`   | Server-Sent Events | `new EventSource(url)`      | Nhận frame + stats cùng lúc |
| `WS /stream/ws`     | WebSocket          | `new WebSocket(url)`        | Latency thấp nhất           |

**WebSocket params**: `?format=binary` (raw JPEG bytes) hoặc `?format=json` (JSON với frame_b64)

**SSE example** (JavaScript):

```js
const es = new EventSource("/stream/sse");
es.onmessage = (e) => {
  const { frame_b64, stats } = JSON.parse(e.data);
  document.getElementById("feed").src = "data:image/jpeg;base64," + frame_b64;
  // stats: { frame_idx, fps, faces_detected }
};
```

**Stats trong mỗi frame**: `{ frame_idx, fps, faces_detected }`

**Khuyến nghị frontend**:

- Dùng `<img src="/stream/mjpeg">` cho live view → đơn giản nhất, browser xử lý native
- Dùng SSE nếu cần cập nhật stats đồng bộ với frame

---

### 3.7 Nhóm: Admin Management

**Bảo mật**: Mật khẩu được hash bằng **PBKDF2-SHA256 với 390.000 vòng lặp** + random salt. **Không lưu plaintext bao giờ.**

**Xác thực**: Hệ thống dùng **session token tạm thời** (sống 2 tiếng) thay vì bắt frontend gửi password liên tục. Sau khi `POST /admin/verify` thành công, frontend nhận 1 token ngẫu nhiên 32-byte (URL-safe base64) và lưu vào `sessionStorage`. Các thao tác admin tiếp theo gửi `token` thay vì `password`.

```
POST /admin/verify { password }  →  { token, expires_at }   ← lưu vào sessionStorage

// Các request sau đó:
DELETE /clips/{id}               { token: "..." }           ← không cần gửi password
DELETE /clips                    { token: "..." }
POST /clips/{id}/decrypt         { password, token }        ← password vẫn cần cho AES key

POST /admin/logout               { token }                  ← thu hồi ngay lập tức
```

**Lưu ý đặc biệt về `/clips/{id}/decrypt`**: endpoint này yêu cầu `password` dù đã có token, vì password được dùng trực tiếp để derive AES key (PBKDF2) giải mã video. Token chỉ bypass bước xác thực admin, không thay thế được vai trò key material.

#### `POST /admin/setup`

Thiết lập mật khẩu admin **lần đầu tiên**.

- Body: `{ "password": "..." }` (tối thiểu 8 ký tự)
- Chỉ thành công 1 lần duy nhất → sau đó dùng `/admin/change-password`
- Trả về HTTP 409 nếu đã setup rồi

#### `GET /admin/status`

Kiểm tra đã setup admin chưa: `{ "admin_setup": true/false }`

**UI flow**: Khi app load lần đầu, gọi endpoint này. Nếu `false` → hiện dialog "Thiết lập mật khẩu admin".

#### `POST /admin/verify`

Xác minh mật khẩu và nhận **session token**:

```json
// Request
{ "password": "my_secret" }

// Response 200
{
  "ok": true,
  "token": "xRt9kL2...",
  "expires_at": "2026-03-14T16:30:00+00:00",
  "ttl_hours": 2.0
}
```

- Token hết hạn sau 2 tiếng từ lúc tạo
- Gọi lại `POST /admin/verify` bất kỳ lúc nào để lấy token mới
- `401 Unauthorized` nếu sai mật khẩu

#### `POST /admin/logout`

Thu hồi token ngay lập tức:

```json
{ "token": "xRt9kL2..." }
```

**UI**: Nút "Đăng xuất Admin" xóa token khỏi sessionStorage và gọi endpoint này.

#### `POST /admin/change-password`

```json
{ "old_password": "...", "new_password": "..." }
```

- HTTP 400 nếu mật khẩu cũ sai hoặc mật khẩu mới quá ngắn
- Sau khi đổi mật khẩu, toàn bộ token cũ vẫn còn hiệu lực đến khi hết hạn (2h)

---

### 3.8 Nhóm: Clips (Video Bằng Chứng)

**Cơ chế tự động**: Khi pipeline chạy với `enable_clip_recording=true`, hệ thống:

1. Duy trì rolling pre-buffer (mặc định 3 giây)
2. Khi phát hiện mặt → trigger lưu clip
3. Clip gồm: N giây pre-buffer + N giây sau trigger
4. Lưu **đồng thời** 2 bản:
   - `blurred/*.mp4` — frame đã fake blur
   - `secure/*.enc` — frame gốc, AES-256-GCM encrypted

**Định dạng file `.enc`**:

```
[32 bytes: random salt][12 bytes: AES nonce][ciphertext AES-GCM...]
```

Key được derive từ mật khẩu admin + salt duy nhất per clip (PBKDF2).

#### `GET /clips`

Danh sách tất cả clip đã lưu:

```json
{
  "clips": [
    {
      "id": "clip_20260314_143052",
      "timestamp": "2026-03-14T14:30:52",
      "created": "2026-03-14 14:30:52",
      "frames": 240,
      "fps": 30.0,
      "duration_s": 8.0
    }
  ],
  "total": 1
}
```

#### `GET /clips/{id}/video`

Stream MP4 đã làm mờ — xem trực tiếp trong browser (`<video>` tag).

#### `GET /clips/{id}/thumbnail`

JPEG frame đầu tiên của clip — dùng làm thumbnail trong grid.

#### `POST /clips/{id}/decrypt` — Admin only

```json
{ "password": "admin_password_here" }
```

- Xác thực mật khẩu → decrypt AES-GCM → trả về MP4 gốc để download
- HTTP 401 nếu sai mật khẩu
- HTTP 404 nếu không có bản encrypted (clip cũ hoặc bị xóa)

#### `DELETE /clips/{id}` — Admin only

```json
{ "password": "admin_password_here" }
```

Xóa cả 2 file (blurred MP4 + encrypted raw) và entry trong index.

#### `DELETE /clips` — Admin only

Xóa **tất cả** clip. Trả về số clip đã xóa.

---

## 4. Bảng Quyền Hạn Đầy Đủ

| Tính năng                                        |  Member (thường)  | Admin |
| ------------------------------------------------ | :---------------: | :---: |
| Xem live stream                                  |        ✅         |  ✅   |
| Điều khiển pipeline (start/stop)                 |        ✅         |  ✅   |
| Hot-update config pipeline                       |        ✅         |  ✅   |
| Xử lý ảnh tĩnh (anonymize/lock/unlock/recognize) |        ✅         |  ✅   |
| Quản lý thành viên (CRUD)                        |        ✅         |  ✅   |
| Xem danh sách clip                               |        ✅         |  ✅   |
| Xem clip đã làm mờ (blurred MP4)                 |        ✅         |  ✅   |
| Xem thumbnail clip                               |        ✅         |  ✅   |
| **Bật ghi clip khi start pipeline**              | ❌ (cần password) |  ✅   |
| **Giải mã xem clip gốc (unblurred)**             |        ❌         |  ✅   |
| **Xóa clip**                                     |        ❌         |  ✅   |
| **Đổi mật khẩu admin**                           |        ❌         |  ✅   |
| Thiết lập mật khẩu admin lần đầu                 |        N/A        |  ✅   |

---

## 5. Toàn Bộ API Endpoints

### Base URL: `http://localhost:8000`

| Method | Path                     | Tag              | Mô tả                              |
| ------ | ------------------------ | ---------------- | ---------------------------------- |
| GET    | `/`                      | —                | Redirect về dashboard (index.html) |
| GET    | `/health`                | Info             | Server alive check                 |
| GET    | `/info`                  | Info             | Capabilities, modes, presets       |
| GET    | `/presets`               | Info             | Chi tiết config các preset         |
| GET    | `/pipeline/modes`        | Info             | Các mode hợp lệ                    |
| POST   | `/anonymize`             | Image Processing | Ẩn danh ảnh tĩnh                   |
| POST   | `/lock`                  | Image Processing | Khóa mặt AES-GCM reversible        |
| POST   | `/unlock`                | Image Processing | Giải khóa mặt                      |
| POST   | `/recognize`             | Recognition      | Nhận diện khuôn mặt trong ảnh      |
| GET    | `/cameras`               | Cameras          | Liệt kê camera trên máy            |
| GET    | `/members`               | Members          | Danh sách thành viên               |
| GET    | `/members/{id}`          | Members          | Chi tiết 1 thành viên              |
| POST   | `/members`               | Members          | Đăng ký thành viên mới             |
| PATCH  | `/members/{id}`          | Members          | Sửa tên / threshold                |
| DELETE | `/members/{id}`          | Members          | Xóa thành viên                     |
| POST   | `/pipeline/start`        | Pipeline         | Khởi động camera pipeline          |
| DELETE | `/pipeline/stop`         | Pipeline         | Dừng pipeline                      |
| GET    | `/pipeline/status`       | Pipeline         | Trạng thái + stats real-time       |
| PATCH  | `/pipeline/config`       | Pipeline         | Hot-update settings                |
| GET    | `/stream/frame`          | Streaming        | Polling — JSON frame base64        |
| GET    | `/stream/mjpeg`          | Streaming        | MJPEG stream cho `<img>` tag       |
| GET    | `/stream/sse`            | Streaming        | Server-Sent Events                 |
| WS     | `/stream/ws`             | Streaming        | WebSocket binary/json              |
| POST   | `/admin/setup`           | Admin            | Setup mật khẩu lần đầu             |
| GET    | `/admin/status`          | Admin            | Đã setup chưa?                     |
| POST   | `/admin/verify`          | Admin            | Xác minh → nhận session token (2h) |
| POST   | `/admin/logout`          | Admin            | Thu hồi session token              |
| POST   | `/admin/change-password` | Admin            | Đổi mật khẩu                       |
| GET    | `/clips`                 | Clips            | Danh sách clip                     |
| GET    | `/clips/{id}/video`      | Clips            | Stream blurred MP4                 |
| GET    | `/clips/{id}/thumbnail`  | Clips            | JPEG thumbnail                     |
| POST   | `/clips/{id}/decrypt`    | Clips            | Decrypt original (admin)           |
| DELETE | `/clips/{id}`            | Clips            | Xóa clip (admin)                   |
| DELETE | `/clips`                 | Clips            | Xóa tất cả clip (admin)            |

**Interactive docs**: `http://localhost:8000/docs` (Swagger UI tự động)

---

## 6. Gợi Ý Thiết Kế Frontend

### 6.1 Cấu Trúc Trang Đề Xuất

```
App
├── Header: logo + server status + admin mode indicator
├── Sidebar / Tabs:
│   ├── 📺  Live View
│   ├── ⚙️  Pipeline Config
│   ├── 🔍  Image Tools
│   │     ├── Anonymize
│   │     ├── Lock / Unlock
│   │     └── Recognize
│   ├── 👥  Members
│   ├── 🎬  Clips
│   └── 🔐  Admin
└── Footer: version info
```

### 6.2 Live View

- `<img src="/stream/mjpeg">` cho video feed
- Overlay stats realtime: FPS, số mặt, mode đang dùng
- Quick-toggle buttons: anonymize_mode và selective_mode (→ `PATCH /pipeline/config`)
- Start/Stop pipeline button với status indicator

### 6.3 Pipeline Config

- Form đầy đủ `PipelineStartRequest`
- Camera dropdown tự động load từ `GET /cameras`
- Preset selector → tự điền tham số
- Slider cho các tham số visual (blur_scale, pixel_block, ...)
- Section "Ghi clip" ẩn/hiện khi toggle `enable_clip_recording`

### 6.4 Image Tools

- Drag & drop ảnh
- Before/After view
- Lock tool: ô nhập passphrase + lưu payloads để unlock sau

### 6.5 Members

- Grid thẻ thành viên (avatar thumbnail + tên + số embedding)
- Modal thêm: drag drop nhiều ảnh + input ID + tên + threshold slider
- Inline edit tên và threshold
- Confirm dialog khi xóa

### 6.6 Clips

- Grid thumbnail (ảnh đầu tiên)
- Click để xem `<video>` blurred inline
- Badge thời gian + duration
- Nếu đang ở Admin mode: nút "🔓 Decrypt & Download" + nút xóa

### 6.7 Admin

- Onboarding: nếu chưa setup → form tạo mật khẩu lần đầu
- Login form → `POST /admin/verify` → lưu **token** vào `sessionStorage` (không lưu password)
- Tất cả thao tác admin tiếp theo gửi `{ token: "..." }` thay vì password
- Badge "🔐 Admin" trên header khi đang đăng nhập + nút đăng xuất (`POST /admin/logout` → xóa token)
- Form đổi mật khẩu (vẫn cần nhập mật khẩu cũ)
- Token tự hết hạn sau 2 tiếng — UI nên bắt HTTP 401 và hiện lại login form

### 6.8 Lưu Ý Kỹ Thuật

```
⚠️  Tất cả ảnh truyền qua API = base64 string
    (trừ POST /members dùng multipart/form-data)

⚠️  Admin auth: POST /admin/verify → nhận token → lưu sessionStorage
    Gửi { token } thay vì password cho DELETE clips, v.v.
    Riêng POST /clips/{id}/decrypt vẫn cần password (dùng để derive AES key)

⚠️  Token hết hạn sau 2h — bắt HTTP 401 và hiện lại login form

⚠️  Pipeline singleton — đang chạy mà gọi start → HTTP 409
    → UI cần kiểm tra status trước khi cho bấm Start

⚠️  PATCH /pipeline/config không cần restart
    → Các slider mode/quality có thể update live ngay khi drag

⚠️  POST /members trigger hot-reload pipeline
    → Không cần báo user "restart pipeline"
```

---

## 7. Flow Sử Dụng Điển Hình

### Lần đầu cài đặt

```
1. GET /admin/status → false
2. POST /admin/setup { password: "..." }
3. GET /cameras → chọn camera
4. POST /members (đăng ký gia đình)
5. POST /pipeline/start { enable_clip_recording: true, admin_password: "..." }
6. Xem live stream
```

### Xem lại bằng chứng có clip

```
1. GET /clips → danh sách
2. GET /clips/{id}/video → xem bản đã blur
3. POST /clips/{id}/decrypt { password: "..." } → tải bản gốc
```

### Thêm thành viên mới trong khi camera đang chạy

```
1. POST /members (upload ảnh)
2. Pipeline tự reload DB → nhận ra ngay lập tức
3. Không cần stop/start pipeline
```

---

> **Tài liệu đầy đủ hơn**: Xem Swagger UI tại `http://localhost:8000/docs`  
> **Mã nguồn**: `src/app_api.py` — toàn bộ implementation ~1700 dòng
