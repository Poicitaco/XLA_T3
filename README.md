# XLA Face Privacy System

A privacy-first video surveillance system that anonymizes faces in real-time while preserving encrypted originals for authorized recovery.

**Core idea:** Every face is blurred by default. Raw footage is encrypted with AES-256-GCM and can only be decrypted by an admin with the correct password — giving you both privacy and accountability.

---

## Features

- **Real-time face detection** — YOLOv11-nano model, runs on CPU
- **Dual-layer storage** — blurred H.264 MP4 (public) + AES-256-GCM encrypted original (admin only)
- **Selective anonymization** — registered members can be shown unblurred; strangers are always blurred
- **Face recognition** — DeepFace (buffalo_l model) for member identification
- **Live streaming** — MJPEG, Server-Sent Events, WebSocket, or polling
- **Web dashboard** — admin panel (pipeline control, member registry, secure archive) + user portal (live view, clips)
- **REST API** — full FastAPI backend with interactive docs at `/docs`

---

## Project Structure

```
xla_demo/
├── src/
│   ├── app_api.py          # FastAPI application (all endpoints)
│   ├── clip_recorder.py    # Clip recording & AES-256-GCM encryption
│   ├── detector.py         # YOLOv11 face detector wrapper
│   ├── face_recognition.py # DeepFace member identification
│   ├── face_lock.py        # Per-face AES-GCM lock/unlock
│   ├── privacy.py          # Blur/overlay anonymization modes
│   ├── bbox_smoother.py    # Temporal bounding-box smoothing
│   ├── family_database.py  # Member CRUD (face embeddings)
│   ├── admin_auth.py       # Admin password hashing & verification
│   ├── anonymizer_backend.py
│   ├── backend_api.py
│   └── run_phone_cam.py    # Standalone CLI runner
├── frontend/
│   ├── index.html          # Landing page
│   ├── admin/
│   │   ├── dashboard.html      # Pipeline control & live view
│   │   ├── face-registry.html  # Member registration
│   │   └── secure-archive.html # Encrypted clip viewer & decryptor
│   ├── user/
│   │   ├── live-portal.html    # Live blurred stream
│   │   ├── clips.html          # Blurred clip history
│   │   └── support.html
│   └── assets/js/xla.js    # Shared API helpers
├── models/
│   └── yolov11n-face.pt    # YOLOv11-nano face detection weights
├── scripts/
│   └── registration_tool.py # CLI tool for registering members
├── evaluation/
│   ├── benchmark.py
│   ├── benchmark_privacy_modes.py
│   ├── security_evaluation.py
│   └── evaluate_anonymization_security.py
├── tests/
│   └── test_face_lock.py
├── data/
│   ├── admin_credentials.example.json
│   ├── family_members.example.json
│   ├── members/            # Member face photos (gitignored)
│   └── clips/              # Recorded clips (gitignored)
├── requirements.txt
└── start_server.ps1
```

---

## Quick Start

### 1. Install Python dependencies

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Install ffmpeg (required for clip recording)

**Windows** — install via winget or download from https://ffmpeg.org/download.html and add to PATH:

```powershell
winget install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
sudo apt install ffmpeg
```

### 3. First-startup — model auto-download

- **YOLOv11 face weights** (`models/yolov11n-face.pt`) — included in the repo, no download needed
- **InsightFace buffalo_l** (~300 MB) — downloaded automatically on first run into `~/.insightface/models/`, requires internet access

### 4. Start the server

```bash
uvicorn src.app_api:app --host 0.0.0.0 --port 8000 --reload
# or on Windows:
.\start_server.ps1
```

### 5. Open the dashboard

- Admin panel: http://localhost:8000/ui/admin/dashboard.html
- User portal: http://localhost:8000/ui/user/live-portal.html
- API docs: http://localhost:8000/docs

### 6. First-time setup

1. Go to the **Admin Dashboard** and set your admin password (stored as PBKDF2-SHA256 hash, never plaintext)
2. Go to **Face Registry** and register members by uploading photos
3. Start the pipeline — select camera source, anonymization mode, and enable clip recording

---

## API Overview

| Category      | Endpoints                                                                                                |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| **Health**    | `GET /health`, `GET /info`                                                                               |
| **Pipeline**  | `POST /pipeline/start`, `DELETE /pipeline/stop`, `GET /pipeline/status`, `PATCH /pipeline/config`        |
| **Streaming** | `GET /stream/mjpeg`, `GET /stream/sse`, `WS /stream/ws`, `GET /stream/frame`                             |
| **Members**   | `GET/POST /members`, `GET/PATCH/DELETE /members/{id}`                                                    |
| **Clips**     | `GET /clips`, `GET /clips/{id}/video`, `POST /clips/{id}/decrypt`, `DELETE /clips/{id}`, `DELETE /clips` |
| **Cameras**   | `GET /cameras`                                                                                           |
| **Image ops** | `POST /anonymize`, `POST /lock`, `POST /unlock`, `POST /recognize`                                       |

Full interactive docs: **http://localhost:8000/docs**

---

## Anonymization Modes

| Mode    | Description                                       |
| ------- | ------------------------------------------------- |
| `blur`  | Gaussian blur over detected faces                 |
| `solid` | Solid color block                                 |
| `noise` | Cryptographic noise pattern                       |
| `rps`   | Reversible Pixel Shuffling (recoverable with key) |

---

## Clip Recording & Encryption

When clip recording is enabled, every triggered clip is saved in two versions:

1. **Blurred MP4** (`data/clips/blurred/`) — H.264, browser-playable, faces anonymized
2. **Encrypted original** (`data/clips/secure/`) — AES-256-GCM, PBKDF2-derived key, unique 32-byte salt per clip

Decryption requires the admin password that was active when the pipeline recorded the clip. Wrong password → authentication tag mismatch → decryption fails.

---

## Selective Mode

Members registered in the face database can be excluded from anonymization:

| Mode             | Behavior                                            |
| ---------------- | --------------------------------------------------- |
| `disabled`       | All faces anonymized                                |
| `family_only`    | Registered members shown clearly, strangers blurred |
| `strangers_only` | Only unrecognized faces anonymized                  |

---

## Security Notes

- Admin credentials are stored as a salted PBKDF2-HMAC-SHA256 hash (390,000 iterations), never in plaintext
- Each clip uses a unique 32-byte random salt, so every encrypted file has an independent key
- All destructive operations (delete clips, decrypt) require both admin password and session token
- `data/admin_credentials.json`, `data/family_members.json`, and `data/members/` are gitignored to protect biometric data

---

## Requirements

- Python 3.10+
- ffmpeg in system PATH (used for H.264 transcoding)
- Webcam or IP camera (physical or virtual, e.g. Camo, OBS)

Key Python packages: `ultralytics`, `deepface`, `fastapi`, `uvicorn`, `cryptography`, `opencv-python`

> **Note:** The `insightface` buffalo_l model (~300 MB) is downloaded automatically on first run into `~/.insightface/models/`. Requires internet access on first startup.

---

## License

MIT
