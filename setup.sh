#!/usr/bin/env bash
# XLA Face Privacy System — One-click Linux/macOS setup & launcher
# Usage: bash setup.sh

set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
step()  { echo -e "\n${CYAN}==> $1${NC}"; }
ok()    { echo -e "    ${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "    ${YELLOW}[!!]${NC} $1"; }
fail()  { echo -e "\n${RED}[FAIL]${NC} $1"; exit 1; }

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ─── 1. Check Python ─────────────────────────────────────────────────────────
step "Checking Python 3.10+..."
PY=""
for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
        if [[ "$ver" == *"(3, 1"* ]] || [[ "$ver" == *"(3, 1"[0-9]*")"* ]]; then
            PY="$cmd"; break
        fi
    fi
done

if [[ -z "$PY" ]]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        warn "Python 3.10+ not found. Attempting install via Homebrew..."
        command -v brew &>/dev/null || fail "Homebrew not found. Install from https://brew.sh then re-run."
        brew install python@3.11
        PY="python3.11"
    else
        fail "Python 3.10+ not found. Install with: sudo apt install python3.11 python3.11-venv"
    fi
fi
ok "Using: $($PY --version)"

# ─── 2. Create virtual environment ──────────────────────────────────────────
step "Setting up virtual environment..."
VENV="$ROOT/.venv"
if [[ ! -d "$VENV" ]]; then
    "$PY" -m venv "$VENV"
    ok "Created .venv"
else
    ok ".venv already exists"
fi

PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

# ─── 3. Install dependencies ─────────────────────────────────────────────────
step "Installing Python packages (this takes a few minutes on first run)..."
"$PIP" install --upgrade pip --quiet
"$PIP" install -r "$ROOT/requirements.txt"
ok "All packages installed."

# ─── 4. Check ffmpeg ─────────────────────────────────────────────────────────
step "Checking ffmpeg..."
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg found."
else
    warn "ffmpeg not found — clip recording will fall back to mp4v codec."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        warn "To install: brew install ffmpeg"
    else
        warn "To install: sudo apt install ffmpeg"
    fi
fi

# ─── 5. Ensure data directories ──────────────────────────────────────────────
step "Preparing data directories..."
mkdir -p "$ROOT/data/clips/blurred" "$ROOT/data/clips/secure" "$ROOT/data/members"
ok "Data directories ready."

# ─── 6. Start server ─────────────────────────────────────────────────────────
step "Starting XLA server on http://localhost:8000 ..."
echo ""
echo -e "  ${YELLOW}Admin dashboard${NC} : http://localhost:8000/ui/admin/dashboard.html"
echo -e "  ${YELLOW}User portal    ${NC} : http://localhost:8000/ui/user/live-portal.html"
echo -e "  ${YELLOW}API docs       ${NC} : http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop the server."
echo ""

cd "$ROOT"
exec "$PYTHON" -m uvicorn src.app_api:app --host 0.0.0.0 --port 8000 --reload
