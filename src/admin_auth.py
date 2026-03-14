"""
Admin credential management.

Flow:
  1. POST /admin/setup         – first-time password setup
  2. POST /admin/verify        – verify password → bool
  3. POST /admin/change-password – update with old+new password

Passwords are NEVER stored on disk.  Only a PBKDF2-SHA256 hash
(salt + hash) is persisted in data/admin_credentials.json.

Encryption key derivation:
  derive_clip_key(password, salt) → 32-byte AES-256 key
  Each clip uses its own random 32-byte salt so every clip has a
  unique key even with the same password.
"""
from __future__ import annotations

import base64
import hmac
import json
import secrets
import time
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ─── tunables ────────────────────────────────────────────────────────────────
_ITERATIONS = 390_000        # OWASP 2023 minimum for PBKDF2-SHA256
_KEY_LEN     = 32            # 256-bit
_DEFAULT_CRED_PATH = "data/admin_credentials.json"


# ─── low-level helpers ────────────────────────────────────────────────────────

def _pbkdf2(password: str, salt: bytes, iterations: int = _ITERATIONS) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_LEN,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(password.encode("utf-8"))


def _ct_equal(a: bytes, b: bytes) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    return hmac.compare_digest(a, b)


# ─── public API ──────────────────────────────────────────────────────────────

def setup_admin(password: str, path: str = _DEFAULT_CRED_PATH) -> None:
    """
    Create admin credentials file.
    Raises ValueError if admin is already set up.
    """
    p = Path(path)
    if p.exists():
        raise ValueError("Admin is already set up. Use change_password() to update.")
    if len(password) < 8:
        raise ValueError("Admin password must be at least 8 characters.")
    p.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_bytes(32)
    digest = _pbkdf2(password, salt)
    p.write_text(
        json.dumps({
            "salt": base64.b64encode(salt).decode(),
            "hash": base64.b64encode(digest).decode(),
            "iterations": _ITERATIONS,
            "created_at": time.time(),
        }),
        encoding="utf-8",
    )


def is_admin_setup(path: str = _DEFAULT_CRED_PATH) -> bool:
    return Path(path).exists()


def verify_admin(password: str, path: str = _DEFAULT_CRED_PATH) -> bool:
    """Returns True iff password matches stored credentials."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        salt   = base64.b64decode(data["salt"])
        stored = base64.b64decode(data["hash"])
        iters  = int(data.get("iterations", _ITERATIONS))
        candidate = _pbkdf2(password, salt, iters)
        return _ct_equal(candidate, stored)
    except Exception:
        return False


def change_password(
    old_password: str,
    new_password: str,
    path: str = _DEFAULT_CRED_PATH,
) -> None:
    """Update admin password. old_password must match current credentials."""
    if not verify_admin(old_password, path):
        raise ValueError("Old password is incorrect.")
    if len(new_password) < 8:
        raise ValueError("New password must be at least 8 characters.")
    # Overwrite with new hash
    p = Path(path)
    salt   = secrets.token_bytes(32)
    digest = _pbkdf2(new_password, salt)
    p.write_text(
        json.dumps({
            "salt": base64.b64encode(salt).decode(),
            "hash": base64.b64encode(digest).decode(),
            "iterations": _ITERATIONS,
            "created_at": time.time(),
        }),
        encoding="utf-8",
    )


def derive_clip_key(password: str, salt: bytes) -> bytes:
    """
    Derive a 32-byte AES-256 key from admin password + per-clip random salt.
    Each clip uses secrets.token_bytes(32) as its salt so every clip has
    a unique key even if the password never changes.
    """
    return _pbkdf2(password, salt)
