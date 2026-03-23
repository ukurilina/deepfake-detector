import json
import time
import uuid
from pathlib import Path


from app.config import AppConfig


def _new_protect_id() -> str:
    return uuid.uuid4().hex


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _create_protect_dir(protect_id: str) -> str:
    base = Path(AppConfig.PROTECT_DIR)
    out_dir = base / protect_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def cleanup_protect_dir() -> int:
    """Remove expired protect artifacts (by meta.json timestamp)."""

    removed = 0
    base = Path(AppConfig.PROTECT_DIR)
    if not base.exists():
        return 0

    ttl = int(getattr(AppConfig, "PROTECT_TTL_SECONDS", 0) or 0)
    if ttl <= 0:
        return 0

    now_ts = time.time()

    for child in base.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        try:
            if not meta_path.exists():
                continue
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at = float(payload.get("created_at", 0.0))
            if created_at <= 0:
                continue
            if now_ts - created_at > ttl:
                for p in child.glob("**/*"):
                    try:
                        if p.is_file():
                            p.unlink(missing_ok=True)
                    except Exception:
                        pass
                try:
                    child.rmdir()
                except Exception:
                    pass
                removed += 1
        except Exception:
            continue

    return removed




