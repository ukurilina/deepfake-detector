import os
from typing import Any, Dict, Optional

import httpx

from app.config import AppConfig


class AntiFakeServiceError(RuntimeError):
    pass


def _get_antifake_service_url() -> str:
    return (
        getattr(AppConfig, "ANTIFAKE_SERVICE_URL", None)
        or os.getenv("ANTIFAKE_SERVICE_URL")
        or "http://localhost:8010"
    ).rstrip("/")


def protect_audio_via_service(input_path: str, output_wav_path: str, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
    """Send audio file to the AntiFake microservice and write protected wav output.

    This keeps the main backend on Python 3.12 while AntiFake runs on Python 3.10.
    """

    base_url = _get_antifake_service_url()

    # AntiFake can take a long time; use explicit timeouts.
    # `timeout_seconds` controls overall READ timeout (most common failure mode).
    read_timeout = timeout_seconds
    if read_timeout is None:
        read_timeout = float(getattr(AppConfig, "ANTIFAKE_SERVICE_TIMEOUT_SECONDS", 1800.0))

    timeout = httpx.Timeout(
        connect=10.0,
        read=read_timeout,
        write=60.0,
        pool=10.0,
    )

    with open(input_path, "rb") as f:
        files = {"file": (os.path.basename(input_path) or "input", f, "application/octet-stream")}
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(f"{base_url}/v1/protect", files=files)
        except httpx.ReadTimeout as exc:
            raise AntiFakeServiceError(
                f"AntiFake service request failed: {exc!r}. "
                f"Hint: increase ANTIFAKE_SERVICE_TIMEOUT_SECONDS (current={read_timeout})."
            )
        except Exception as exc:
            raise AntiFakeServiceError(f"AntiFake service request failed: {exc!r}")

    if resp.status_code != 200:
        detail = None
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise AntiFakeServiceError(f"AntiFake service error: HTTP {resp.status_code}; detail={detail}")

    # Response is raw wav bytes.
    content_type = resp.headers.get("content-type", "")
    if "audio" not in content_type and "octet-stream" not in content_type:
        # Still write it; but return metadata for debugging.
        pass

    with open(output_wav_path, "wb") as out:
        out.write(resp.content)

    meta: Dict[str, Any] = {
        "service_url": base_url,
        "bytes": len(resp.content),
    }

    # Optional metadata headers
    if "x-antifake-duration" in resp.headers:
        try:
            meta["duration"] = float(resp.headers["x-antifake-duration"])
        except Exception:
            pass

    return meta


