import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse


# Repo root is 3 levels above this file:
#   services/antifake_service/app/main.py -> app -> antifake_service -> services -> <repo_root>
BASE_DIR = Path(__file__).resolve().parents[3]
ANTIFAKE_DIR = Path(os.getenv("ANTIFAKE_DIR", str(BASE_DIR / "thirdparty" / "AntiFake"))).resolve()
ANTIFAKE_RUN_PY = Path(os.getenv("ANTIFAKE_RUN_PY", str(ANTIFAKE_DIR / "run.py"))).resolve()

ANTIFAKE_MAX_SECONDS = float(os.getenv("ANTIFAKE_MAX_SECONDS", "20"))
ANTIFAKE_FORCE_CPU = os.getenv("ANTIFAKE_FORCE_CPU", "1") == "1"
ANTIFAKE_RANDOM_TARGETS = int(os.getenv("ANTIFAKE_RANDOM_TARGETS", "8"))

# Required external weights (per AntiFake README)
ANTIFAKE_REQUIRED_FILES = {
    "tortoise_autoregressive": str(ANTIFAKE_DIR / "tortoise" / "autoregressive.pth"),
    "tortoise_diffusion_decoder": str(ANTIFAKE_DIR / "tortoise" / "diffusion_decoder.pth"),
    "rtvc_encoder": str(ANTIFAKE_DIR / "saved_models" / "default" / "encoder.pt"),
    "rtvc_vocoder": str(ANTIFAKE_DIR / "saved_models" / "default" / "vocoder.pt"),
}


app = FastAPI(title="AntiFake Service", version="1.0.0")


def _get_duration_seconds(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return float(frames) / float(rate)


def _convert_to_antifake_input_wav(src_path: str, dst_wav_path: str) -> Dict[str, Any]:
    """Convert arbitrary audio -> WAV PCM16, mono, 16k."""
    try:
        audio, sr = sf.read(src_path, always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
        sf.write(dst_wav_path, audio, sr, subtype="PCM_16")
        return {"sample_rate": sr, "backend": "soundfile/librosa"}
    except Exception as exc:
        raise RuntimeError(f"Failed to convert audio for AntiFake: {exc!r}")


def _validate_antifake_installation() -> Tuple[bool, str]:
    if not ANTIFAKE_DIR.is_dir():
        return False, f"AntiFake directory not found: {ANTIFAKE_DIR}"
    if not ANTIFAKE_RUN_PY.is_file():
        return False, f"AntiFake run.py not found: {ANTIFAKE_RUN_PY}"

    missing = [k for k, p in ANTIFAKE_REQUIRED_FILES.items() if not os.path.exists(p)]
    if missing:
        missing_paths = {k: ANTIFAKE_REQUIRED_FILES.get(k) for k in missing}
        return (
            False,
            "AntiFake required weight files are missing: "
            + ", ".join(missing)
            + ".\nExpected paths:\n"
            + "\n".join([f"- {k}: {v}" for k, v in missing_paths.items()]),
        )

    return True, "ok"


def _patch_antifake_run_py(tmp_dir: str) -> str:
    """Patch AntiFake run.py to remove interactive pygame scoring."""

    text = ANTIFAKE_RUN_PY.read_text(encoding="utf-8", errors="ignore")

    # AntiFake upstream imports pygame for interactive target selection.
    # In this integration we never use pygame; remove the import to avoid
    # requiring pygame (often problematic in headless environments).
    text = text.replace("import pygame\n", "")
    text = text.replace("import pygame\r\n", "")

    if ANTIFAKE_FORCE_CPU:
        text = text.replace('DEVICE = "cuda"', 'DEVICE = "cpu"')
        text = text.replace("DEVICE = 'cuda'", "DEVICE = 'cpu'")
        text = text.replace(
            "spectrogram = torchaudio.transforms.Spectrogram().cuda()",
            "spectrogram = torchaudio.transforms.Spectrogram().to(DEVICE)",
        )

    start_anchor = "pygame.mixer.init()"
    end_anchor = "# Compute source and target embedding differences"
    if start_anchor not in text or end_anchor not in text:
        raise RuntimeError("AntiFake run.py patch failed: expected anchors were not found")

    lines = text.splitlines(keepends=True)
    start_line_idx = None
    end_line_idx = None
    for i, line in enumerate(lines):
        if start_line_idx is None and start_anchor in line:
            start_line_idx = i
        if end_line_idx is None and end_anchor in line:
            end_line_idx = i
        if start_line_idx is not None and end_line_idx is not None:
            break

    if start_line_idx is None or end_line_idx is None or end_line_idx <= start_line_idx:
        raise RuntimeError("AntiFake run.py patch failed: could not locate anchor lines")

    indent = lines[start_line_idx][: len(lines[start_line_idx]) - len(lines[start_line_idx].lstrip(" \t"))]
    replacement_lines = [
        f"{indent}# User listens to source and targets, assign score to each\n",
        f"{indent}# (patched out in deepfake_detector antifake_service: non-interactive selection)\n",
        f"{indent}user_scores = [3 for _ in target_speakers_selected]\n",
        "\n",
    ]

    patched_lines = lines[:start_line_idx] + replacement_lines + lines[end_line_idx:]
    text = "".join(patched_lines)

    text = text.replace("NUM_RANDOM_TARGET_SPEAKER = 24", f"NUM_RANDOM_TARGET_SPEAKER = {ANTIFAKE_RANDOM_TARGETS}")

    out_path = Path(tmp_dir) / "run_patched.py"
    out_path.write_text(text, encoding="utf-8")
    compile(text, str(out_path), "exec")
    return str(out_path)


@app.get("/health")
def health():
    ok, msg = _validate_antifake_installation()
    return {
        "status": "ok" if ok else "error",
        "detail": msg,
        "antifake_dir": str(ANTIFAKE_DIR),
        "antifake_run_py": str(ANTIFAKE_RUN_PY),
    }


@app.post("/v1/protect")
async def protect(file: UploadFile = File(...)):
    ok, msg = _validate_antifake_installation()
    if not ok:
        raise HTTPException(status_code=503, detail=msg)

    work_dir = tempfile.mkdtemp(prefix="antifake_service_")
    try:
        in_path = os.path.join(work_dir, file.filename or "input")
        with open(in_path, "wb") as f:
            f.write(await file.read())

        src_wav = os.path.join(work_dir, "input.wav")
        conv_meta = _convert_to_antifake_input_wav(in_path, src_wav)

        duration = _get_duration_seconds(src_wav)
        if duration > ANTIFAKE_MAX_SECONDS:
            raise HTTPException(status_code=400, detail=f"Audio too long ({duration:.1f}s). Max {ANTIFAKE_MAX_SECONDS:.0f}s")

        out_wav = os.path.join(work_dir, "output.wav")
        patched_run = _patch_antifake_run_py(work_dir)

        cmd = [sys.executable, patched_run, src_wav, out_wav]
        env = os.environ.copy()
        # Ensure local AntiFake packages (adaptive_voice_conversion, rtvc, etc.) are importable.
        env["PYTHONPATH"] = str(ANTIFAKE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.run(
            cmd,
            cwd=str(ANTIFAKE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "AntiFake failed",
                    "stdout_tail": (proc.stdout or "")[-2000:],
                    "stderr_tail": (proc.stderr or "")[-2000:],
                },
            )

        if not os.path.exists(out_wav):
            raise HTTPException(status_code=500, detail="AntiFake completed but output was not created")

        data = Path(out_wav).read_bytes()

        headers = {
            "X-AntiFake-Duration": str(duration),
            "X-AntiFake-Converted": str(conv_meta.get("backend", "")),
        }
        return Response(content=data, media_type="audio/wav", headers=headers)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception):
    # Prevent leaking internal tracebacks but keep a usable error.
    return JSONResponse(status_code=500, content={"error": repr(exc)})






