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

from app.config import AppConfig


def _resolve_ffmpeg_executable() -> str:
    """Return ffmpeg executable path/name.

    On Windows ffmpeg is often not on PATH, so we allow configuring it.
    """

    return (getattr(AppConfig, "FFMPEG_PATH", None) or os.getenv("FFMPEG_PATH") or "ffmpeg")


def _get_duration_seconds(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return float(frames) / float(rate)


def _convert_to_antifake_input_wav(src_path: str, dst_wav_path: str) -> Dict[str, Any]:
    """Convert arbitrary audio to: WAV PCM16, mono, 16k.

    Uses soundfile where possible; falls back to librosa for resampling.
    """

    # Read with soundfile if possible
    try:
        audio, sr = sf.read(src_path, always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # Resample if needed
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        # Normalize to [-1, 1]
        audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
        sf.write(dst_wav_path, audio, sr, subtype="PCM_16")
        return {"sample_rate": sr, "backend": "soundfile/librosa"}
    except Exception:
        # Last resort: try ffmpeg if available
        cmd = [
            _resolve_ffmpeg_executable(),
            "-y",
            "-i",
            src_path,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            dst_wav_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-4000:]
            hint = ""
            if "not recognized" in stderr_tail.lower() or "no such file" in stderr_tail.lower():
                hint = "\nHint: install ffmpeg and ensure it's in PATH, or set FFMPEG_PATH env var."
            raise RuntimeError(f"Failed to convert audio using ffmpeg. ffmpeg error: {stderr_tail}{hint}")
        return {"sample_rate": 16000, "backend": "ffmpeg"}


def _validate_antifake_installation() -> Tuple[bool, str]:
    base = AppConfig.ANTIFAKE_DIR
    if not os.path.isdir(base):
        return False, f"AntiFake directory not found: {base}"

    run_py = AppConfig.ANTIFAKE_RUN_PY
    if not os.path.isfile(run_py):
        return False, f"AntiFake run.py not found: {run_py}"

    # Required external weights (per AntiFake README)
    missing = [k for k, p in AppConfig.ANTIFAKE_REQUIRED_FILES.items() if not os.path.exists(p)]
    if missing:
        missing_paths = {k: AppConfig.ANTIFAKE_REQUIRED_FILES.get(k) for k in missing}
        return (
            False,
            "AntiFake required weight files are missing: "
            + ", ".join(missing)
            + ".\nExpected paths:\n"
            + "\n".join([f"- {k}: {v}" for k, v in missing_paths.items()])
            + "\n\nPlease download the supplementary bundle and place the files as described in thirdparty/AntiFake/README.md",
        )

    return True, "ok"


def _patch_antifake_run_py(tmp_dir: str) -> str:
    """Create a patched copy of AntiFake run.py without interactive pygame target selection.

    Original AntiFake requires human scoring to select target speaker. For automation we:
      - select a random target speaker from speakers_database (or "auto" = max embedding diff would require heavy code).
      - disable pygame usage.

    We keep the rest of the pipeline intact.
    """

    src = Path(AppConfig.ANTIFAKE_RUN_PY)
    text = src.read_text(encoding="utf-8", errors="ignore")

    # Force device to CPU if configured (Windows servers often have no CUDA).
    if AppConfig.ANTIFAKE_FORCE_CPU:
        # Keep this patch minimal to avoid corrupting upstream code.
        text = text.replace('DEVICE = "cuda"', 'DEVICE = "cpu"')
        text = text.replace("DEVICE = 'cuda'", "DEVICE = 'cpu'")
        text = text.replace(
            "spectrogram = torchaudio.transforms.Spectrogram().cuda()",
            "spectrogram = torchaudio.transforms.Spectrogram().to(DEVICE)",
        )

    # Make target selection non-interactive:
    # Remove the pygame/input loop and inject deterministic scores.
    #
    # We patch using code anchors (less brittle than comments) and preserve indentation.
    start_anchor = "pygame.mixer.init()"
    end_anchor = "# Compute source and target embedding differences"
    if start_anchor not in text or end_anchor not in text:
        raise RuntimeError(
            "AntiFake run.py patch failed: expected anchors were not found. "
            "The thirdparty/AntiFake/run.py file likely changed; update the adapter patch logic."
        )

    start_idx = text.index(start_anchor)
    end_idx = text.index(end_anchor)
    if end_idx <= start_idx:
        raise RuntimeError("AntiFake run.py patch failed: invalid anchor order")

    # Determine indentation level at the start_anchor line.
    line_start = text.rfind("\n", 0, start_idx) + 1
    line_end = text.find("\n", start_idx)
    if line_end == -1:
        line_end = len(text)
    anchor_line = text[line_start:line_end]
    indent = anchor_line[: len(anchor_line) - len(anchor_line.lstrip(" \t"))]

    replacement = (
        f"{indent}# User listens to source and targets, assign score to each\n"
        f"{indent}# (patched out in deepfake_detector integration: non-interactive selection)\n"
        f"{indent}user_scores = [3 for _ in target_speakers_selected]\n\n"
    )

    # Replace only the interactive block lines, preserving the indentation of the remaining file.
    # IMPORTANT: do NOT re-indent the tail; that was the cause of IndentationError on Windows.
    lines = text.splitlines(keepends=True)

    # Find line indices that contain the anchors.
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

    # The indentation of the interactive block is the indentation of the start anchor line.
    indent = lines[start_line_idx][: len(lines[start_line_idx]) - len(lines[start_line_idx].lstrip(" \t"))]
    replacement_lines = [
        f"{indent}# User listens to source and targets, assign score to each\n",
        f"{indent}# (patched out in deepfake_detector integration: non-interactive selection)\n",
        f"{indent}user_scores = [3 for _ in target_speakers_selected]\n",
        "\n",
    ]

    # Keep the end_anchor line itself (it likely starts the next stage).
    patched_lines = lines[:start_line_idx] + replacement_lines + lines[end_line_idx:]
    text = "".join(patched_lines)

    # Also reduce random targets to speed up.
    text = text.replace("NUM_RANDOM_TARGET_SPEAKER = 24", f"NUM_RANDOM_TARGET_SPEAKER = {int(AppConfig.ANTIFAKE_RANDOM_TARGETS)}")

    out_path = Path(tmp_dir) / "run_patched.py"
    out_path.write_text(text, encoding="utf-8")

    # Validate that we didn't corrupt the script (fail fast with a readable error).
    try:
        compile(text, str(out_path), "exec")
    except SyntaxError as e:
        # Provide a small context window around the failing line.
        lines = text.splitlines()
        ln = max(1, int(getattr(e, "lineno", 1)))
        lo = max(1, ln - 5)
        hi = min(len(lines), ln + 5)
        snippet = "\n".join([f"{i:04d}: {lines[i-1]}" for i in range(lo, hi + 1)])
        raise RuntimeError(
            "AntiFake run.py patch produced invalid Python. "
            f"SyntaxError: {e.msg} at line {ln}.\n\nContext:\n{snippet}"
        )

    return str(out_path)


def protect_audio_with_antifake(input_path: str, output_wav_path: str) -> Dict[str, Any]:
    """Protect audio using AntiFake. Output is a wav file."""

    ok, msg = _validate_antifake_installation()
    if not ok:
        raise RuntimeError(msg)

    # IMPORTANT: do NOT place AntiFake temporary work dir under project `temp/`.
    # When running the API with --reload / watchfiles, changes in temp trigger
    # a server reload loop (because we generate run_patched.py there).
    # Use system temp instead.
    work_dir = tempfile.mkdtemp(prefix="antifake_")
    try:
        src_wav = os.path.join(work_dir, "input.wav")
        conv_meta = _convert_to_antifake_input_wav(input_path, src_wav)

        duration = _get_duration_seconds(src_wav)
        if duration > AppConfig.ANTIFAKE_MAX_SECONDS:
            raise ValueError(
                f"Audio is too long for protection ({duration:.1f}s). Max is {AppConfig.ANTIFAKE_MAX_SECONDS:.0f}s"
            )

        patched_run = _patch_antifake_run_py(work_dir)

        # Run AntiFake as a subprocess to avoid polluting sys.path and global torch state.
        cmd = [sys.executable, patched_run, src_wav, output_wav_path]
        proc = subprocess.run(
            cmd,
            cwd=AppConfig.ANTIFAKE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "AntiFake failed. "
                f"Work dir: {work_dir}\n"
                f"Patched script: {patched_run}\n"
                f"stdout: {proc.stdout[-2000:]}\n"
                f"stderr: {proc.stderr[-2000:]}"
            )

        if not os.path.exists(output_wav_path):
            raise RuntimeError("AntiFake completed but output file was not created")

        return {
            "converted": conv_meta,
            "duration": float(duration),
            "stdout_tail": proc.stdout[-1000:],
        }
    finally:
        # Keep work_dir on failure to aid debugging. Clean it only on success.
        if os.path.exists(output_wav_path):
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass














