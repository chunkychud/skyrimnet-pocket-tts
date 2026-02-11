#!/usr/bin/env python3
"""
SkyrimNet Simplified FastAPI TTS Service
Simplified FastAPI service modeling APIs from xtts_api_server but using methodology from skyrimnet-xtts.py
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from pocket_tts import TTSModel

# Local imports - Handle both direct execution and module execution
try:
    from .shared_config import SUPPORTED_LANGUAGE_CODES, validate_language
    from .shared_args import parse_api_args
    from .shared_app_utils import setup_application_logging, initialize_application_environment
    from .shared_app_cleanup import output_cleanup_worker, cleanup_output_directory
except ImportError:
    from shared_config import SUPPORTED_LANGUAGE_CODES, validate_language
    from shared_args import parse_api_args
    from shared_app_utils import setup_application_logging, initialize_application_environment
    from shared_app_cleanup import output_cleanup_worker, cleanup_output_directory


# =============================================================================
# CONFIG / PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
SPEAKER_DIRECTORY = ROOT_DIR / "speakers"
OUTPUT_DIRECTORY = ROOT_DIR / "output"
WEIGHTS_DIRECTORY = ROOT_DIR / "weights"

SPEAKER_DIRECTORY.mkdir(parents=True, exist_ok=True)
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

SILENCE_AUDIO_PATH = SPEAKER_DIRECTORY / "silence_100ms.wav"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        return default


# For testing you can set:
#   set OUTPUT_CLEANUP_MAX_AGE_SECONDS=30
#   set OUTPUT_CLEANUP_INTERVAL_SECONDS=10
OUTPUT_CLEANUP_MAX_AGE_SECONDS = _env_int("OUTPUT_CLEANUP_MAX_AGE_SECONDS", 10 * 60)   # default 10 min
OUTPUT_CLEANUP_INTERVAL_SECONDS = _env_int("OUTPUT_CLEANUP_INTERVAL_SECONDS", 5 * 60)  # default 5 min
OUTPUT_CLEANUP_GLOB = os.environ.get("OUTPUT_CLEANUP_GLOB", "*").strip() or "*"

# Global model/state
CURRENT_MODEL: Optional[TTSModel] = None
IGNORE_PING: Optional[bool | str] = None
CACHED_TEMP_DIR: Optional[Path] = None

SPEAKERS_AVAILABLE: dict[str, Path] = {}

# Parse CLI args
args = parse_api_args("SkyrimNet Simplified TTS API")

# Logging init guard
_LOGGING_INITIALIZED = False


def initialize_api_logging() -> None:
    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_application_logging()
        _LOGGING_INITIALIZED = True


if __name__ == "__main__":
    initialize_api_logging()


# =============================================================================
# MODELS
# =============================================================================

class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: Optional[str] = None
    language: Optional[str] = "en"
    accent: Optional[str] = None
    save_path: Optional[str] = None
    # (kept for compatibility; currently not used by pocket_tts)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    speed: Optional[float] = None
    repetition_penalty: Optional[float] = None
    override: Optional[bool] = False


# =============================================================================
# HELPERS
# =============================================================================

def get_cached_temp_dir() -> Path:
    global CACHED_TEMP_DIR
    if CACHED_TEMP_DIR is None or not CACHED_TEMP_DIR.exists():
        CACHED_TEMP_DIR = Path(tempfile.mkdtemp(prefix="skyrimnet_tts_"))
        logger.info("Using cached temp directory: {}", str(CACHED_TEMP_DIR))
    return CACHED_TEMP_DIR


def update_available_speakers() -> None:
    global SPEAKERS_AVAILABLE
    SPEAKERS_AVAILABLE = {}
    for p in SPEAKER_DIRECTORY.glob("**/*.wav"):
        if p.is_file():
            SPEAKERS_AVAILABLE[p.stem] = p
    logger.info("Speaker currently available: {}", list(SPEAKERS_AVAILABLE.keys()))


_SPEAKER_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def _sanitize_speaker_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="speaker_name is required")
    if not _SPEAKER_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="speaker_name must be 1-64 chars of letters/numbers/_/- only",
        )
    return name


def _ensure_silence_wav(sample_rate: int) -> None:
    if SILENCE_AUDIO_PATH.exists():
        return
    SILENCE_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples = int(sample_rate * 0.1)  # 100ms
    silence = np.zeros(samples, dtype=np.float32)
    scipy.io.wavfile.write(str(SILENCE_AUDIO_PATH), sample_rate, silence)
    logger.info("Created silence WAV: {}", str(SILENCE_AUDIO_PATH))


def _detect_p_core_logical_cpus_windows() -> list[int]:
    """
    Returns logical CPU indices that appear to be P-cores by picking CPUs with the highest
    EfficiencyClass from Windows CPU Set info. If detection fails, returns [].
    """
    if os.name != "nt":
        return []

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    GetSystemCpuSetInformation = kernel32.GetSystemCpuSetInformation
    GetSystemCpuSetInformation.argtypes = [
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG),
        wintypes.HANDLE,
        wintypes.ULONG,
    ]
    GetSystemCpuSetInformation.restype = wintypes.BOOL

    class _CPUSET_INFO_HEADER(ctypes.Structure):
        _fields_ = [("Size", wintypes.ULONG), ("Type", wintypes.ULONG)]

    class _CPUSET_INFO_CPUSET(ctypes.Structure):
        _fields_ = [
            ("Id", wintypes.ULONG),
            ("Group", wintypes.USHORT),
            ("LogicalProcessorIndex", wintypes.BYTE),
            ("CoreIndex", wintypes.BYTE),
            ("LastLevelCacheIndex", wintypes.BYTE),
            ("NumaNodeIndex", wintypes.BYTE),
            ("EfficiencyClass", wintypes.BYTE),
            ("AllFlags", wintypes.BYTE),
            ("Reserved", wintypes.USHORT),
            ("AllocationTag", wintypes.ULONG),
        ]

    needed = wintypes.ULONG(0)
    ok = GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), None, 0)
    if ok or ctypes.get_last_error() != 122 or needed.value == 0:
        return []

    buf = (ctypes.c_ubyte * needed.value)()
    ok = GetSystemCpuSetInformation(buf, needed.value, ctypes.byref(needed), None, 0)
    if not ok:
        return []

    base = ctypes.addressof(buf)
    offset = 0
    header_size = ctypes.sizeof(_CPUSET_INFO_HEADER)

    cpu_sets: list[dict] = []
    while offset < needed.value:
        hdr = _CPUSET_INFO_HEADER.from_address(base + offset)
        if hdr.Size == 0:
            break

        if hdr.Type == 0 and hdr.Size >= header_size + ctypes.sizeof(_CPUSET_INFO_CPUSET):
            cs = _CPUSET_INFO_CPUSET.from_address(base + offset + header_size)
            cpu_sets.append({"lpi": int(cs.LogicalProcessorIndex), "eff": int(cs.EfficiencyClass)})

        offset += hdr.Size

    if not cpu_sets:
        return []

    max_eff = max(d["eff"] for d in cpu_sets)
    pcores = sorted({d["lpi"] for d in cpu_sets if d["eff"] == max_eff})
    return pcores


def pin_process_to_pcores() -> None:
    """
    Optional: pin the *process* to detected P-cores on Windows hybrid CPUs.
    Safe no-op on non-Windows.
    """
    if os.name != "nt":
        return

    try:
        pcores = _detect_p_core_logical_cpus_windows()
        if not pcores:
            logger.info("P-core detection returned empty; skipping pinning")
            return

        try:
            import psutil  # optional dependency in your project
        except Exception:
            logger.info("psutil not available; skipping pinning")
            return

        proc = psutil.Process()
        proc.cpu_affinity(pcores)
        logger.info("Pinned process to detected P-cores: {}", pcores)
    except Exception as e:
        logger.warning("Failed to pin process to P-cores: {}", e)


def _safe_output_filename(speaker_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_speaker = speaker_name.replace("\\", "_").replace("/", "_")
    return f"{ts}_{safe_speaker}.wav"


# =============================================================================
# FASTAPI APP + LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global CURRENT_MODEL

    logger.info("Lifespan STARTUP enter")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SPEAKER_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Optional: pin to P-cores (Windows only)
    pin_process_to_pcores()

    # Refresh speakers
    update_available_speakers()

    # Load model (off the event loop)
    try:
        t0 = time.perf_counter()
        CURRENT_MODEL = await asyncio.to_thread(TTSModel.load_model)
        logger.info("Model loaded successfully in {:.2f}s", time.perf_counter() - t0)
        _ensure_silence_wav(CURRENT_MODEL.sample_rate)
    except Exception as e:
        logger.error("Failed to load model: {}", e)
        raise

    # Start cleanup worker
    stop_event = asyncio.Event()
    cleanup_task = asyncio.create_task(
        output_cleanup_worker(
            stop_event=stop_event,
            output_dir=OUTPUT_DIRECTORY,
            max_age_seconds=OUTPUT_CLEANUP_MAX_AGE_SECONDS,
            interval_seconds=OUTPUT_CLEANUP_INTERVAL_SECONDS,
            glob_pattern=OUTPUT_CLEANUP_GLOB,
        )
    )

    try:
        yield
    finally:
        logger.info("Lifespan SHUTDOWN enter")

        stop_event.set()
        try:
            await asyncio.wait_for(cleanup_task, timeout=5)
        except asyncio.TimeoutError:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            logger.warning("Cleanup worker ended with error: {}", e)

        # Final cleanup pass (always log)
        deleted = cleanup_output_directory(
            output_dir=OUTPUT_DIRECTORY,
            max_age_seconds=OUTPUT_CLEANUP_MAX_AGE_SECONDS,
            glob_pattern=OUTPUT_CLEANUP_GLOB,
        )
        logger.info("Shutdown cleanup pass deleted {} file(s).", deleted)

        logger.info("Lifespan SHUTDOWN exit")


app = FastAPI(
    title="SkyrimNet TTS API",
    description="Simplified TTS API service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed = time.perf_counter() - start
        logger.info("{} {} -> {:.4f}s", request.method, request.url.path, elapsed)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.post("/tts_to_audio/")
async def tts_to_audio(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """
    Generate TTS audio from text with specified speaker voice.
    """
    global IGNORE_PING

    if CURRENT_MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Ping shortcut
    if text == "ping":
        if IGNORE_PING is None:
            IGNORE_PING = "pending"
        else:
            logger.info("Ping request received, sending silence audio.")
            return FileResponse(path=SILENCE_AUDIO_PATH, filename=SILENCE_AUDIO_PATH.name, media_type="audio/wav")

    # Validate language (even if pocket_tts ignores it today, keep consistent API behavior)
    try:
        _ = validate_language(request.language or "en")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    speaker_wav = (request.speaker_wav or "malecommoner").strip()

    # Ensure speaker exists; refresh cache if needed
    if speaker_wav not in SPEAKERS_AVAILABLE:
        update_available_speakers()
    if speaker_wav not in SPEAKERS_AVAILABLE:
        raise HTTPException(status_code=400, detail=f"Unknown speaker_wav: {speaker_wav}")

    start_time = time.perf_counter()
    logger.info(
        "POST /tts_to_audio text_len={} speaker_wav='{}' language='{}'",
        len(text),
        speaker_wav,
        request.language or "en",
    )

    # Generate audio (CPU-heavy) off the event loop
    voice_state_start_time = time.perf_counter()
    voice_state = CURRENT_MODEL.get_state_for_audio_prompt(SPEAKERS_AVAILABLE[speaker_wav])
    voice_state_gen_time = time.perf_counter() - voice_state_start_time
    logger.info("Generated voice state in {:.3f}s", voice_state_gen_time)

    audio = await asyncio.to_thread(CURRENT_MODEL.generate_audio, voice_state, text)

    gen_time = time.perf_counter() - start_time
    dur_s = float(audio.shape[-1]) / float(CURRENT_MODEL.sample_rate)
    logger.info("Generated {:.3f}s audio in {:.3f}s", dur_s, gen_time)

    # Write output (also off event loop)
    wav_name = _safe_output_filename(speaker_wav)
    wav_path = OUTPUT_DIRECTORY / wav_name
    await asyncio.to_thread(scipy.io.wavfile.write, str(wav_path), CURRENT_MODEL.sample_rate, audio.numpy())

    # Handle "ignore ping" flow safely
    if IGNORE_PING == "pending":
        IGNORE_PING = True
        wav_path.unlink(missing_ok=True)
        wav_path = SILENCE_AUDIO_PATH
        wav_name = wav_path.name

    return FileResponse(path=wav_path, filename=wav_name, media_type="audio/wav")

@app.post("/create_and_store_latents")
async def create_and_store_latents(
    speaker_name: str = Form(...),
    language: str = Form("en"),
    wav_file: UploadFile = File(...),
):
    """
    Store an uploaded WAV as a speaker prompt (speaker_name.wav).
    (This endpoint returns the expected JSON shape but does not compute latents.)
    """
    if CURRENT_MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    speaker_name = _sanitize_speaker_name(speaker_name)

    try:
        _ = validate_language(language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not (wav_file.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    dest = SPEAKER_DIRECTORY / f"{speaker_name}.wav"

    if speaker_name in SPEAKERS_AVAILABLE and dest.exists():
        logger.info("Speaker '{}' already exists; not overwriting.", speaker_name)
    else:
        # Stream copy to disk to avoid holding full audio in memory
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as wav with open:
            shutil.copyfileobj(wav_file.file, wav)
            voice_state = CURRENT_MODEL.get_state_for_audio_prompt(wav_file.file)

        update_available_speakers()
        logger.info("Stored speaker wav: {}", str(dest))

    return {
        "message": f"Successfully stored '{speaker_name}' in language '{language}'",
        "speaker_name": speaker_name,
        "language": language,
        "latent_shapes": {
            "gpt_cond_latent": [],
            "speaker_embedding": [],
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": CURRENT_MODEL is not None,
        "supported_languages": SUPPORTED_LANGUAGE_CODES,
        "cleanup": {
            "max_age_seconds": OUTPUT_CLEANUP_MAX_AGE_SECONDS,
            "interval_seconds": OUTPUT_CLEANUP_INTERVAL_SECONDS,
            "glob": OUTPUT_CLEANUP_GLOB,
        },
    }


def setup_catch_all_route():
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def catch_undefined_endpoints(request: Request, path: str):
        logger.warning("UNDEFINED ENDPOINT: {} /{}", request.method, path)
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Endpoint not found: {request.method} /{path}",
                "available_endpoints": [
                    "POST /tts_to_audio",
                    "POST /create_and_store_latents",
                    "GET /health",
                    "GET /docs",
                    "GET /redoc",
                ],
            },
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    initialize_application_environment("SkyrimNet TTS API")
    setup_catch_all_route()

    logger.info("Starting server on {}:{}", args.server, args.port)
    uvicorn.run(
        app,
        host=args.server,
        port=args.port,
        log_level="info",
        access_log=False,
        log_config=None,
        lifespan="on",
    )
