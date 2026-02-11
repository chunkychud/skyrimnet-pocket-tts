#!/usr/bin/env python3
"""
SkyrimNet Simplified FastAPI TTS Service
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np
import scipy.io.wavfile
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

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
LATENTS_DIRECTORY = ROOT_DIR / "latents"
OUTPUT_DIRECTORY = ROOT_DIR / "output"
WEIGHTS_DIRECTORY = ROOT_DIR / "weights"

SPEAKER_DIRECTORY.mkdir(parents=True, exist_ok=True)
LATENTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
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


# Cleanup tuning (seconds)
OUTPUT_CLEANUP_MAX_AGE_SECONDS = _env_int("OUTPUT_CLEANUP_MAX_AGE_SECONDS", 10 * 60)   # default 10 min
OUTPUT_CLEANUP_INTERVAL_SECONDS = _env_int("OUTPUT_CLEANUP_INTERVAL_SECONDS", 5 * 60)  # default 5 min
OUTPUT_CLEANUP_GLOB = os.environ.get("OUTPUT_CLEANUP_GLOB", "*").strip() or "*"

# Global model/state
CURRENT_MODEL: Optional[TTSModel] = None
IGNORE_PING: Optional[bool | str] = None

# Speaker index:
#   speaker_name -> latents/{speaker_name}.json
SPEAKERS_AVAILABLE: dict[str, Path] = {}

# In-memory voice_state cache:
#   speaker_name -> voice_state dict
_VOICE_STATE_CACHE: dict[str, dict[str, Any]] = {}
_VOICE_CACHE_LOCK = Lock()

# Parse CLI args
args = parse_api_args("SkyrimNet Simplified TTS API")

_LOGGING_INITIALIZED = False


def initialize_api_logging() -> None:
    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_application_logging()
        _LOGGING_INITIALIZED = True


if __name__ == "__main__":
    initialize_api_logging()


# =============================================================================
# REQUEST MODELS
# =============================================================================

class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: Optional[str] = None
    language: Optional[str] = "en"
    accent: Optional[str] = None
    save_path: Optional[str] = None
    # kept for compatibility; not used by pocket_tts currently
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    speed: Optional[float] = None
    repetition_penalty: Optional[float] = None
    override: Optional[bool] = False


# =============================================================================
# SPEAKER / LATENT UTILITIES
# =============================================================================

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


def update_available_speakers() -> None:
    """
    Speakers are defined by the presence of a latent json in latents/{speaker}.json.
    """
    global SPEAKERS_AVAILABLE
    SPEAKERS_AVAILABLE = {}

    for p in LATENTS_DIRECTORY.glob("**/*.json"):
        if p.is_file():
            SPEAKERS_AVAILABLE[p.stem] = p

    logger.info("Speakers (latents) currently available: {}", list(SPEAKERS_AVAILABLE.keys()))


def _safe_output_filename(speaker_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_speaker = speaker_name.replace("\\", "_").replace("/", "_")
    return f"{ts}_{safe_speaker}.wav"


def _ensure_silence_wav(sample_rate: int) -> None:
    if SILENCE_AUDIO_PATH.exists():
        return
    SILENCE_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples = int(sample_rate * 0.1)  # 100ms
    silence = np.zeros(samples, dtype=np.float32)
    scipy.io.wavfile.write(str(SILENCE_AUDIO_PATH), sample_rate, silence)
    logger.info("Created silence WAV: {}", str(SILENCE_AUDIO_PATH))


# =============================================================================
# JSON (DE)SERIALIZATION FOR voice_state
# =============================================================================

def _json_default(obj: Any) -> Any:
    """
    Make voice_state JSON-serializable.

    Handles:
    - Path -> str
    - numpy arrays -> typed payload
    - torch tensors -> typed payload (cpu list)
    - bytes -> hex string payload
    """
    if isinstance(obj, Path):
        return {"__type__": "path", "value": str(obj)}

    if isinstance(obj, (np.ndarray,)):
        return {
            "__type__": "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tolist(),
        }

    if torch is not None and isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        return {
            "__type__": "tensor",
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "data": t.tolist(),
        }

    if isinstance(obj, (bytes, bytearray)):
        return {"__type__": "bytes_hex", "value": bytes(obj).hex()}

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_object_hook(d: dict[str, Any]) -> Any:
    """
    Reverse _json_default payloads back into Python objects.
    """
    t = d.get("__type__")
    if not t:
        return d

    if t == "path":
        return Path(d["value"])

    if t == "ndarray":
        arr = np.array(d["data"], dtype=np.dtype(d["dtype"]))
        shape = tuple(d.get("shape", []))
        if shape and arr.shape != shape:
            arr = arr.reshape(shape)
        return arr

    if t == "tensor":
        if torch is None:
            # If torch isn't available, keep as ndarray so we don't crash on load.
            arr = np.array(d["data"])
            shape = tuple(d.get("shape", []))
            if shape and arr.shape != shape:
                arr = arr.reshape(shape)
            return arr
        dtype_str = d.get("dtype", "float32")
        dtype = getattr(torch, dtype_str, torch.float32)
        tns = torch.tensor(d["data"], dtype=dtype)
        shape = d.get("shape")
        if shape:
            tns = tns.reshape(shape)
        return tns

    if t == "bytes_hex":
        return bytes.fromhex(d["value"])

    return d


def _write_voice_state_json(path: Path, voice_state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(voice_state, f, ensure_ascii=False, indent=2, default=_json_default)


def _read_voice_state_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f, object_hook=_json_object_hook)
    if not isinstance(data, dict):
        raise ValueError(f"Latent file did not contain a dict: {path}")
    return data


def get_cached_voice_state(speaker_name: str) -> dict[str, Any]:
    """
    Load voice_state from in-memory cache; fall back to latents/{speaker}.json.
    """
    with _VOICE_CACHE_LOCK:
        cached = _VOICE_STATE_CACHE.get(speaker_name)
        if cached is not None:
            return cached

    # Not cached; load from disk
    latent_path = SPEAKERS_AVAILABLE.get(speaker_name)
    if latent_path is None:
        raise KeyError(speaker_name)

    voice_state = _read_voice_state_json(latent_path)

    with _VOICE_CACHE_LOCK:
        _VOICE_STATE_CACHE[speaker_name] = voice_state

    return voice_state


def cache_voice_state(speaker_name: str, voice_state: dict[str, Any]) -> None:
    with _VOICE_CACHE_LOCK:
        _VOICE_STATE_CACHE[speaker_name] = voice_state


async def _build_and_store_voice_state_from_wav(speaker_name: str, wav_path: Path) -> dict[str, Any]:
    """
    Build voice_state dict from a WAV prompt, store to latents/{speaker}.json, update indexes + cache.
    """
    if CURRENT_MODEL is None:
        raise RuntimeError("Model not loaded")

    latent_path = LATENTS_DIRECTORY / f"{speaker_name}.json"

    t0 = time.perf_counter()
    voice_state = await asyncio.to_thread(CURRENT_MODEL.get_state_for_audio_prompt, wav_path)
    logger.info("Built voice_state for '{}' in {:.3f}s", speaker_name, time.perf_counter() - t0)

    await asyncio.to_thread(_write_voice_state_json, latent_path, voice_state)

    # Update speaker availability and cache
    SPEAKERS_AVAILABLE[speaker_name] = latent_path
    cache_voice_state(speaker_name, voice_state)

    return voice_state


# =============================================================================
# WINDOWS P-CORE PINNING (optional)
# =============================================================================

def _detect_p_core_logical_cpus_windows() -> list[int]:
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

    cpu_sets: list[dict[str, int]] = []
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
    return sorted({d["lpi"] for d in cpu_sets if d["eff"] == max_eff})


def pin_process_to_pcores() -> None:
    if os.name != "nt":
        return
    try:
        pcores = _detect_p_core_logical_cpus_windows()
        if not pcores:
            logger.info("P-core detection returned empty; skipping pinning")
            return

        try:
            import psutil  # optional dependency
        except Exception:
            logger.info("psutil not available; skipping pinning")
            return

        proc = psutil.Process()
        proc.cpu_affinity(pcores)
        logger.info("Pinned process to detected P-cores: {}", pcores)
    except Exception as e:
        logger.warning("Failed to pin process to P-cores: {}", e)


# =============================================================================
# FASTAPI APP + LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global CURRENT_MODEL

    logger.info("Lifespan STARTUP enter")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SPEAKER_DIRECTORY.mkdir(parents=True, exist_ok=True)
    LATENTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    pin_process_to_pcores()

    # Index existing speakers (latents)
    update_available_speakers()

    # Load model off event loop
    t0 = time.perf_counter()
    CURRENT_MODEL = await asyncio.to_thread(TTSModel.load_model)
    logger.info("Model loaded successfully in {:.2f}s", time.perf_counter() - t0)
    _ensure_silence_wav(CURRENT_MODEL.sample_rate)

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

        deleted = cleanup_output_directory(
            output_dir=OUTPUT_DIRECTORY,
            max_age_seconds=0,
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

@app.post("/create_and_store_latents")
async def create_and_store_latents(
    speaker_name: str = Form(...),
    language: str = Form("en"),
    wav_file: UploadFile = File(...),
):
    """
    Store an uploaded WAV as speakers/{speaker_name}.wav and store voice_state as latents/{speaker_name}.json.
    The stored voice_state is later used directly in /tts_to_audio for faster generation.
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

    wav_path = SPEAKER_DIRECTORY / f"{speaker_name}.wav"
    latent_path = LATENTS_DIRECTORY / f"{speaker_name}.json"

    # Write wav to disk (stream copy)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wav_path.open("wb") as f:
        shutil.copyfileobj(wav_file.file, f)

    # Build + store voice_state
    voice_state = await _build_and_store_voice_state_from_wav(speaker_name, wav_path)

    # Refresh index (in case other changes happened)
    update_available_speakers()

    return {
        "message": f"Stored speaker '{speaker_name}' and latents for language '{language}'",
        "speaker_name": speaker_name,
        "language": language,
        "speaker_wav_path": str(wav_path),
        "latent_path": str(latent_path),
        "voice_state_keys": list(voice_state.keys()),
    }


@app.post("/tts_to_audio/")
async def tts_to_audio(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """
    Generate TTS audio from text with specified speaker voice.

    Uses cached voice_state loaded from latents/{speaker}.json for faster generation.
    If latent file is missing but speakers/{speaker}.wav exists, it will build and persist the latent automatically.
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

    # Validate language
    try:
        _ = validate_language(request.language or "en")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    speaker = (request.speaker_wav or "malecommoner").strip()
    if not speaker:
        speaker = "malecommoner"

    start_time = time.perf_counter()
    logger.info("POST /tts_to_audio text_len={} speaker='{}' language='{}'", len(text), speaker, request.language or "en")

    # Ensure speaker exists in index; refresh if not
    if speaker not in SPEAKERS_AVAILABLE:
        update_available_speakers()

    voice_state: dict[str, Any]

    # If still not present, attempt to auto-build from speakers/{speaker}.wav if it exists
    if speaker not in SPEAKERS_AVAILABLE:
        wav_path = SPEAKER_DIRECTORY / f"{speaker}.wav"
        if wav_path.exists():
            logger.info("Latent missing for '{}', auto-building from {}", speaker, str(wav_path))
            voice_state = await _build_and_store_voice_state_from_wav(speaker, wav_path)
            update_available_speakers()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown speaker: {speaker}")
    else:
        # Load from cache / disk
        try:
            voice_state = get_cached_voice_state(speaker)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load voice_state for '{speaker}': {e}")

    # Generate audio (CPU-heavy) off the event loop
    t_gen0 = time.perf_counter()
    audio = await asyncio.to_thread(CURRENT_MODEL.generate_audio, voice_state, text)
    gen_elapsed = time.perf_counter() - t_gen0

    dur_s = float(audio.shape[-1]) / float(CURRENT_MODEL.sample_rate)
    logger.info("Generated {:.3f}s audio in {:.3f}s (rtf={:.2f}x)", dur_s, gen_elapsed, (dur_s / gen_elapsed) if gen_elapsed > 0 else 0.0)

    # Write output off event loop
    wav_name = _safe_output_filename(speaker)
    wav_path = OUTPUT_DIRECTORY / wav_name
    t_write0 = time.perf_counter()
    await asyncio.to_thread(scipy.io.wavfile.write, str(wav_path), CURRENT_MODEL.sample_rate, audio.numpy())
    write_elapsed = time.perf_counter() - t_write0

    total_elapsed = time.perf_counter() - start_time
    logger.info("Wrote {} in {:.3f}s | total {:.3f}s", str(wav_path), write_elapsed, total_elapsed)

    # Handle "ignore ping" flow
    if IGNORE_PING == "pending":
        IGNORE_PING = True
        wav_path.unlink(missing_ok=True)
        wav_path = SILENCE_AUDIO_PATH
        wav_name = wav_path.name

    return FileResponse(path=wav_path, filename=wav_name, media_type="audio/wav")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": CURRENT_MODEL is not None,
        "supported_languages": SUPPORTED_LANGUAGE_CODES,
        "speakers_available": sorted(SPEAKERS_AVAILABLE.keys()),
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
