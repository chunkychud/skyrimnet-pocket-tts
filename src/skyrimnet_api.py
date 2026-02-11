
#!/usr/bin/env python3
"""
SkyrimNet Simplified FastAPI TTS Service
Simplified FastAPI service modeling APIs from xtts_api_server but using methodology from skyrimnet-xtts.py
"""

# Standard library imports
from datetime import datetime
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
import asyncio

# Third-party imports
import scipy
import uvicorn
import time
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

# TTS Model
from pocket_tts import TTSModel

# Windows specific optimizations
import ctypes
from ctypes import wintypes
import psutil

# Local imports - Handle both direct execution and module execution
try:
    # Try relative imports first (for module execution: python -m skyrimnet-xtts)
    from .shared_config import SUPPORTED_LANGUAGE_CODES, validate_language, get_tts_params
    from .shared_args import parse_api_args
    from .shared_app_utils import setup_application_logging, initialize_application_environment
    from .shared_app_cleanup import output_cleanup_worker, cleanup_output_directory
except ImportError:
    # Fall back to absolute imports (for direct execution: python skyrimnet_api.py)
    from shared_config import SUPPORTED_LANGUAGE_CODES, validate_language, get_tts_params
    from shared_args import parse_api_args
    from shared_app_utils import setup_application_logging, initialize_application_environment
    from shared_app_cleanup import output_cleanup_worker, cleanup_output_directory
# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# Global Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
SPEAKER_DIRECTORY = ROOT_DIR / "speakers"
OUTPUT_DIRECTORY = ROOT_DIR / "output"
WEIGHTS_DIRECTORY = ROOT_DIR / "weights"

# Ensure the paths exist
SPEAKER_DIRECTORY.mkdir(parents=True, exist_ok=True)
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Global model state
CURRENT_MODEL = None
IGNORE_PING = None
CACHED_TEMP_DIR = None
AVAILABLE_SPEAKERS = {}
SILENCE_AUDIO_PATH =  SPEAKER_DIRECTORY / "silence_100ms.wav"
OUTPUT_CLEANUP_MAX_AGE_MINUTES = 5
OUTPUT_CLEANUP_INTERVAL_SECONDS = 5 * 60  # To check the output dir every 5 minutes for cleanup

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

args = parse_api_args("SkyrimNet Simplified TTS API")

# =============================================================================
# LOGGING SETUP
# =============================================================================

# Global flag to track if logging has been initialized
_LOGGING_INITIALIZED = False

def initialize_api_logging():
    """Initialize logging for the API module"""
    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        # Setup standardized logging (only when not already configured)
        setup_application_logging()
        _LOGGING_INITIALIZED = True

# Only setup logging when running as standalone script
if __name__ == "__main__":
    initialize_api_logging()

# =============================================================================
# PYDANTIC REQUEST/RESPONSE MODELS
# =============================================================================

class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: Optional[str] = None
    language: Optional[str] = "en"
    accent: Optional[str] = None
    save_path: Optional[str] = None
    # TTS inference parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    speed: Optional[float] = None
    repetition_penalty: Optional[float] = None
    # Override flag: if True, payload values override config file
    override: Optional[bool] = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cached_temp_dir():
    """Get or create the cached temporary directory"""
    global CACHED_TEMP_DIR
    
    if CACHED_TEMP_DIR is None:
        CACHED_TEMP_DIR = Path(tempfile.mkdtemp(prefix="skyrimnet_tts_"))
        logger.info(f"Created cached temp directory: {CACHED_TEMP_DIR}")
    elif not CACHED_TEMP_DIR.exists():
        # Recreate if it was somehow deleted
        CACHED_TEMP_DIR = Path(tempfile.mkdtemp(prefix="skyrimnet_tts_"))
        logger.info(f"Recreated cached temp directory: {CACHED_TEMP_DIR}")
    
    return CACHED_TEMP_DIR


def update_available_speakers():
    global SPEAKERS_AVAILABLE
    SPEAKERS_AVAILABLE = {}

    for p in SPEAKER_DIRECTORY.glob("**/*.wav"):  # recursive
        if p.is_file():
            speaker_name = p.name.replace(".wav", "") # Speaker name is the filename minus the extension
            SPEAKERS_AVAILABLE[speaker_name] = p
    
    logger.info(f"Speaker currently available: {[speaker_name for speaker_name in SPEAKERS_AVAILABLE.keys()]}")


def _detect_p_core_logical_cpus_windows() -> list[int]:
    """
    Returns logical CPU indices that appear to be P-cores by picking CPUs with the highest
    EfficiencyClass from Windows CPU Set info.

    If detection fails, returns [].
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
    # ERROR_INSUFFICIENT_BUFFER = 122 is expected on the sizing call
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

        # Type 0 is CpuSet entries on current Windows builds
        if hdr.Type == 0 and hdr.Size >= header_size + ctypes.sizeof(_CPUSET_INFO_CPUSET):
            cs = _CPUSET_INFO_CPUSET.from_address(base + offset + header_size)
            cpu_sets.append(
                {
                    "lpi": int(cs.LogicalProcessorIndex),
                    "eff": int(cs.EfficiencyClass),
                }
            )

        offset += hdr.Size

    if not cpu_sets:
        return []

    max_eff = max(x["eff"] for x in cpu_sets)
    pcores = sorted({x["lpi"] for x in cpu_sets if x["eff"] == max_eff})
    return pcores

def pin_process_to_p_cores(logger) -> None:
    """
    Pins the current process to P-cores (or best-guess) to avoid scheduling on E-cores.
    - Uses env var PCORE_CPUS="0,1,2,3" if provided (highest priority).
    - Else uses Windows CPU Set EfficiencyClass detection.
    """
    if os.name != "nt":
        return

    # Manual override is often the most reliable.
    override = os.environ.get("PCORE_CPUS", "").strip()
    if override:
        try:
            pcore_cpus = [int(x.strip()) for x in override.split(",") if x.strip()]
        except ValueError:
            pcore_cpus = []
        if pcore_cpus:
            psutil.Process(os.getpid()).cpu_affinity(pcore_cpus)
            logger.info(f"Pinned process to PCORE_CPUS override: {pcore_cpus}")
            return

    pcore_cpus = _detect_p_core_logical_cpus_windows()
    if pcore_cpus:
        psutil.Process(os.getpid()).cpu_affinity(pcore_cpus)
        logger.info(f"Pinned process to detected P-cores: {pcore_cpus}")
    else:
        logger.warning("Could not detect P-cores; leaving default CPU scheduling in place.")

# =============================================================================
# Cache management
# =============================================================================
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Cache manager")

    stop_event = asyncio.Event()
    task = asyncio.create_task(output_cleanup_worker(
        stop_event,
        OUTPUT_DIRECTORY, 
        OUTPUT_CLEANUP_MAX_AGE_MINUTES,
        OUTPUT_CLEANUP_INTERVAL_SECONDS
        ))
    try:
        yield
    finally:
        logger.info("Lifespan SHUTDOWN enter")
        stop_event.set()

        try:
            await asyncio.wait_for(task, timeout=5)
        except asyncio.TimeoutError:
            logger.warning("Cleanup worker didn't stop in time; cancelling")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        except Exception:
            logger.exception("Cleanup worker crashed during shutdown")

        deleted = cleanup_output_directory(OUTPUT_DIRECTORY, 0) # On shutdown the output folder should be cleaned
        logger.info(f"Shutdown cleanup pass deleted {deleted} file(s).")

        logger.info("Lifespan SHUTDOWN exit")

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(title="SkyrimNet TTS API", description="Simplified TTS API service", version="1.0.0", lifespan=lifespan)

# Request logging middleware (logs ALL requests, even undefined endpoints)
#@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log the incoming request
    logger.info(f"📥 INCOMING REQUEST: {request.method} {request.url}")
    logger.info(f"   Headers: {dict(request.headers)}")
    logger.info(f"   Client: {request.client.host if request.client else 'unknown'}")
    
    # Log query parameters if any
    if request.query_params:
        logger.info(f"   Query params: {dict(request.query_params)}")
    
    # Try to log request body for POST requests (be careful with large files)
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                # For JSON requests, we can log the body
                body = await request.body()
                if len(body) < 2000:  # Only log small bodies
                    logger.info(f"   Body: {body.decode('utf-8')}")
                else:
                    logger.info(f"   Body: <large body {len(body)} bytes>")
            elif "multipart/form-data" in content_type:
                logger.info(f"   Body: <multipart form data>")
            else:
                logger.info(f"   Body: <{content_type}>")
        except Exception as e:
            logger.warning(f"   Body: <failed to read body: {e}>")
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log the response
        logger.info(f"📤 RESPONSE: {response.status_code} for {request.method} {request.url.path}")
        logger.info(f"   Processing time: {process_time:.4f}s")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ REQUEST FAILED: {request.method} {request.url.path}")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Processing time: {process_time:.4f}s")
        raise

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API ENDPOINTS
# =============================================================================


##@app.post("/tts_to_audio")
@app.post("/tts_to_audio/")
async def tts_to_audio(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """
    Generate TTS audio from text with specified speaker voice.
    
    Parameter priority (highest to lowest):
    1. Payload with override=True: payload values override everything
    2. Config file "api" mode: uses payload values when config says "api"
    3. Config file numeric values: uses config file values
    4. DEFAULT_TTS_PARAMS: default fallback values
    """
    global IGNORE_PING
    try:
        wav_path = None
        start_time = time.perf_counter()
        logger.info(f"Post tts_to_audio - Processing TTS to audio with request: "
                   f"text='{request.text}' speaker_wav='{request.speaker_wav}' "
                   f"language='{request.language}' accent={request.accent} save_path='{request.save_path}' "
                   f"override={request.override}")
        
        if not CURRENT_MODEL:
            raise HTTPException(status_code=500, detail="Model not loaded")

        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")

        if request.text == "ping":
            if IGNORE_PING is None:
                IGNORE_PING = "pending"
            else:
                logger.info("Ping request received, sending silence audio.")            
                return FileResponse(
                    path=SILENCE_AUDIO_PATH,
                    filename=request.save_path,
                    media_type="audio/wav"
                )
        try:
            language = validate_language(request.language or "en")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Generate Audio
        speaker_wav = request.speaker_wav or "malecommoner"
        voice_state = CURRENT_MODEL.get_state_for_audio_prompt(SPEAKERS_AVAILABLE[speaker_wav])
        t_gen0 = time.perf_counter()
        audio = CURRENT_MODEL.generate_audio(voice_state, request.text)
        gen_elapsed = time.perf_counter() - t_gen0
        
        dur_s = float(audio.shape[-1]) / float(CURRENT_MODEL.sample_rate)
        logger.info("Generated {:.3f}s audio in {:.3f}s (rtf={:.2f}x)", dur_s, gen_elapsed, (dur_s / gen_elapsed) if gen_elapsed > 0 else 0.0)        
        
        #TODO, I'm not sure where skyrimnet expects the output to be or how it cleans up old files
        wav_path = OUTPUT_DIRECTORY / f"{datetime.now().strftime("%Y%m%d_%H%M%S")}_{speaker_wav.replace("\\", "_")}.wav"  
        scipy.io.wavfile.write(wav_path, CURRENT_MODEL.sample_rate, audio.numpy())

        if IGNORE_PING == "pending":
            IGNORE_PING = True
            Path(wav_path).unlink(missing_ok=True)
            wav_path = SILENCE_AUDIO_PATH

        return FileResponse(
            path=wav_path,
            filename=wav_path.name,
            media_type="audio/wav"
        )  
              
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /tts_to_audio - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

##@app.post("/create_and_store_latents")
@app.post("/create_and_store_latents")
async def create_and_store_latents(
    speaker_name: str = Form(...),
    language: str = Form("en"),
    wav_file: UploadFile = File(...)
):
    """
    Create and store latent embeddings from uploaded audio file
    """    
    try:
        logger.info(f"POST /create_and_store_latents - Creating and storing latents for speaker: {speaker_name}, language: {language}, file: {wav_file.filename}")
        
        if not CURRENT_MODEL:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Validate language
        try:
            language = validate_language(language)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate file type
        if not wav_file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only WAV files are supported")

        # Check to see if the speaker already exists, if not, we'll add it in
        audio_path = SPEAKER_DIRECTORY.joinpath(f"{speaker_name}.wav")
        if speaker_name not in SPEAKERS_AVAILABLE:

            with open(audio_path, "wb") as buffer:
                content = await wav_file.read()
                buffer.write(content)            
        
            update_available_speakers()
            logger.info(f"Successfully stored wav for speaker: {speaker_name}")

        else: # We do this to not continuously mutate the speaker overtime
            logger.info(f"Speaker already has a stored wav: {speaker_name}")

        return {
            "message": f"Successfully stored '{speaker_name}' in language '{language}'",
            "speaker_name": speaker_name,
            "language": language,
            "latent_shapes": {
                "gpt_cond_latent": [],
                "speaker_embedding": []
            }
        }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /create_and_store_latents - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": CURRENT_MODEL is not None,
        "supported_languages": SUPPORTED_LANGUAGE_CODES
    }


# =============================================================================
# CATCH-ALL ROUTE CONFIGURATION
# =============================================================================

def setup_catch_all_route():
    """
    Set up catch-all route for undefined API endpoints.
    This should only be called when NOT mounting Gradio UI.
    """
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def catch_undefined_endpoints(request: Request, path: str):
        """
        Catch-all route to log attempts to access undefined endpoints
        This helps with debugging missing routes and API discovery
        """
        logger.warning(f"🚫 UNDEFINED ENDPOINT: {request.method} /{path}")
        logger.warning(f"   Full URL: {request.url}")
        logger.warning(f"   Available endpoints:")
        logger.warning(f"     POST /tts_to_audio")
        logger.warning(f"     POST /create_and_store_latents") 
        logger.warning(f"     GET  /health")
        logger.warning(f"     GET  /docs (Swagger UI)")
        logger.warning(f"     GET  /redoc (ReDoc)")
        
        raise HTTPException(
            status_code=404, 
            detail={
                "error": f"Endpoint not found: {request.method} /{path}",
                "available_endpoints": [
                    "POST /tts_to_audio",
                    "GET /health",
                    "GET /docs",
                    "GET /redoc"
                ]
            }
        )


def setup_api_only_catch_all_route():
    """
    Set up a limited catch-all route that only catches API paths when Gradio is mounted.
    This avoids conflicts with Gradio's routing while still providing API endpoint discovery.
    """
    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def catch_undefined_api_endpoints(request: Request, path: str):
        """
        Catch-all route for undefined /api/* endpoints only
        This helps with API debugging without interfering with Gradio
        """
        logger.warning(f"🚫 UNDEFINED API ENDPOINT: {request.method} /api/{path}")
        logger.warning(f"   Full URL: {request.url}")
        logger.warning(f"   Available API endpoints:")
        logger.warning(f"     POST /tts_to_audio")
        logger.warning(f"     GET  /health")
        logger.warning(f"     GET  /docs (Swagger UI)")
        logger.warning(f"     GET  /redoc (ReDoc)")
        
        raise HTTPException(
            status_code=404, 
            detail={
                "error": f"API endpoint not found: {request.method} /api/{path}",
                "available_endpoints": [
                    "POST /tts_to_audio",
                    "POST /create_and_store_latents",
                    "GET /health",
                    "GET /docs",
                    "GET /redoc"
                ],
                "note": "For the Gradio UI, visit the root path '/'"
            }
        )
    
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Pin server process to P-cores so /tts_to_audio runs on performance cores
    pin_process_to_p_cores(logger)

    # Initialize application environment
    initialize_application_environment("SkyrimNet TTS API")
    
    # Set up full catch-all route for standalone API mode
    setup_catch_all_route()
    
    # Get list of available speakers
    update_available_speakers()

    # Load model with standardized initialization
    try:
        CURRENT_MODEL = TTSModel.load_model("skyrimnet")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Start server
    logger.info(f"Starting server on {args.server}:{args.port}")
    uvicorn.run(
        app, 
        host=args.server, 
        port=args.port, 
        log_level="info",
        access_log=False,  # Disable uvicorn's access logging to use our format
        log_config=None,    # Use default Python logging instead of uvicorn's custom format
        lifespan="on",
        reload=False
    )