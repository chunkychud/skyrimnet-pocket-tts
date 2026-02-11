# =============================================================================
import torch
import time
from loguru import logger
from typing import Optional
from pathlib import Path
from pocket_tts import TTSModel
try:
    from .shared_cache_utils import save_torchaudio_wav # get_latent_from_audio, 
    from .shared_config import DEFAULT_TTS_PARAMS
    
except ImportError:
    from shared_cache_utils import save_torchaudio_wav #get_latent_from_audio, 
    from shared_config import DEFAULT_TTS_PARAMS

# =============================================================================

DEFAULT_CHAR_LIMIT = 500
ENABLE_TEXT_SPLITTING = True

# def generate_audio_file(
#     model: TTSModel,
#     language: str,
#     speaker_wav: str,
#     text: str,
#     stream: bool = False,
#     **inference_kwargs
# ) -> Path:
#     """
#     Generate audio file using streaming inference with CUDA optimization to avoid CPU transfers.
    
#     Args:
#         model: The TTS model to use for inference
#         language: Language code for synthesis
#         speaker_wav: Speaker reference (file path or speaker name)
#         text: Text to synthesize
#         **inference_kwargs: Additional parameters for model.inference_stream()
#             Supported kwargs:
#             - temperature: Controls randomness (default from DEFAULT_TTS_PARAMS)
#             - top_p: Nucleus sampling parameter (default from DEFAULT_TTS_PARAMS)
#             - top_k: Top-k sampling parameter (default from DEFAULT_TTS_PARAMS)
#             - speed: Speech speed (default from DEFAULT_TTS_PARAMS)
#             - repetition_penalty: Penalty for repetition (default from DEFAULT_TTS_PARAMS)
#             - enable_text_splitting: Whether to enable text splitting (auto-detected if not provided)
#             - stream_chunk_size: Chunk size for streaming (default: 20)
#             - overlap_wav_len: Overlap length for streaming (default: 1024)
    
#     Returns:
#         tuple: (Path to the generated audio file, audio length in seconds)
#     """
    
#     # Start timing the entire function
#     func_start_time = time.perf_counter()

#     output_sample_rate = model.sample_rate

#     logger.info(f"Generating audio for text='{text[:50]}...', speaker='{Path(speaker_wav).stem}', language='{language}', stream={stream}")
    
#     # Check text length and enable splitting if needed
#     if ENABLE_TEXT_SPLITTING and len(text) > DEFAULT_CHAR_LIMIT:
#         logger.info(f"Text length {len(text)} exceeds limit {DEFAULT_CHAR_LIMIT}, enabling text splitting")
    
#     wav_out_path = save_torchaudio_wav(
#         wav_tensor=wav_out.unsqueeze(0),
#         sr=output_sample_rate,
#         audio_path=speaker_wav,
#     )
#     wav_length_s = wav_out.shape[0] / output_sample_rate

#     func_end_time = time.perf_counter()
#     total_duration_s = func_end_time - func_start_time
#     if speaker_wav:
#         input_wav = speaker_wav.split('\\')[-1]
#         logger.info(f"Total 'generate_audio' output of {wav_out_path} for {input_wav} length: {wav_length_s:.2f}s execution time: {total_duration_s:.2f}s Speed: {wav_length_s/total_duration_s:.2f}x")
#     else:
#         logger.info(f"Total 'generate_audio' execution time: {total_duration_s:.2f} seconds")

#     del wav_out
#     if wav_chunks is not None:
#         del wav_chunks
#     return wav_out_path
