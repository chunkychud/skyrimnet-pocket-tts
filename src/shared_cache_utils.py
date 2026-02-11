import functools
import threading
import psutil
import torch
from datetime import datetime
import torchaudio
import os
import warnings
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from loguru import logger


def get_model_device(model):
    """Safely get the device from a model, handling different model types."""
    # Cache the device on the model object to avoid repeated parameter iteration
    if hasattr(model, '_cached_device'):
        return model._cached_device
    
    try:
        # Try the device property first (works with our BaseTTS models)
        if hasattr(model, 'device'):
            device = model.device
        else:
            # Fall back to getting device from parameters (only once!)
            device = next(model.parameters()).device
        
        # Cache the result for future calls
        model._cached_device = device
        return device
        
    except Exception:
        # Final fallback to CPU
        logger.warning("Could not determine model device, falling back to CPU")
        device = torch.device('cpu')
        model._cached_device = device
        return device


class LatentCacheManager:
    """in-memory cache manager for latent embeddings."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested operations

    def get(self, language: str, cache_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._cache.get(language, {}).get(cache_key)

    def set(self, language: str, cache_key: str, latents: Dict[str, Any]) -> None:
        with self._lock:
            if language not in self._cache:
                self._cache[language] = {}
            self._cache[language][cache_key] = latents

    def get_all_keys(self) -> List[Tuple[str, str]]:
        with self._lock:
            keys = []
            for lang, lang_cache in self._cache.items():
                for key in lang_cache.keys():
                    keys.append((lang, key))
            return keys

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            total_entries = sum(len(lang_cache)
                                for lang_cache in self._cache.values())
            return {
                'total_entries': total_entries,
                'languages': len(self._cache),
                'languages_list': list(self._cache.keys())
            }


# Global cache manager instance
cache_manager = LatentCacheManager()


@functools.cache
def get_process_creation_time():
    """Get the process creation time as a datetime object"""
    p = psutil.Process(os.getpid())
    return datetime.fromtimestamp(p.create_time())


@functools.cache
def get_latent_dir(language: str = "en") -> Path:
    """Get or create the conditionals cache directory"""
    cache_dir = Path("latents_pt").joinpath(language)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

@functools.cache
def get_speakers_dir(language: str = "en") -> Path:
    """Get or create the speakers directory"""
    speakers_dir = Path("speakers").joinpath(language)
    speakers_dir.mkdir(parents=True, exist_ok=True)
    return speakers_dir

@functools.cache
def get_cache_key(audio_path) -> Optional[str]:
    """Generate a cache key based on audio file"""
    if audio_path is None:
        return None

    cache_prefix = Path(audio_path).stem
    return cache_prefix


def load_pt_latents(path, device):
    expected_shapes = {
        "gpt_cond_latent": torch.Size([1, 32, 1024]),
        "speaker_embedding": torch.Size([1, 512, 1]),
    }
    latents = torch.load(path, map_location=device)
    for key, expected_shape in expected_shapes.items():
        actual_shape = latents[key].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}")
    return latents


def load_json_latents(path, device):
    """Load and convert legacy JSON latents to proper tensor format."""
    expected_shapes = {
        "gpt_cond_latent": torch.Size([1, 32, 1024]),
        "speaker_embedding": torch.Size([1, 512, 1]),
    }
    
    try:
        with open(path, 'r') as f:
            json_data = json.load(f)
        
        # Convert lists back to tensors with proper shapes based on xtts-api-server format
        gpt_cond_latent = torch.tensor(json_data["gpt_cond_latent"], dtype=torch.float32).to(device)
        speaker_embedding = torch.tensor(json_data["speaker_embedding"], dtype=torch.float32).to(device)
        
        # Check if we need reshaping (if loaded from flattened lists)
        if gpt_cond_latent.dim() == 2:  # Flattened format [32, 1024]
            gpt_cond_latent = gpt_cond_latent.unsqueeze(0)  # Add batch dimension [1, 32, 1024]
        elif gpt_cond_latent.dim() == 1:  # Completely flattened format
            gpt_cond_latent = gpt_cond_latent.reshape((-1, 1024)).unsqueeze(0)
            
        if speaker_embedding.dim() == 2:  # Flattened format [512, 1] or [1, 512]
            if speaker_embedding.shape[0] == 1:  # [1, 512] 
                speaker_embedding = speaker_embedding.unsqueeze(-1)  # [1, 512, 1]
            elif speaker_embedding.shape[1] == 1:  # [512, 1]
                speaker_embedding = speaker_embedding.unsqueeze(0)  # [1, 512, 1]
        elif speaker_embedding.dim() == 1:  # Completely flattened [512]
            speaker_embedding = speaker_embedding.unsqueeze(0).unsqueeze(-1)  # [1, 512, 1]
        
        latents = {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding
        }
        
        # Validate shapes
        for key, expected_shape in expected_shapes.items():
            actual_shape = latents[key].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}")
        
        return latents
        
    except (KeyError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to load JSON latents from {path}: {e}")
        raise


def _save_pt_to_disk(filename, data):
    try:
        torch.save(data, filename)
    except Exception as e:
        logger.error(f"Failed to save data: {e}")

@functools.cache
def get_wavout_dir():
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    wavout_dir = Path("output_temp").joinpath(formatted_start_time)
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir


def save_torchaudio_wav(wav_tensor, sr, audio_path) -> Path:
    """Save a tensor as a WAV file using torchaudio"""

    if wav_tensor.device.type != 'cpu':
        wav_tensor = wav_tensor.cpu()

    formatted_now_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path)}"
    path = Path(get_wavout_dir(), f"{filename}.wav")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torchaudio.save(path, wav_tensor, sr, encoding="PCM_S")
    del wav_tensor
    return path #.resolve()
