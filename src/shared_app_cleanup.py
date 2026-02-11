#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        return default


def cleanup_output_directory(
    output_dir: Path,
    max_age_seconds: int,
    glob_pattern: str = "*",
    now_ts: float | None = None,
) -> int:
    """
    Delete files in output_dir older than max_age_seconds.
    Returns number of deleted files.
    """
    if max_age_seconds < 0:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    now_ts = time.time() if now_ts is None else now_ts
    cutoff_ts = now_ts - max_age_seconds

    deleted = 0
    try:
        for p in output_dir.glob(glob_pattern):
            if not p.is_file():
                continue
            try:
                st = p.stat()
                if st.st_mtime < cutoff_ts:
                    p.unlink(missing_ok=True)
                    deleted += 1
            except Exception as e:
                logger.warning(f"Cleanup: failed to delete '{p}': {e}")
    except Exception as e:
        logger.warning(f"Cleanup: failed scanning '{output_dir}': {e}")

    return deleted


async def output_cleanup_worker(
    stop_event: asyncio.Event,
    output_dir: Path,
    max_age_seconds: int,
    interval_seconds: int,
    glob_pattern: str = "*",
) -> None:
    """
    Periodically cleans old files until stop_event is set.
    Sleeps most of the time; wakes promptly on shutdown.
    """
    logger.info(
        "Output cleanup worker started: dir='{}' max_age={}s interval={}s pattern='{}'",
        str(output_dir),
        max_age_seconds,
        interval_seconds,
        glob_pattern,
    )

    while not stop_event.is_set():
        try:
            deleted = await asyncio.to_thread(
                cleanup_output_directory,
                output_dir,
                max_age_seconds,
                glob_pattern,
            )
            logger.info("Output cleanup pass deleted {} file(s).", deleted)
        except Exception as e:
            logger.warning(f"Output cleanup worker iteration failed: {e}")

        # Sleep, but wake promptly when stop_event is set
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass

    logger.info("Output cleanup worker stopped.")
