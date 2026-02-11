import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

def cleanup_output_directory(
    output_dir: Path,
    max_age_minutes: int,
    now: datetime | None = None,
) -> int:
    """
    Delete files older than max_age_minutes in output_dir.
    Returns count deleted.
    Safe to call from both the periodic worker and teardown.
    """
    if max_age_minutes < 0:
        return 0

    now = now or datetime.now()
    cutoff = now - timedelta(minutes=max_age_minutes)

    deleted = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in output_dir.glob("*.wav"):
        if not p.is_file():
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            if mtime < cutoff:
                p.unlink(missing_ok=True)
                deleted += 1
                logger.info(f"Deleted {p.name}")
        except Exception as e:
            logger.warning(f"Cleanup: failed for '{p}': {e}")

    return deleted


async def output_cleanup_worker(
    stop_event: asyncio.Event,
    output_dir: Path,
    max_age_minutes: int,
    interval_seconds: int,
) -> None:
    logger.info(
        f"Output cleanup worker STARTED dir={output_dir} max_age={max_age_minutes}m interval={interval_seconds}s"
    )

    while not stop_event.is_set():
        try:
            deleted = await asyncio.to_thread(cleanup_output_directory, output_dir, max_age_minutes)
            if deleted:
                logger.info(f"Output cleanup deleted {deleted} file(s).")
        except Exception:
            logger.exception("Output cleanup worker iteration failed")

        # Sleep, but wake immediately on shutdown
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass

    logger.info("Output cleanup worker STOPPED")


