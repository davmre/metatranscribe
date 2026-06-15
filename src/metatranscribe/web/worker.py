from __future__ import annotations

import logging
import queue
import threading

from metatranscribe.config import Settings
from metatranscribe.orchestrator import run_pipeline

logger = logging.getLogger(__name__)

# Sentinel placed on the queue to ask the worker thread to stop.
_STOP = object()


class PipelineWorker:
    """Runs the (slow, blocking) transcription pipeline off the request thread.

    A single daemon thread drains a wake-up queue and runs ``run_pipeline`` under a
    lock so only one pipeline run executes at a time. ``run_pipeline`` already
    processes *all* pending records, so multiple rapid uploads naturally coalesce
    into a single run -- the queue is just a wake signal, not a per-job channel.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._queue: queue.Queue = queue.Queue()
        self._run_lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="pipeline-worker", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()
        # Resume any records left mid-flight by a previous process.
        self.enqueue()

    def enqueue(self) -> None:
        """Signal the worker that there is pipeline work to do."""
        self._queue.put(None)

    def stop(self) -> None:
        self._queue.put(_STOP)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            # Drain any additional pending wake-ups; one run handles them all.
            self._drain()
            try:
                with self._run_lock:
                    succeeded, failed = run_pipeline(self._settings)
                logger.info("Pipeline run finished succeeded=%d failed=%d", succeeded, failed)
            except Exception:  # noqa: BLE001 -- keep the worker alive across failures
                logger.exception("Pipeline run raised an unexpected error")

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _STOP:
                # Re-queue the stop so the main loop observes it next.
                self._queue.put(_STOP)
                return
