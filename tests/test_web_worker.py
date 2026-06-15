import threading
from pathlib import Path

from metatranscribe.web.worker import PipelineWorker
from tests.test_web_app import _settings


def test_enqueue_runs_pipeline(tmp_path, monkeypatch):
    calls = []
    done = threading.Event()

    def fake_run_pipeline(settings):
        calls.append(settings)
        done.set()
        return (1, 0)

    monkeypatch.setattr("metatranscribe.web.worker.run_pipeline", fake_run_pipeline)

    worker = PipelineWorker(_settings(tmp_path))
    worker.start()  # start() enqueues one resume run

    assert done.wait(timeout=5)
    worker.stop()
    assert len(calls) >= 1


def test_worker_survives_exceptions(tmp_path, monkeypatch):
    results = []
    first = threading.Event()
    second = threading.Event()

    def flaky_run_pipeline(settings):
        if not first.is_set():
            first.set()
            raise RuntimeError("boom")
        results.append("ok")
        second.set()
        return (0, 0)

    monkeypatch.setattr("metatranscribe.web.worker.run_pipeline", flaky_run_pipeline)

    worker = PipelineWorker(_settings(tmp_path))
    worker.start()  # triggers first run, which raises
    assert first.wait(timeout=5)

    worker.enqueue()  # thread should still be alive to handle this
    assert second.wait(timeout=5)
    worker.stop()
    assert results == ["ok"]


def test_only_one_run_at_a_time(tmp_path, monkeypatch):
    overlap = []
    running = threading.Lock()
    release = threading.Event()
    started = threading.Event()

    def slow_run_pipeline(settings):
        acquired = running.acquire(blocking=False)
        overlap.append(acquired)
        started.set()
        release.wait(timeout=5)
        if acquired:
            running.release()
        return (0, 0)

    monkeypatch.setattr("metatranscribe.web.worker.run_pipeline", slow_run_pipeline)

    worker = PipelineWorker(_settings(tmp_path))
    worker.start()
    assert started.wait(timeout=5)
    worker.enqueue()  # coalesces; serialized by the worker's run lock anyway
    release.set()
    worker.stop()
    assert all(overlap)  # every run acquired the lock without overlap
