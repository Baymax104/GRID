import src.utils.distributed as distributed


def test_distributed_barrier_skips_when_distributed_is_not_initialized(monkeypatch):
    barrier_called = False

    def fake_barrier():
        nonlocal barrier_called
        barrier_called = True

    monkeypatch.setattr(distributed, "is_distributed_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "barrier", fake_barrier)

    distributed.distributed_barrier()

    assert not barrier_called


def test_distributed_barrier_calls_torch_barrier_when_distributed_is_initialized(monkeypatch):
    barrier_called = False

    def fake_barrier():
        nonlocal barrier_called
        barrier_called = True

    monkeypatch.setattr(distributed, "is_distributed_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "barrier", fake_barrier)

    distributed.distributed_barrier()

    assert barrier_called
