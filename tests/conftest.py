"""For every test: no request reaches the elevation or land-cover servers (a test that forgets its fake server fails
instead), and the Pinless cache is a fresh folder inside the test's tmp_path."""
import pytest

from pinless import landcover, relief


@pytest.fixture(autouse=True)
def _offline_and_cached_in_tmp(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("network")

    monkeypatch.setattr(relief, "_fetch", no_network)
    monkeypatch.setattr(landcover, "_fetch", no_network)
    monkeypatch.setenv("PINLESS_CACHE", str(tmp_path / "cache"))
