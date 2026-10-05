import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def release_root():
    """Released scores + vectors (downloaded once, ~120 MB). Skipped offline or with CMSTEER_OFFLINE=1."""
    if os.environ.get("CMSTEER_OFFLINE"):
        pytest.skip("offline")
    from cmsteer import hub
    try:
        for m in ("uniar", "emu35", "liquid"):
            hub.fetch(m, ("scores", "vectors"))
    except Exception as e:  # no network
        pytest.skip(f"cannot download the release: {e}")
    return Path(hub.data_root())
