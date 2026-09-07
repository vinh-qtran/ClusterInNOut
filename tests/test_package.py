from __future__ import annotations

import importlib.metadata

import clusterinnout as m


def test_version() -> None:
    assert importlib.metadata.version("clusterinnout") == m.__version__
