"""Optional NVIDIA NVTX ranges; zero runtime dependency when disabled."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator


@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    if os.environ.get("ENGINE_NVTX") != "1":
        yield
        return
    try:
        import torch

        torch.cuda.nvtx.range_push(name)
    except (ImportError, RuntimeError):
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(RuntimeError):
            torch.cuda.nvtx.range_pop()
