# SPDX-License-Identifier: Apache-2.0
"""Private KV transfer pools for Shunt's per-port Mooncake clients.

Each device-bound Mooncake client of the KVLB routing backend owns one pool,
registered with that client only: a GPU pool for backend ports (GPU-direct
RDMA) or a pinned host pool for the frontend, which is reached through the
host. Registering the same memory with two clients of one process is avoided
on purpose.
"""

# Standard
from typing import List, Optional
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator

logger = init_logger(__name__)


class KVPool:
    """A flat buffer with a thread-safe sub-allocator."""

    def __init__(self, nbytes: int, on_gpu: bool, name: str):
        self.name = name
        self.on_gpu = on_gpu
        if on_gpu:
            dev = torch.cuda.current_device()
            self.buffer = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dev}")
        else:
            self.buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self._alloc = TensorMemoryAllocator(self.buffer)
        self._lock = threading.Lock()
        self.lo = self.buffer.data_ptr()
        self.hi = self.lo + self.buffer.numel()
        logger.info(
            "Shunt KV pool %s: %d MiB on %s",
            name,
            nbytes >> 20,
            "GPU" if on_gpu else "pinned host",
        )

    def owns(self, ptr: int) -> bool:
        return self.lo <= ptr < self.hi

    def register_with_store(self, store) -> None:
        rc = store.register_buffer(self.lo, self.buffer.numel())
        if rc != 0:
            raise RuntimeError(f"Mooncake register_buffer failed for {self.name}: {rc}")

    def allocate(self, shapes, dtypes, fmt: MemoryFormat = MemoryFormat.KV_2LTD):
        with self._lock:
            return self._alloc.allocate(shapes, dtypes, fmt)

    def batched_allocate(
        self,
        shapes,
        dtypes,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        wait_s: float = 0.0,
    ) -> Optional[List]:
        """Allocate ``batch_size`` objects; retry up to ``wait_s`` seconds."""
        deadline = time.monotonic() + wait_s
        while True:
            with self._lock:
                objs = self._alloc.batched_allocate(shapes, dtypes, batch_size, fmt)
            if objs is not None or time.monotonic() >= deadline:
                return objs
            time.sleep(0.0005)
