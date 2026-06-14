# SPDX-License-Identifier: Apache-2.0
"""GPU-resident allocator backend for Shunt's GPU-direct KV path (§4.1).

The layerwise store path allocates one MemoryObj per chunk from the storage
manager's allocator backend, copies the paged KV into it, then hands it to the
backends. Default LMCache uses a host (pinned-CPU) allocator, so the Mooncake
transfer reads/writes from host memory. This backend allocates those chunks in a
single GPU pool registered with Mooncake via dma-buf, so:

- the GPU connector copies paged-cache -> GPU pool (GPU-to-GPU), never host;
- the one-sided RDMA write sources straight from the prefill GPU; and
- inbound prefix-KV is read straight into the same GPU pool.

The pool is registered with every device-bound Mooncake store
(:meth:`register_with_store`) so a chunk can be sent over whichever NIC the KVLB
plan routes it to. Used only for allocation; the routing backend performs the
actual transfers, so the storage methods are inert.
"""
from __future__ import annotations

from concurrent.futures import Future
from typing import List, Optional, Sequence, Union

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    TensorMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface

logger = init_logger(__name__)

_CHUNK_BYTES_HINT = 512 * 1024  # ~one 256-token KV chunk/layer, for budget sizing


class GpuAllocatorBackend(AllocatorBackendInterface):
    """Allocates KV chunks in a dma-buf-registered GPU pool."""

    def __init__(self, pool_bytes: int, dst_device: str = "cuda"):
        super().__init__(dst_device=dst_device)
        dev = torch.cuda.current_device()
        self._pool = torch.empty(pool_bytes, dtype=torch.uint8, device=f"cuda:{dev}")
        self._alloc = TensorMemoryAllocator(self._pool)
        self._registered_stores: list = []
        logger.info("Shunt GPU-direct: %d-byte GPU allocator pool on cuda:%d",
                    pool_bytes, dev)

    @property
    def data_ptr(self) -> int:
        return self._pool.data_ptr()

    @property
    def nbytes(self) -> int:
        return self._pool.numel()

    def register_with_store(self, store) -> None:
        """dma-buf-register the pool with a Mooncake store (idempotent per store)."""
        if store in self._registered_stores:
            return
        try:
            rc = store.register_buffer(self._pool.data_ptr(), self._pool.numel())
            self._registered_stores.append(store)
            logger.info("Shunt GPU-direct: registered pool with Mooncake (rc=%s)", rc)
        except Exception as e:  # noqa: BLE001
            logger.error("Shunt GPU-direct: pool registration failed: %s", e)

    # --- allocation (the reason this backend exists) ---------------------------

    def allocate(self, shapes, dtypes, fmt: MemoryFormat = MemoryFormat.KV_2LTD,
                 eviction: bool = True, busy_loop: bool = True
                 ) -> Optional[MemoryObj]:
        return self._alloc.allocate(shapes, dtypes, fmt)

    def batched_allocate(self, shapes, dtypes, batch_size: int,
                         fmt: MemoryFormat = MemoryFormat.KV_2LTD,
                         eviction: bool = True, busy_loop: bool = True
                         ) -> Optional[list[MemoryObj]]:
        return self._alloc.batched_allocate(shapes, dtypes, batch_size, fmt)

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        return self._alloc

    def initialize_allocator(self, config: LMCacheEngineConfig,
                             metadata: LMCacheMetadata) -> MemoryAllocatorInterface:
        return self._alloc

    def calculate_chunk_budget(self) -> int:
        return max(1, self._pool.numel() // _CHUNK_BYTES_HINT)

    # --- storage interface: inert (transfers happen in the routing backend) ----

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

    def batched_submit_put_task(self, keys: Sequence[CacheEngineKey],
                                objs: List[MemoryObj], transfer_spec=None,
                                on_complete_callback=None) -> Union[List[Future], None]:
        return None

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return None

    def pin(self, key: CacheEngineKey) -> bool:
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        return False

    def get_allocator_backend(self) -> "GpuAllocatorBackend":
        return self

    def close(self) -> None:
        self._pool = None
