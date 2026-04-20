# SPDX-License-Identifier: Apache-2.0
"""
Route dispatch backend — splits KV chunk operations between two underlying
RemoteBackends (tail + head), one per RNIC "route".

Motivation
----------
The cluster has two RDMA planes per node:
  * tail = GPU-affine bonded RNICs (mlx5_bond_0..3) — the default path
  * head = management-plane RNIC   (mlx5_0)         — the split-off path

We want to ship some KV chunks via the head plane so they don't contend with
the MoE EP all-to-all traffic on tail during prefill / decode forward. The
routing decision is per (layer_id, chunk_id) via
:func:`lmcache.v1.kv_routing.route_kv_chunk` — a deterministic placeholder
function. Producer and consumer MUST agree on the route for any given chunk
because head and tail Mooncake segments are registered on different devices
and are not inter-reachable.

Design: Two RemoteBackends, not one
-----------------------------------
An earlier attempt kept a single ``RemoteBackend`` with two underlying
Mooncake ``Client`` connectors (tail + head) that shared the same pinned
CPU buffer via two ``register_buffer`` calls. That design is fundamentally
incompatible with RDMA semantics: registering the *same* VA range with two
different Protection Domains (one per Client) creates ambiguous MR lookup
on transfer submit, producing Mooncake ``TRANSFER_FAIL`` (status=6) after
the 60 s hard timeout on the tail path.

The correct shape is **two real ``RemoteBackend`` instances**, each owning
its private ``LocalCPUBackend`` → private pinned pool → private MR, each
registered with exactly one device. The wrapper below routes chunks per
:func:`route_kv_chunk` and dispatches to the appropriate backend.

Memory flow
-----------
StorageManager allocates chunks from the *tail* backend's allocator (the
wrapper's ``get_allocator_backend()`` returns the tail allocator). When a
chunk is routed to head:

* **Put**:   memcpy  tail-pool slot  →  head-pool staging slot, then
             ``head_backend.batched_submit_put_task`` uses that staging
             slot as the RDMA WRITE source. Head pool & head Client are
             registered with mlx5_0 only — no MR conflict.

* **Get**:   ``head_backend.batched_get_blocking`` fills a slot in the
             head pool, then memcpy  head-pool slot  →  tail-pool slot
             and return the tail slot. The head slot is released.

The memcpy cost (~0.5–1 ms per 24 MB chunk on a modern CPU) is the
price paid for keeping two clean, independent RDMA MRs. If this overhead
shows up in benchmarks we can later route allocation itself instead
of copying — but for correctness-first this is the simplest working
shape.

route_kv_chunk agreement between producer / consumer
----------------------------------------------------
The routing decision is pure in (layer_id, chunk_id, num_layers,
num_chunks), so producer and consumer running the same function derive
the same route for the same chunk. We rely on that property; the
wrapper never persists the decision.

Bypass behavior
---------------
When ``head_backend is None`` (i.e.
``config.enable_head_nic_split`` was false at construction time), every
call is a straight pass-through to tail. The wrapper in that mode adds
one Python method call per backend op — no branching in the data path.
"""

# Standard
import ctypes
from concurrent.futures import Future
from typing import Any, Callable, List, Optional, Sequence, Union

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.kv_routing import route_kv_chunk
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

logger = init_logger(__name__)


class RouteDispatchBackend(StorageBackendInterface):
    """Dispatches KV chunk operations to tail or head RemoteBackend by route."""

    # StorageManager.batched_put uses this marker to know it's safe to pass
    # layer_id / num_layers kwargs.
    accepts_route_kwargs = True

    def __init__(
        self,
        tail_backend: RemoteBackend,
        head_backend: Optional[RemoteBackend] = None,
    ):
        super().__init__(dst_device=tail_backend.dst_device)
        self.tail = tail_backend
        self.head = head_backend
        if self.head is not None:
            logger.info(
                "RouteDispatchBackend initialized with head+tail "
                "(head=%s tail=%s)",
                self.head.remote_url if hasattr(self.head, "remote_url") else "?",
                self.tail.remote_url if hasattr(self.tail, "remote_url") else "?",
            )
        else:
            logger.info("RouteDispatchBackend initialized (tail-only, no head)")

    # ------------------------------------------------------------------
    # Identity / plumbing
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        # Expose the same string as the underlying primary backend so
        # StorageManager's string-keyed dispatch ("RemoteBackend" entry)
        # keeps working without churn.
        return str(self.tail)

    def get_allocator_backend(self) -> AllocatorBackendInterface:
        """Return the tail backend's allocator.

        All chunks are allocated from the tail pool at StorageManager
        time. Head-routed chunks are staged to head's pool via memcpy
        inside this wrapper.
        """
        return self.tail.get_allocator_backend()

    # ------------------------------------------------------------------
    # Put path
    # ------------------------------------------------------------------

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
        layer_id: Optional[int] = None,
        num_layers: int = 0,
        req_id: Optional[str] = None,
    ) -> Union[List[Future], None]:
        # Bypass when head-split isn't wired up, or when we can't route
        # (layer_id unknown → non-layerwise non-overlap mode).
        if self.head is None or layer_id is None:
            return self.tail.batched_submit_put_task(
                keys,
                objs,
                transfer_spec=transfer_spec,
                on_complete_callback=on_complete_callback,
                layer_id=layer_id,
                num_layers=num_layers,
                req_id=req_id,
            )

        num_chunks = len(keys)
        head_idx: list[int] = []
        tail_idx: list[int] = []
        for ci in range(num_chunks):
            route = route_kv_chunk(layer_id, ci, num_layers, num_chunks)
            if route == "head":
                head_idx.append(ci)
            else:
                tail_idx.append(ci)

        # Tail side: direct pass-through (no staging needed, chunks are
        # already in tail's pool).
        if tail_idx:
            t_keys = [keys[i] for i in tail_idx]
            t_objs = [objs[i] for i in tail_idx]
            self.tail.batched_submit_put_task(
                t_keys,
                t_objs,
                transfer_spec=transfer_spec,
                on_complete_callback=on_complete_callback,
                layer_id=layer_id,
                num_layers=num_layers,
                req_id=req_id,
            )

        # Head side: stage (memcpy tail-pool → head-pool), then submit via
        # the head backend. The head Mooncake Client has its head-pool MR
        # on mlx5_0, so the RDMA WRITE uses mlx5_0.
        if head_idx:
            h_keys = [keys[i] for i in head_idx]
            h_source_objs = [objs[i] for i in head_idx]
            head_staged_objs = self._stage_to_head_pool(h_source_objs)
            if head_staged_objs is None:
                # Staging failed — fall back to tail so we don't lose
                # the chunk. This is a safety valve; the benchmark
                # should not hit it.
                logger.warning(
                    "Head pool staging failed for %d chunks, falling "
                    "back to tail path",
                    len(head_idx),
                )
                self.tail.batched_submit_put_task(
                    h_keys,
                    h_source_objs,
                    transfer_spec=transfer_spec,
                    on_complete_callback=on_complete_callback,
                    layer_id=layer_id,
                    num_layers=num_layers,
                    req_id=req_id,
                )
            else:
                self.head.batched_submit_put_task(
                    h_keys,
                    head_staged_objs,
                    transfer_spec=transfer_spec,
                    on_complete_callback=on_complete_callback,
                    layer_id=layer_id,
                    num_layers=num_layers,
                    req_id=req_id,
                )
        return None

    def _stage_to_head_pool(
        self, source_objs: List[MemoryObj]
    ) -> Optional[List[MemoryObj]]:
        """Allocate slots in head pool and memcpy source chunks into them."""
        assert self.head is not None
        head_alloc = self.head.get_allocator_backend()
        staged: List[MemoryObj] = []
        for src in source_objs:
            shape = src.get_shape()
            dtype = src.get_dtype()
            fmt = src.meta.fmt
            slot = head_alloc.allocate(shape, dtype, fmt=fmt, eviction=True)
            if slot is None:
                # Roll back any partial allocation
                for s in staged:
                    s.ref_count_down()
                return None
            ctypes.memmove(slot.data_ptr, src.data_ptr, src.get_size())
            staged.append(slot)
        return staged

    # ------------------------------------------------------------------
    # Get path
    # ------------------------------------------------------------------

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
        num_layers: int = 0,
    ) -> List[Optional[MemoryObj]]:
        if self.head is None or not keys:
            return self.tail.batched_get_blocking(keys, num_layers=num_layers)

        layer_id = getattr(keys[0], "layer_id", None)
        if layer_id is None:
            return self.tail.batched_get_blocking(keys, num_layers=num_layers)

        num_chunks = len(keys)
        head_idx: list[int] = []
        tail_idx: list[int] = []
        for ci in range(num_chunks):
            route = route_kv_chunk(layer_id, ci, num_layers, num_chunks)
            if route == "head":
                head_idx.append(ci)
            else:
                tail_idx.append(ci)

        results: List[Optional[MemoryObj]] = [None] * num_chunks

        if tail_idx:
            t_keys = [keys[i] for i in tail_idx]
            t_objs = self.tail.batched_get_blocking(t_keys, num_layers=num_layers)
            for i, idx in enumerate(tail_idx):
                results[idx] = t_objs[i] if t_objs else None

        if head_idx:
            h_keys = [keys[i] for i in head_idx]
            h_objs_head_pool = self.head.batched_get_blocking(
                h_keys, num_layers=num_layers
            )
            # Copy head-pool slots back to the tail pool so callers only
            # ever see tail-pool-owned MemoryObjs and can release them
            # via the allocator StorageManager knows about.
            for i, idx in enumerate(head_idx):
                h_obj = h_objs_head_pool[i] if h_objs_head_pool else None
                if h_obj is None:
                    results[idx] = None
                else:
                    results[idx] = self._copy_from_head_pool_to_tail(h_obj)
                    # Release the head-pool slot regardless of copy outcome.
                    h_obj.ref_count_down()

        return results

    def _copy_from_head_pool_to_tail(self, h_obj: MemoryObj) -> Optional[MemoryObj]:
        tail_alloc = self.tail.get_allocator_backend()
        slot = tail_alloc.allocate(
            h_obj.get_shape(),
            h_obj.get_dtype(),
            fmt=h_obj.meta.fmt,
            eviction=True,
        )
        if slot is None:
            logger.warning(
                "Failed to allocate tail-pool slot to copy from head pool"
            )
            return None
        ctypes.memmove(slot.data_ptr, h_obj.data_ptr, h_obj.get_size())
        return slot

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # Try tail first (default path). If head is configured and tail
        # misses, the key might have been Put via head for some layer —
        # try head and copy back.
        obj = self.tail.get_blocking(key)
        if obj is not None or self.head is None:
            return obj
        h_obj = self.head.get_blocking(key)
        if h_obj is None:
            return None
        result = self._copy_from_head_pool_to_tail(h_obj)
        h_obj.ref_count_down()
        return result

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        # Non-blocking path: we don't know the route here (no layer_id),
        # so just try tail. Higher-level code that cares about per-chunk
        # routing should use the batched blocking path.
        return self.tail.get_non_blocking(key, location=location)

    # ------------------------------------------------------------------
    # Existence / pinning / removal — union semantics
    # ------------------------------------------------------------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        if self.tail.contains(key, pin=pin):
            return True
        if self.head is not None and self.head.contains(key, pin=pin):
            return True
        return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        if self.tail.exists_in_put_tasks(key):
            return True
        if self.head is not None and self.head.exists_in_put_tasks(key):
            return True
        return False

    def pin(self, key: CacheEngineKey) -> bool:
        # Pin in whichever backend actually holds the key.
        if self.tail.contains(key):
            return self.tail.pin(key)
        if self.head is not None and self.head.contains(key):
            return self.head.pin(key)
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        ok = self.tail.unpin(key)
        if self.head is not None:
            ok = self.head.unpin(key) or ok
        return ok

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        removed = self.tail.remove(key, force=force)
        if self.head is not None:
            removed = self.head.remove(key, force=force) or removed
        return removed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.tail.close()
        if self.head is not None:
            self.head.close()

    # ------------------------------------------------------------------
    # Per-request Put / Get tracking (timing metrics plumbing)
    # ------------------------------------------------------------------

    def wait_put_done(
        self, req_id: str, timeout: float = 30.0
    ) -> tuple[Optional[float], Optional[float]]:
        """Aggregate per-request put completion times across tail+head.

        Returns ``(first_start_ts, last_end_ts)`` in wall-clock seconds,
        covering whichever backend(s) actually received a Put for this
        req_id. Returns ``(None, None)`` if neither backend tracked the
        request.
        """
        t_start, t_end = self.tail.wait_put_done(req_id, timeout=timeout)
        if self.head is None:
            return t_start, t_end
        h_start, h_end = self.head.wait_put_done(req_id, timeout=timeout)

        starts = [x for x in (t_start, h_start) if x is not None]
        ends = [x for x in (t_end, h_end) if x is not None]
        first_start = min(starts) if starts else None
        last_end = max(ends) if ends else None
        return first_start, last_end

    def wait_get_done(
        self, req_id: str, timeout: float = 30.0
    ) -> tuple[Optional[float], Optional[float]]:
        t_start, t_end = self.tail.wait_get_done(req_id, timeout=timeout)
        if self.head is None:
            return t_start, t_end
        h_start, h_end = self.head.wait_get_done(req_id, timeout=timeout)

        starts = [x for x in (t_start, h_start) if x is not None]
        ends = [x for x in (t_end, h_end) if x is not None]
        first_start = min(starts) if starts else None
        last_end = max(ends) if ends else None
        return first_start, last_end

    # ------------------------------------------------------------------
    # Touch / cache-policy hook
    # ------------------------------------------------------------------

    def touch_cache(self) -> None:
        self.tail.touch_cache()
        if self.head is not None:
            self.head.touch_cache()
