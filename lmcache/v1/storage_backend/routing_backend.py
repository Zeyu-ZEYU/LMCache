# SPDX-License-Identifier: Apache-2.0
"""Shunt KVLB routing backend.

One ``RemoteBackend`` (a Mooncake store client) per RDMA device of the prefill
node: the node's backend devices and the frontend device. All clients reach the
same Mooncake store, so any of them can read or write any key; the device only
decides which NIC carries the transfer. Each client owns a private transfer
pool (``shunt_pool.KVPool``): GPU memory for backend devices, so transfers are
GPU-direct, and pinned host memory for the frontend device.

For every KV chunk, the Shunt runtime's router (``shunt.runtime.kv_router``)
picks a port from the iteration's KVLB plan; the chunk then moves through the
client of that port's device. Outbound chunks are allocated in that client's
pool when LMCache stores the layer (``allocate_chunk``), so the RDMA write
needs no copy.

Configuration (``extra_config``):

- ``shunt_kvlb_port_devices``: device of every backend port of a node, indexed
  by the node-local GPU index, e.g. ``["mlx5_bond_0", "mlx5_bond_0", ...]``.
- ``shunt_kvlb_frontend_device``: the frontend device (optional).
- ``shunt_gpu_direct``: backend clients use GPU pools (default ``true``).
- ``shunt_gpu_pool_mb`` / ``shunt_host_pool_mb``: pool size per client.
- ``shunt_kv_tc_own`` / ``shunt_kv_tc_borrow``: RDMA traffic class of KV on the
  client of the worker's own device and on the other devices (optional). This
  needs the patched Mooncake, which reads ``MC_IB_TC`` per transfer engine and
  gives both directions of a connection the class of the side that opened it,
  so RDMA READ responses (inbound prefix-KV) carry the same class.
"""

# Standard
from typing import Any, Callable, Dict, List, Optional, Sequence
import asyncio
import copy
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.shunt_pool import KVPool

logger = init_logger(__name__)

IN, OUT = "in", "out"


def _router():
    try:
        # Third Party
        from shunt.runtime import kv_router
    except ImportError:
        return None
    return kv_router.get()


def _nbytes(obj: MemoryObj) -> int:
    t = getattr(obj, "tensor", None)
    if t is not None:
        return t.numel() * t.element_size()
    return obj.get_size()


def _dp_rank() -> int:
    for var in ("VLLM_DP_RANK", "SHUNT_DP_RANK"):
        if os.environ.get(var):
            return int(os.environ[var])
    try:
        # Third Party
        from vllm.distributed.parallel_state import get_dp_group

        return get_dp_group().rank_in_group
    except Exception:
        return 0


class RoutingBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend,
        dst_device: str = "cuda",
    ):
        super().__init__(dst_device=dst_device)
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else None
        extra = dict(config.extra_config or {})
        self.port_devices: List[str] = list(extra["shunt_kvlb_port_devices"])
        fe_device = extra.get("shunt_kvlb_frontend_device")
        gpu_direct = bool(extra.get("shunt_gpu_direct", True))
        gpu_pool = int(extra.get("shunt_gpu_pool_mb", 2048)) << 20
        host_pool = int(extra.get("shunt_host_pool_mb", 4096)) << 20
        tc_own = extra.get("shunt_kv_tc_own")
        tc_borrow = extra.get("shunt_kv_tc_borrow")

        self.local_cpu_backend = local_cpu_backend
        self.router = _router()
        wpn = len(self.port_devices)
        own = self.router.own_port if self.router is not None else _dp_rank() % wpn
        self.own_port = own
        own_dev = self.port_devices[own]
        kv_shape = metadata.kv_shape  # (layers, 2, chunk, heads, head_size)
        itemsize = metadata.kv_dtype.itemsize
        self.chunk_bytes = kv_shape[1] * kv_shape[2] * kv_shape[3] * kv_shape[4] * itemsize

        # One client per device: own device first, then the others in port order.
        order = [own_dev] + [d for d in dict.fromkeys(self.port_devices) if d != own_dev]
        self.children: List[Any] = []
        self.pools: List[KVPool] = []
        self.names: List[str] = []
        child_of_device: Dict[str, int] = {}
        for dev in order:
            tc = tc_own if dev == own_dev else tc_borrow
            child_of_device[dev] = self._add_child(
                config, metadata, loop, dst_device, dev, tc,
                KVPool(gpu_pool if gpu_direct else host_pool, gpu_direct, dev))
        self.port_child = [child_of_device[d] for d in self.port_devices]
        self.fe_child: Optional[int] = None
        if fe_device:
            self.fe_child = self._add_child(
                config, metadata, loop, dst_device, fe_device, tc_own,
                KVPool(host_pool, False, fe_device))
        if self._device is not None:
            torch.cuda.set_device(self._device)
        self.primary = self.children[0]
        self._put_child: Dict[Any, int] = {}
        self._lock = threading.Lock()
        self._io: Dict[tuple, list] = {}
        self._io_seq = -1
        logger.info(
            "Shunt KVLB: own port %d, clients %s, gpu_direct=%s",
            own, self.names, gpu_direct,
        )

    def _add_child(self, config, metadata, loop, dst_device, dev, tc, pool) -> int:
        # First Party
        from lmcache.v1.storage_backend.remote_backend import RemoteBackend

        cfg = copy.copy(config)
        extra = dict(config.extra_config or {})
        extra["mooncake_device_name"] = dev
        extra["shunt_private_pool"] = True
        cfg.extra_config = extra
        saved = os.environ.get("MC_IB_TC")
        if tc is not None:
            os.environ["MC_IB_TC"] = str(tc)
        # Mooncake may switch the current CUDA device while it probes the
        # topology; keep this worker's device.
        device = self._device
        try:
            child = RemoteBackend(cfg, metadata, loop, self.local_cpu_backend,
                                  dst_device)
        finally:
            if device is not None:
                torch.cuda.set_device(device)
            if tc is not None:
                if saved is None:
                    os.environ.pop("MC_IB_TC", None)
                else:
                    os.environ["MC_IB_TC"] = saved
        if child.connection is None:
            raise RuntimeError(f"Shunt KVLB: no Mooncake connection on {dev}")
        conn = getattr(child.connection, "_connector", child.connection)
        conn.attach_pool(pool)
        self.children.append(child)
        self.pools.append(pool)
        self.names.append(dev)
        return len(self.children) - 1

    def __str__(self) -> str:
        return "RoutingBackend"

    # --- routing ---------------------------------------------------------------

    def _child_for(self, direction: str, key: CacheEngineKey) -> int:
        if self.router is None:
            return self.port_child[self.own_port]
        port = self.router.route(direction, key.chunk_hash, self.chunk_bytes)
        if port < 0:
            if self.fe_child is not None:
                return self.fe_child
            port = self.own_port
        return self.port_child[port]

    def allocate_chunk(
        self,
        key: CacheEngineKey,
        shape,
        dtype,
        num_layers: int,
        fmt: MemoryFormat,
    ) -> Optional[List[MemoryObj]]:
        """Allocate one chunk's per-layer objects in the pool of the client
        that will send it; fall back to other pools, then to the default
        allocator, when a pool is full."""
        first = self._child_for(OUT, key)
        order = [first] + [i for i in range(len(self.children)) if i != first]
        for i, idx in enumerate(order):
            objs = self.pools[idx].batched_allocate(
                shape, dtype, num_layers, fmt, wait_s=0.05 if i == 0 else 0.0)
            if objs is not None:
                with self._lock:
                    self._put_child[key.chunk_hash] = idx
                return objs
        return None

    # --- data path ----------------------------------------------------------------

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ):
        groups: Dict[int, tuple] = {}
        for k, o in zip(keys, objs, strict=False):
            with self._lock:
                idx = self._put_child.get(k.chunk_hash)
            if idx is None:
                idx = self._child_for(OUT, k)
            groups.setdefault(idx, ([], []))
            groups[idx][0].append(k)
            groups[idx][1].append(o)
        for idx, (ks, os_) in groups.items():
            t0 = time.time()
            rec = self._io_record(OUT, idx, len(ks), sum(_nbytes(o) for o in os_), t0)
            pending = [len(ks)]

            def done(_key, rec=rec, pending=pending, cb=on_complete_callback):
                pending[0] -= 1
                if pending[0] == 0:
                    rec[3] = max(rec[3], time.time())
                if cb is not None:
                    cb(_key)

            self.children[idx].batched_submit_put_task(
                ks, os_, transfer_spec, on_complete_callback=done)
        return None

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> List[MemoryObj]:
        groups: Dict[int, List[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(self._child_for(IN, k), []).append(i)
        results: List[Optional[MemoryObj]] = [None] * len(keys)
        t0 = time.time()

        async def one(idx: int, pos: List[int]):
            got = await self.children[idx].batched_get_non_blocking(
                lookup_id, [keys[i] for i in pos], transfer_spec)
            nbytes = sum(_nbytes(o) for o in got if o is not None)
            rec = self._io_record(IN, idx, len(pos), nbytes, t0)
            rec[3] = max(rec[3], time.time())
            return pos, got

        for pos, got in await asyncio.gather(*(one(i, p) for i, p in groups.items())):
            for i, obj in zip(pos, got, strict=False):
                results[i] = obj
        # LMCache expects a prefix of hits: stop at the first miss
        out: List[MemoryObj] = []
        for obj in results:
            if obj is None:
                break
            out.append(obj)
        for obj in results[len(out):]:
            if obj is not None:
                obj.ref_count_down()
        return out

    # --- per-iteration transfer log ------------------------------------------------

    def _io_record(self, direction: str, child: int, n: int, nbytes: int,
                   t0: float) -> list:
        """Aggregate transfers per (iteration, direction, client); the Shunt
        runtime writes them to ``kvio.rank<r>.jsonl`` when the next plan comes."""
        seq = self.router.seq if self.router is not None else -1
        with self._lock:
            if seq != self._io_seq:
                self._flush_io()
                self._io_seq = seq
            rec = self._io.get((direction, child))
            if rec is None:
                rec = self._io[(direction, child)] = [0, 0, t0, t0]
            rec[0] += n
            rec[1] += nbytes
            rec[2] = min(rec[2], t0)
            return rec

    def _flush_io(self) -> None:
        if not self._io or self.router is None:
            self._io = {}
            return
        logs = self.router.rt.logs
        if logs.enabled():
            logs.write("kvio", {
                "seq": self._io_seq,
                "io": [{"dir": d, "dev": self.names[c], "chunks": r[0],
                        "bytes": r[1], "first": r[2], "last": r[3]}
                       for (d, c), r in self._io.items()],
            })
        self._io = {}

    # --- metadata: the store is shared, so the own client answers ----------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        return self.primary.contains(key, pin)

    def batched_contains(self, keys: List[CacheEngineKey], pin: bool = False) -> int:
        return self.primary.batched_contains(keys, pin)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return any(c.exists_in_put_tasks(key) for c in self.children)

    async def batched_async_contains(
        self, lookup_id: str, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        return await self.primary.batched_async_contains(lookup_id, keys, pin)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return self.children[self._child_for(IN, key)].get_blocking(key)

    def batched_get_blocking(self, keys: List[CacheEngineKey]):
        fut = asyncio.run_coroutine_threadsafe(
            self.batched_get_non_blocking("blocking", keys), self.primary.loop)
        return fut.result()

    def pin(self, key: CacheEngineKey) -> bool:
        return self.primary.pin(key)

    def unpin(self, key: CacheEngineKey) -> bool:
        return self.primary.unpin(key)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        return self.primary.remove(key, force)

    def get_allocator_backend(self):
        # Objects come allocated in the right pool (allocate_chunk); returning
        # the storage manager's allocator keeps it from copying them.
        return self.local_cpu_backend

    def close(self) -> None:
        for c in self.children:
            c.close()
