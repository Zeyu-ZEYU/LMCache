# SPDX-License-Identifier: Apache-2.0
"""Shunt KVLB routing backend (§3.4).

A thin :class:`StorageBackendInterface` that fans a worker's KV transfers out
across several device-bound Mooncake backends -- the worker's own backend NIC,
its on-node neighbours' NICs (borrowed), and the frontend NIC -- according to the
per-iteration plan from Alg. 3. All children connect to the *same* Mooncake
store, differing only in which local RDMA device carries the one-sided transfer,
so any child can serve any key; routing only balances *where* the bytes flow.

Priority (MoE A2A over KV) is not enforced here: it rides the fabric's DSCP
classes (NCCL_IB_TC for the A2A, MC_IB_TC for KV, see the launcher). This backend
only does the spatial split.

The chunk->NIC split comes from :mod:`routing_policy`, fed by a plan the node
planner writes each iteration. Construct via :func:`build_kvlb_backend`.
"""
from __future__ import annotations

# Standard
import asyncio
from concurrent.futures import Future
from typing import Any, Callable, List, Optional, Sequence, Union

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.routing_policy import (
    FileRoutingPlanSource,
    RoutingPlanSource,
    StaticRoutingPlanSource,
    partition_by_weights,
)

logger = init_logger(__name__)


class RoutingBackend(StorageBackendInterface):
    """Routes KV chunks across several device-bound child backends.

    ``children[0]`` is the worker's own backend NIC; the rest are borrowable
    on-node NICs and, last, the frontend NIC. Metadata operations (contains,
    pin, remove, allocator) go to the primary child, since the store is shared.
    """

    def __init__(self, children: List[StorageBackendInterface],
                 plan_source: RoutingPlanSource, dst_device: str = "cuda"):
        super().__init__(dst_device=dst_device)
        assert children, "RoutingBackend needs at least one child backend"
        self.children = children
        self.primary = children[0]
        self.plan = plan_source

    # --- the actual KVLB split -------------------------------------------------

    def _split(self, n: int, direction: str) -> list[list[int]]:
        """Per-child lists of chunk indices for ``n`` chunks, per the plan."""
        weights = self.plan.weights_for(direction).weights
        # pad/truncate weights to the number of children
        if len(weights) < len(self.children):
            weights = list(weights) + [0.0] * (len(self.children) - len(weights))
        child_of = partition_by_weights(n, weights[:len(self.children)])
        buckets: list[list[int]] = [[] for _ in self.children]
        for i, c in enumerate(child_of):
            buckets[c].append(i)
        return buckets

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        buckets = self._split(len(keys), "outbound")   # new-KV write
        futures: List[Future] = []
        for child, idxs in zip(self.children, buckets):
            if not idxs:
                continue
            sub_keys = [keys[i] for i in idxs]
            sub_objs = [objs[i] for i in idxs]
            fs = child.batched_submit_put_task(
                sub_keys, sub_objs, transfer_spec, on_complete_callback)
            if fs:
                futures.extend(fs)
        return futures or None

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        buckets = self._split(len(keys), "inbound")    # prefix-KV read
        results: list[Optional[MemoryObj]] = [None] * len(keys)
        coros = []
        index_lists = []
        for child, idxs in zip(self.children, buckets):
            if not idxs:
                continue
            sub_keys = [keys[i] for i in idxs]
            coros.append(child.batched_get_non_blocking(
                lookup_id, sub_keys, transfer_spec))
            index_lists.append(idxs)
        for idxs, res in zip(index_lists, await asyncio.gather(*coros)):
            for pos, obj in zip(idxs, res):
                results[pos] = obj
        return results  # type: ignore[return-value]

    # --- metadata ops: the store is shared, so the primary suffices ------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        return self.primary.contains(key, pin)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return any(c.exists_in_put_tasks(key) for c in self.children)

    async def batched_async_contains(self, lookup_id: str,
                                     keys: list[CacheEngineKey],
                                     pin: bool = False) -> int:
        return await self.primary.batched_async_contains(lookup_id, keys, pin)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return self.primary.get_blocking(key)

    def get_non_blocking(self, key: CacheEngineKey,
                         location: Optional[str] = None) -> Optional[Future]:
        return self.primary.get_non_blocking(key, location)

    def pin(self, key: CacheEngineKey) -> bool:
        return self.primary.pin(key)

    def unpin(self, key: CacheEngineKey) -> bool:
        return self.primary.unpin(key)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        return self.primary.remove(key, force)

    def get_allocator_backend(self) -> AllocatorBackendInterface:
        return self.primary.get_allocator_backend()

    def close(self) -> None:
        for c in self.children:
            c.close()


# ---------------------------------------------------------------------------
# Factory: build the device-bound children and wire the routing source.
# ---------------------------------------------------------------------------

def build_kvlb_backend(config: LMCacheEngineConfig, metadata: LMCacheMetadata,
                       loop: asyncio.AbstractEventLoop, local_cpu_backend,
                       dst_device: str = "cuda") -> RoutingBackend:
    """Construct a :class:`RoutingBackend` from ``config.extra_config``.

    Expected ``extra_config`` keys:

    - ``shunt_kvlb_devices``: ordered NIC list, own backend NIC first, frontend
      last, e.g. ``["mlx5_bond_0", "mlx5_bond_1", ..., "mlx5_0"]``.
    - ``shunt_kvlb_plan_path``: file the node planner writes each iteration
      (JSON ``{"inbound": [...], "outbound": [...]}``); omit for a static mode.
    - ``shunt_kvlb_static``: optional ``"own"`` / ``"frontend"`` to force all KV
      onto the own backend NIC or the frontend (the measurement / ablation modes).

    Each child is a Mooncake RemoteBackend bound to one device. The store is
    shared across devices, so the per-device binding only selects the local NIC.
    """
    # Imported here so the policy module stays import-light.
    from lmcache.v1.storage_backend.remote_backend import RemoteBackend

    extra = config.extra_config or {}

    # Fig 10 no-contention variant: keep all KV in local CPU, no backend RDMA,
    # so the backend fabric carries only the MoE A2A. Prefix KV is pre-warmed
    # into the local pool, so reuse still hits (§2.3).
    if extra.get("shunt_no_contention"):
        assert local_cpu_backend is not None
        logger.info("Shunt KVLB: no-contention mode (KV stays in local CPU)")
        return RoutingBackend([local_cpu_backend],
                              StaticRoutingPlanSource.all_own(1), dst_device)

    devices: list[str] = list(extra.get("shunt_kvlb_devices", []))
    if len(devices) < 2:
        raise ValueError("shunt_kvlb_devices must list the node's NICs "
                         "(>=1 backend + frontend last)")

    # One config serves all of the node's DP ranks, so rotate the backend NICs
    # per rank: this worker's own NIC first, the rest (nearest first) after, and
    # the frontend (the last entry) always last. This makes child[0] the own NIC,
    # matching the KVLB plan's own/borrowed/frontend ordering (§3.4).
    import os

    rank = int(os.environ.get("VLLM_DP_RANK",
                              os.environ.get("SHUNT_WORKER_RANK", "0")))
    *backends, frontend = devices
    own = rank % len(backends)
    devices = backends[own:] + backends[:own] + [frontend]

    children: List[StorageBackendInterface] = []
    for dev in devices:
        child_cfg = _config_for_device(config, dev)
        children.append(RemoteBackend(child_cfg, metadata, loop,
                                      local_cpu_backend, dst_device,
                                      plugin_name="mooncakestore"))

    static = extra.get("shunt_kvlb_static")
    if static == "own":
        source: RoutingPlanSource = StaticRoutingPlanSource.all_own(len(devices))
    elif static == "frontend":
        source = StaticRoutingPlanSource.all_frontend(len(devices))
    elif extra.get("shunt_kvlb_plan_path"):
        source = FileRoutingPlanSource(extra["shunt_kvlb_plan_path"], len(devices))
    else:
        # default: everything on the own NIC (equivalent to no KVLB)
        source = StaticRoutingPlanSource.all_own(len(devices))

    logger.info("Shunt KVLB: %d device-bound backends %s (source=%s)",
                len(devices), devices, type(source).__name__)
    return RoutingBackend(children, source, dst_device)


def _config_for_device(config: LMCacheEngineConfig, device: str) -> LMCacheEngineConfig:
    """Clone ``config`` with the Mooncake connector bound to ``device``.

    The Mooncake connector reads ``device_name`` from ``extra_config`` (key
    ``mooncake_device_name``); we override just that, keeping everything else.
    """
    import copy

    cfg = copy.copy(config)
    new_extra = dict(config.extra_config or {})
    new_extra["mooncake_device_name"] = device
    cfg.extra_config = new_extra
    return cfg
