# SPDX-License-Identifier: Apache-2.0
"""KVLB routing policy: how to spread a layer's KV chunks across a worker's NICs.

This is the dependency-free core of Shunt's KV traffic load balancing (§3.4) on
the LMCache side. Mooncake gives each NIC its own transfer endpoint, all reaching
the same store; routing a chunk to NIC *i* just chooses which local NIC carries
that one-sided RDMA transfer. The node planner runs Alg. 3 and emits, per
direction, the per-NIC byte budget for this worker; :func:`partition_by_weights`
turns those budgets into a concrete per-chunk NIC assignment.

By convention a worker's child backends are ordered ``[own_nic, other_node_nics
..., frontend_nic]``, so weights line up with the planner's
own / borrowed / frontend split.

No torch / lmcache imports here, so the policy is unit-testable on its own.
"""
from __future__ import annotations

import abc
import json
import os
from dataclasses import dataclass


def partition_by_weights(num_chunks: int, weights: list[float]) -> list[int]:
    """Assign each of ``num_chunks`` chunks to a child NIC index.

    Per-NIC counts are proportional to ``weights`` (largest-remainder
    apportionment, so they sum exactly to ``num_chunks``); chunks are handed out
    in contiguous blocks, which keeps each NIC's transfers together. A NIC with
    zero weight gets no chunks. Falls back to NIC 0 if all weights are zero.
    """
    n = len(weights)
    if num_chunks <= 0 or n == 0:
        return []
    total = sum(weights)
    if total <= 0:
        return [0] * num_chunks
    raw = [num_chunks * w / total for w in weights]
    counts = [int(x) for x in raw]
    rem = num_chunks - sum(counts)
    # hand the leftover chunks to the largest fractional remainders
    order = sorted(range(n), key=lambda i: raw[i] - counts[i], reverse=True)
    for k in range(rem):
        counts[order[k]] += 1
    out: list[int] = []
    for i, c in enumerate(counts):
        out.extend([i] * c)
    return out


@dataclass
class RoutingWeights:
    """Per-child byte budgets for one direction, in child order."""

    weights: list[float]

    def normalized(self) -> list[float]:
        s = sum(self.weights)
        return [w / s for w in self.weights] if s > 0 else self.weights


class RoutingPlanSource(abc.ABC):
    """Supplies the current per-NIC weights for inbound / outbound KV."""

    @abc.abstractmethod
    def weights_for(self, direction: str) -> RoutingWeights:
        """``direction`` is "inbound" (prefix-KV read) or "outbound" (new-KV write)."""
        raise NotImplementedError


class StaticRoutingPlanSource(RoutingPlanSource):
    """Fixed weights.

    Used for the measurement modes (all on the own backend NIC, or all on the
    frontend) and the /KVLB ablation arm (everything on the backend), where the
    split does not change across iterations.
    """

    def __init__(self, inbound: list[float], outbound: list[float]):
        self._w = {"inbound": RoutingWeights(inbound),
                   "outbound": RoutingWeights(outbound)}

    def weights_for(self, direction: str) -> RoutingWeights:
        return self._w[direction]

    @classmethod
    def all_own(cls, num_children: int) -> "StaticRoutingPlanSource":
        w = [1.0] + [0.0] * (num_children - 1)
        return cls(list(w), list(w))

    @classmethod
    def all_frontend(cls, num_children: int) -> "StaticRoutingPlanSource":
        w = [0.0] * (num_children - 1) + [1.0]
        return cls(list(w), list(w))


class FileRoutingPlanSource(RoutingPlanSource):
    """Reads weights the node planner writes once per iteration.

    The planner atomically writes a small JSON ``{"inbound": [...], "outbound":
    [...]}`` for this worker; we re-read it only when its mtime changes, so the
    per-chunk lookup stays cheap. Until the first plan appears, all KV rides the
    own backend NIC.
    """

    def __init__(self, path: str, num_children: int):
        self.path = path
        self.num_children = num_children
        self._mtime = 0.0
        self._cache = {"inbound": RoutingWeights(self._default()),
                       "outbound": RoutingWeights(self._default())}

    def _default(self) -> list[float]:
        return [1.0] + [0.0] * (self.num_children - 1)

    def _refresh(self) -> None:
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return
        if m == self._mtime:
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._cache = {
                "inbound": RoutingWeights(d.get("inbound", self._default())),
                "outbound": RoutingWeights(d.get("outbound", self._default())),
            }
            self._mtime = m
        except (OSError, ValueError):
            pass

    def weights_for(self, direction: str) -> RoutingWeights:
        self._refresh()
        return self._cache[direction]
