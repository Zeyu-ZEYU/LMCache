"""
KV chunk routing for head/tail NIC splitting.

The routing function decides, per (layer_id, chunk_id), whether a KV chunk
should be transferred via the head NIC (management RNIC, mlx5_0) or the
tail NICs (GPU-affine bonded RNICs, mlx5_bond_0~3).

Requirements
------------

- **Deterministic in (layer_id, chunk_id)**: prefill and decode run this
  function independently and must reach the same decision for the same
  chunk. No randomness-with-state, no globals. Different layers may
  choose differently; for a fixed layer, different chunks may choose
  differently. But given the same (layer_id, chunk_id), the answer must
  be identical on both sides.

- **Batch interface**: called ONCE per layer-level batch, returning two
  disjoint index lists. This keeps the per-chunk dispatch cost on the
  wrapper (:class:`RouteDispatchBackend`) side O(1) for the common
  "all tail" / "all head" case, and O(num_chunks) only when the routing
  is actually heterogeneous.

- **Default**: all chunks → tail (backward compatible with the
  non-split setup).
"""


def route_kv_chunks(
    layer_id: int,
    num_layers: int,
    num_chunks: int,
) -> tuple[list[int], list[int]]:
    """
    Partition the chunks of one layer-level batch into head and tail routes.

    Args:
        layer_id:   Current layer index (0 .. num_layers-1). In
                    non-layerwise (non-overlap) mode, RouteDispatchBackend
                    passes ``layer_id=0`` and ``num_layers=1`` — the
                    whole request is treated as one synthetic "layer 0".
        num_layers: Total number of model layers (or 1 in non-layerwise).
        num_chunks: Number of KV chunks in this batch.

    Returns:
        ``(head_indices, tail_indices)``: two disjoint lists of chunk
        positions (each in ``[0, num_chunks)``) whose union equals
        ``range(num_chunks)``. Chunks at positions in ``head_indices``
        are shipped via mlx5_0; those in ``tail_indices`` via the bonded
        mlx5_bond_* group.

    Determinism contract:
        For the same ``(layer_id, chunk_id)`` pair, prefill and decode
        must classify the chunk identically. This function is pure; use
        only the arguments to decide, never wall-clock time, hostname,
        RNG state without a seed pinned to (layer_id, chunk_id), etc.

    Example implementations
    -----------------------

    All tail (default, used below)::

        return [], list(range(num_chunks))

    All head::

        return list(range(num_chunks)), []

    First 30% of each layer to head (deterministic, O(1) cheap)::

        head_cutoff = int(num_chunks * 0.30)
        return (
            list(range(head_cutoff)),
            list(range(head_cutoff, num_chunks)),
        )

    Hash-based 30% per (layer, chunk) — useful when you want head-routed
    chunks scattered rather than clumped at the front::

        head_idx, tail_idx = [], []
        for c in range(num_chunks):
            # Hash collapses to a stable 16-bit int; compare to the
            # ratio's 16-bit equivalent. Purely (layer, chunk) driven,
            # so prefill and decode see the same partition.
            if (hash((layer_id, c)) & 0xFFFF) < int(0.30 * 0xFFFF):
                head_idx.append(c)
            else:
                tail_idx.append(c)
        return head_idx, tail_idx

    Layer-stripe (even layers head, odd layers tail)::

        if layer_id % 2 == 0:
            return list(range(num_chunks)), []
        return [], list(range(num_chunks))
    """
    # EXPERIMENT (no-overlap full-matrix rerun v3 — head mode):
    # paired with the tail run via commit swap:
    #     all-tail swap:  return [], list(range(num_chunks))
    return list(range(num_chunks)), []
