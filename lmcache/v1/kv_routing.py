"""
KV chunk routing for head/tail NIC splitting.

The routing function decides, per (layer_id, chunk_id), whether a KV chunk
should be transferred via the head NIC (management RNIC, mlx5_0) or the
tail NICs (GPU-affine bonded RNICs, mlx5_bond_0~3).

Requirements:
  - Deterministic: same inputs → same output on both prefill and decode.
  - Default: all chunks go to "tail" (backward compatible).
  - Easy to modify for experiments (e.g., all "head", or percentage-based).
"""


def route_kv_chunk(
    layer_id: int,
    chunk_id: int,
    num_layers: int,
    num_chunks: int,
) -> str:
    """
    Decide whether a KV chunk goes via "head" or "tail" NIC.

    Args:
        layer_id:   Layer index (0 .. num_layers-1).
        chunk_id:   Chunk index within this layer (0 .. num_chunks-1).
        num_layers: Total number of model layers.
        num_chunks: Total number of chunks in this store operation.

    Returns:
        "head" or "tail".

    NOTE: This is a placeholder. To experiment:
      - Return "head" unconditionally to route all KV via head NIC.
      - Use a percentage: e.g., return "head" if chunk_id < num_chunks * 0.3
        to send 30% of chunks via head NIC.
    """
    return "tail"
