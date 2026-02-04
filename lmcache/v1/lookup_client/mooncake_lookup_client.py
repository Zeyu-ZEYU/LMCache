# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


class MooncakeLookupClient(LookupClientInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        master_addr: str,
    ):
        # Third Party
        from mooncake.store import MooncakeDistributedStore

        self.store = MooncakeDistributedStore()
        self.store.setup(
            "localhost",
            "P2PHANDSHAKE",
            0,
            16 * 1024 * 1024,
            "tcp",
            "",
            master_addr,
        )

        # Initialize token database for processing tokens
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed."
        )

        # First Party
        from lmcache.v1.token_database import ChunkedTokenDatabase

        assert not config.enable_blending, (
            "LMCache v1 blending is not supported in MooncakeLookupClient yet."
        )
        self.token_database = ChunkedTokenDatabase(config, metadata)
        # Needed to check layerwise keys when producer stores per-layer KV.
        self.num_layers = metadata.kv_shape[0]

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: Optional[str] = None,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        # Process token_ids to CacheEngineKeys.
        processed: list[tuple[int, CacheEngineKey]] = []
        for _, end, key in self.token_database.process_tokens(
            tokens=token_ids,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            processed.append((end, key))

        if not processed:
            return 0

        ends = [end for end, _ in processed]
        base_key_strs = [key.to_string() for _, key in processed]

        # Batch check base keys.
        # rets is list of int: 1 = found, 0 = not found, -1 = error
        base_rets = self.store.batch_is_exist(base_key_strs)

        # For chunks missing the base key, also check if all layerwise keys exist.
        missing_indices = [i for i, r in enumerate(base_rets) if r != 1]
        layerwise_ok: dict[int, bool] = {}
        if missing_indices:
            logger.debug(
                "Mooncake lookup: %d base chunks missing, checking layerwise keys",
                len(missing_indices),
            )
            layerwise_flat: list[str] = []
            offsets: list[tuple[int, int]] = []
            for i in missing_indices:
                _, key = processed[i]
                keys_multi = key.split_layers(self.num_layers)
                offsets.append((i, len(layerwise_flat)))
                layerwise_flat.extend([k.to_string() for k in keys_multi])

            layerwise_rets = (
                self.store.batch_is_exist(layerwise_flat)
                if layerwise_flat
                else []
            )
            for i, off in offsets:
                all_exist = True
                for r in layerwise_rets[off : off + self.num_layers]:
                    if r != 1:
                        all_exist = False
                        break
                layerwise_ok[i] = all_exist

        # Prefix match: stop at first missing chunk.
        for i in range(len(processed)):
            if base_rets[i] == 1:
                continue
            if layerwise_ok.get(i, False):
                continue
            logger.debug(
                "Mooncake lookup: prefix hit ends at chunk %d (end=%d)",
                i - 1,
                ends[i - 1] if i > 0 else 0,
            )
            return ends[i - 1] if i > 0 else 0

        logger.debug("Mooncake lookup: full prefix hit, end=%d", ends[-1])
        return ends[-1]

    def supports_producer_reuse(self) -> bool:
        """Return True as MooncakeLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        # nothing here
        pass
