# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
import json
import os
import time

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm.version import __version__ as VLLM_VERSION
import torch

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache import utils
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    lmcache_get_or_create_config,
)
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheStoreEvent, _lmcache_nvtx_annotate, cdiv
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import validate_and_set_config_value
from lmcache.v1.manager import LMCacheManager

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    # First Party
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    receiver_rdma_host: Optional[str] = None  # Mooncake RDMA hostname
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0


tmp_disagg_tracker: dict[str, DisaggSpec] = {}
# Module-level staging for proxy-injected correlation_id on decode side.
# update_state_after_alloc runs in the scheduler process; the consumer
# JSONL is emitted in the worker process (different Python process, so
# a per-instance dict attribute would not survive the hop). We key by
# vLLM's request.req_id and drain it into RequestTracker.correlation_id
# during from_new_request_data, which IS carried into ReqMeta and
# shipped to workers via build_connector_meta.
tmp_correlation_id_tracker: dict[str, str] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params and sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    allocated_block_ids: list[int]

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    # The number of tokens that are cached in LMCache for this request
    num_lmcache_cached_tokens: int = 0

    # Proxy-injected join key (= prefill's completion id) for decode's
    # consumer JSONL — lets the benchmark prefix-match producer JSONL
    # (which already starts with prefill's id) against consumer JSONL.
    correlation_id: Optional[str] = None

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # tuple[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)
        correlation_id = tmp_correlation_id_tracker.pop(
            new_request.req_id, None
        )

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
            correlation_id=correlation_id,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
        preempted: bool = False,
        lmcache_cached_tokens: int = 0,
        vllm_cached_tokens: int = 0,
        all_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again

        vllm_cached_tokens: the number of tokens that are cached in vLLM
        is only used for preempted requests
        all_token_ids: the full token list from the vLLM request, used to
        restore token_ids for preempted requests to ensure chunk keys match
        """

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            # If input is a list, flatten it to handle potential nesting.
            # This also correctly processes already-flat lists.
            new_block_ids = [
                i
                for elem in new_block_ids
                for i in (elem if isinstance(elem, list) else [elem])
            ]
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            # the block ids will change after preemption
            self.allocated_block_ids = new_block_ids
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            num_computed_tokens = max(lmcache_cached_tokens, vllm_cached_tokens)

            # FIX: For preempted requests, restore token_ids from the full
            # token list to ensure chunk keys match what was used during
            # lookup. The lookup uses request.all_token_ids, so we need the
            # same tokens for retrieve.
            num_tokens_needed = max(
                num_computed_tokens + len(new_token_ids),
                lmcache_cached_tokens,
            )
            self.token_ids = all_token_ids[:num_tokens_needed]
        else:
            self.allocated_block_ids.extend(new_block_ids)
            self.token_ids.extend(new_token_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None
    # Proxy-injected join key (= prefill's completion id) for decode's
    # consumer JSONL. Passed through from RequestTracker so the worker
    # process can use it when writing consumer metrics.
    correlation_id: Optional[str] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len >= tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = input_token_ids[:num_tokens_to_save]

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks"
                " for request %s. "
                "Something might be wrong in scheduling logic!",
                tracker.req_id,
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: log if the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens (%d cached in vLLM) for request %s",
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
                tracker.req_id,
            )

        # Note: We keep load_spec even when can_load=False to pass metrics to worker
        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
            correlation_id=tracker.correlation_id,
        )


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size

        # Load and configure LMCache config
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)
        self.config = config

        service_factory = VllmServiceFactory(config, vllm_config, role.name.lower())
        self._manager = LMCacheManager(config, service_factory, connector=self)

        # Start services managed by LMCacheManager
        self._manager.start_services()

        # Initialize connector-specific state
        self._init_connector_state(role, vllm_config, config)

        # Setup metrics for monitoring data structures
        self._setup_metrics()

        logger.info(
            "LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            utils.get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    def _apply_extra_config(
        self, config: LMCacheEngineConfig, vllm_config: "VllmConfig"
    ) -> None:
        """Apply extra config from vLLM to LMCache config."""
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            "Updated config %s from vLLM extra config",
                            config_key,
                        )

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize connector-specific state variables."""
        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []
        self._layerwise_save_storers: dict[
            str, Generator[Optional[torch.Tensor], None, None]
        ] = {}
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()

        # Role-specific initialization
        if role == KVConnectorRole.SCHEDULER:
            self._unfinished_requests: dict[str, "Request"] = {}
        else:
            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending

            if self.enable_blending:
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                )

        # Legacy compatibility check
        self._check_legacy_register_kv_caches()

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._block_size = vllm_config.cache_config.block_size
        self.load_specs: dict[str, LoadSpec] = {}
        self.kv_cache_manager: Optional["KVCacheManager"] = None
        self._request_trackers: dict[str, RequestTracker] = {}

        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))
        self._requests_priority: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()

        # ── Metric state per tasks/对齐Metrics.md ──────────────────────
        #
        # Each request emits one row on the producer (prefill) and one row
        # on the consumer (decode); the benchmark joins them by req_id.
        # Per-request timestamps captured in start_load_kv /
        # save_kv_layer are staged in these dicts and drained in
        # wait_for_save when the JSONL line is written.
        #
        # Producer side (prefill):
        #   _t_sl_kv_start[req_id]  = t_prefill_start_load_kv_start
        #       (captured in start_load_kv entry, AFTER cuda.synchronize())
        #   _t_sl_kv_end[req_id]    = t_prefill_start_load_kv_end
        #       (captured in start_load_kv exit, NO sync — retrieve's
        #       internal load_stream.synchronize() already fenced H2D)
        #   _e_prefill_start[req_id] = cuda.Event recorded on default
        #       stream at the END of start_load_kv. Its GPU-side
        #       completion = "GPU forward about to begin". Drained in
        #       wait_for_save to compute GPU-anchored t_prefill_start and
        #       d_prefill = elapsed_time(e_prefill_start, e_prefill_end).
        #   _t_kv_pipeline_start[req_id] = t_kv_start for overlap mode
        #       (first save_kv_layer call, with a preceding
        #       cuda.synchronize() per spec). Non-overlap ignores this
        #       and uses t_prefill_wait_for_save_start as t_kv_start.
        #
        # Consumer side (decode):
        #   _t_kv_mnck_out_start[req_id] = t_kv_mnck_out_start
        #       (captured in start_load_kv entry on the consumer, AFTER
        #       cuda.synchronize()).
        self._t_sl_kv_start: dict[str, float] = {}
        self._t_sl_kv_end:   dict[str, float] = {}
        self._e_prefill_start: dict[str, torch.cuda.Event] = {}
        self._t_kv_pipeline_start: dict[str, float] = {}
        self._t_kv_mnck_out_start: dict[str, float] = {}

        # ── Anchor for GPU→wall-clock translation ─────────────────────
        # Lazily initialized on first start_load_kv call. Anchor = a pair
        # (cpu_wall_clock, cuda.Event) taken when the GPU is known idle
        # (preceded by a one-time cuda.synchronize). Subsequent cuda
        # events E translate to wall-clock via
        #     wall_clock(E) = anchor_cpu + anchor_event.elapsed_time(E) / 1000.
        # Narrow event.synchronize() at JSONL-write time (event is
        # typically long-complete by then, so sync returns instantly).
        self._anchor_cpu: Optional[float] = None
        self._anchor_event: Optional[torch.cuda.Event] = None

    def _check_legacy_register_kv_caches(self) -> None:
        """Check for legacy connector without register_kv_caches implementation."""
        if self.lmcache_engine is None:
            return

        child_class = self._parent.__class__
        parent_class = KVConnectorBase_V1
        child_method = getattr(child_class, "register_kv_caches", None)
        parent_method = getattr(parent_class, "register_kv_caches", None)

        if child_method is None or parent_method is None:
            implements = False
        else:
            implements = child_method is not parent_method

        if not implements:
            logger.warning(
                "Please use the latest lmcache connector, otherwise some "
                "features may not work, such as DSA"
            )
            self._manager.post_init()

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance from manager."""
        return self._manager.lmcache_engine

    @property
    def lmcache_engine_metadata(self):
        """Get the LMCache engine metadata from manager."""
        return self._manager.lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional["LookupClientInterface"]:
        """Get the lookup client from manager."""
        return self._manager.lookup_client

    @property
    def lookup_server(self):
        """Get the lookup server from manager."""
        return self._manager.lookup_server

    def _setup_metrics(self):
        """Setup metrics for monitoring data structures in the connector."""
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is None:
            logger.warning(
                "PrometheusLogger is not initialized, "
                "connector metrics will not be collected"
            )
            return

        # Set up metrics for scheduler-specific and general data structures
        metrics_map = {
            "_unfinished_requests": "scheduler_unfinished_requests_count",
            "load_specs": "connector_load_specs_count",
            "_request_trackers": "connector_request_trackers_count",
            "kv_caches": "connector_kv_caches_count",
            "layerwise_retrievers": "connector_layerwise_retrievers_count",
            "_invalid_block_ids": "connector_invalid_block_ids_count",
            "_requests_priority": "connector_requests_priority_count",
        }

        for attr_name, metric_name in metrics_map.items():
            if hasattr(self, attr_name):
                metric = getattr(prometheus_logger, metric_name)
                # Use a default argument in the lambda to capture
                # the current value of `attr_name`
                # to avoid issues with late binding in closures.
                metric.set_function(lambda name=attr_name: len(getattr(self, name)))

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    def _build_kv_layer_groups(self):
        # Build KV layer groups structure if not already built
        if self.lmcache_engine is not None:
            assert len(self.kv_caches) > 0
            kv_layer_groups_manager = (
                self.lmcache_engine.metadata.kv_layer_groups_manager
            )
            kv_layer_groups_manager.build_kv_layer_groups(self.kv_caches)

    # TODO(chunxiaozheng): in the latest lmcache_connector, we use `register_kv_caches`
    #  to init self.kv_caches, we keep it in order to be compatible with old versions
    #  and will be removed in the future.
    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

        self._build_kv_layer_groups()

    ####################
    # Worker side APIs
    ####################

    # Written at `/home/zeyu/...` so the file lives on the host-visible
    # XFS mount and benchmark.py can collect it via plain `ssh cat`
    # without docker exec. Producer and consumer use separate files
    # since on any given node one worker process only plays one role
    # (prefill or decode), but the benchmark collects from both.
    _METRICS_FILE_PRODUCER = "/home/zeyu/lmcache_metrics_producer.jsonl"
    _METRICS_FILE_CONSUMER = "/home/zeyu/lmcache_metrics_consumer.jsonl"

    @staticmethod
    def _normalize_jsonl_req_id(req_id: str) -> str:
        """Collapse vLLM's four-part internal ``cmpl-<hex>-<rank>-<uuid>``
        down to the two-part completion id ``cmpl-<hex>`` that the
        client sees in its first streaming chunk. Producer and consumer
        sides both normalize to this same form, letting the benchmark's
        server-side metric join match both sides with a single key
        instead of multiple prefix-match hits.
        """
        if req_id is None or not req_id.startswith("cmpl-"):
            return req_id
        parts = req_id.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])
        return req_id

    def _ensure_timing_anchor(self) -> None:
        """Lazily build the (cpu_wall_clock, cuda.Event) anchor on first use.

        A one-time ``torch.cuda.synchronize()`` ensures the GPU is idle at
        the anchor point so ``self._anchor_cpu`` and the GPU completion of
        ``self._anchor_event`` refer to the same real-world instant.
        Subsequent requests can translate any cuda event's GPU-time to
        wall-clock via :meth:`_event_wall_clock` without ever syncing
        device-wide again.

        Called at the top of every ``start_load_kv``; initialization runs
        on the **first** invocation only (typically the first warm-up
        request, when GPU is already idle — the sync cost is tiny).
        """
        if self._anchor_event is not None:
            return
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        self._anchor_cpu = time.time()
        self._anchor_event = torch.cuda.Event(enable_timing=True)
        self._anchor_event.record()

    def _event_wall_clock(self, event: Optional[torch.cuda.Event]) -> Optional[float]:
        """Return the wall-clock epoch (seconds, matching ``time.time()``)
        at which the given cuda event completed on the GPU.

        Uses ``event.synchronize()`` (narrow, single-event) to ensure the
        event has completed; by the time we call this (at JSONL-write
        point, after ``wait_put_done`` returned / after a narrow load_stream
        sync) the event is virtually always already complete, so the
        synchronize call returns in nanoseconds.

        Returns ``None`` when the anchor hasn't been initialized or the
        event is ``None`` (fallback paths).
        """
        if event is None or self._anchor_event is None or self._anchor_cpu is None:
            return None
        event.synchronize()
        gpu_elapsed_ms = self._anchor_event.elapsed_time(event)
        return self._anchor_cpu + gpu_elapsed_ms / 1000.0

    def _get_remote_backend(self):
        """Return the RemoteBackend instance, or None if not present.

        Identified by the presence of a ``wait_put_done`` method so the
        adapter doesn't hard-code the backend class name.
        """
        sm = getattr(self.lmcache_engine, "storage_manager", None)
        if sm is None:
            return None
        for backend in sm.storage_backends.values():
            if hasattr(backend, "wait_put_done"):
                return backend
        return None

    def _write_producer_metric(
        self, req_id: str, mode: str,
        t_prefill_start_load_kv_start: float,
        t_prefill_start_load_kv_end: float,
        t_prefill_wait_for_save_start: float,
        t_kv_start: float,
        t_kv_mnck_in_end: float,
        t_prefill_start: Optional[float] = None,
        t_prefill_end: Optional[float] = None,
        d_prefill_ms: Optional[float] = None,
    ) -> None:
        """Append one prefill-side request record to producer JSONL.

        Schema per ``tasks/对齐Metrics.md``:

        CPU wall-clock timestamps (``time.time()`` epoch seconds):
        - ``t_prefill_start_load_kv_start``: start_load_kv entry, AFTER
          ``cuda.synchronize()`` (so prior GPU work is drained and the
          CPU boundary is truthful).
        - ``t_prefill_start_load_kv_end``: start_load_kv exit, no
          sync (retrieve's internal load_stream.synchronize already
          fenced any H2D memcpy).
        - ``t_prefill_wait_for_save_start``: wait_for_save entry, AFTER
          ``cuda.synchronize()`` (so forward kernels are all drained).
        - ``t_kv_start``: "KV transmission begins"; for non-overlap it
          equals ``t_prefill_wait_for_save_start``. For layerwise overlap
          it is the first save_kv_layer's stamp (also after a sync).
        - ``t_kv_mnck_in_end``: after ``wait_put_done`` returned for
          every Put of this request — RDMA CQE observed = all KV
          physically in the target Mooncake segment.

        GPU-anchored wall-clocks (from ``cuda.Event.elapsed_time`` + anchor):
        - ``t_prefill_start``: when the ``e_prefill_start`` event fired
          on the default stream (= GPU forward about to begin).
        - ``t_prefill_end``: when the ``e_prefill_end`` event fired
          (= last forward kernel completed on GPU).

        Authoritative prefill duration:
        - ``d_prefill_ms``: ``e_prefill_start.elapsed_time(e_prefill_end)``
          directly — the cudaEventElapsedTime value, which is the source
          of truth for GPU forward duration.
        """
        try:
            record = {
                "req_id": req_id,
                "mode": mode,
                "t_prefill_start_load_kv_start": t_prefill_start_load_kv_start,
                "t_prefill_start_load_kv_end":   t_prefill_start_load_kv_end,
                "t_prefill_wait_for_save_start": t_prefill_wait_for_save_start,
                "t_kv_start":       t_kv_start,
                "t_kv_mnck_in_end": t_kv_mnck_in_end,
                # GPU-anchored wall-clocks (None if CUDA unavailable);
                # fall back to the nearest CPU boundary so the join
                # contract stays intact.
                "t_prefill_start": (t_prefill_start
                                    if t_prefill_start is not None
                                    else t_prefill_start_load_kv_end),
                "t_prefill_end":   (t_prefill_end
                                    if t_prefill_end is not None
                                    else t_prefill_wait_for_save_start),
                "ts": time.time(),
            }
            # d_prefill_ms from the direct event-pair elapsed_time is
            # the authoritative value — benchmark.py reads it as-is
            # instead of recomputing from t_prefill_end - t_prefill_start
            # (those go through the anchor and lose precision).
            if d_prefill_ms is not None:
                record["d_prefill_ms"] = d_prefill_ms
            line = json.dumps(record)
            with open(self._METRICS_FILE_PRODUCER, "a") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.debug("Failed to write producer metrics JSONL: %s", e)

    def _write_consumer_metric(
        self, req_id: str, mode: str,
        t_kv_mnck_out_start: float,
        t_kv_end: float,
    ) -> None:
        """Append one decode-side request record to consumer JSONL.

        Per ``tasks/对齐Metrics.md``:
        - ``t_kv_mnck_out_start``: captured at consumer start_load_kv
          entry, AFTER ``cuda.synchronize()``.
        - ``t_kv_end``: captured at consumer start_load_kv exit, no
          sync (retrieve's internal load_stream.synchronize already
          fenced H2D memcpy). In the layerwise branch we do call
          ``load_stream.synchronize()`` before t_kv_end because
          retrieve_layer (the generator) does not itself sync; that's
          handled in wait_for_layer_load, not here.
        """
        try:
            line = json.dumps({
                "req_id": req_id,
                "mode": mode,
                "t_kv_mnck_out_start": t_kv_mnck_out_start,
                "t_kv_end": t_kv_end,
                "ts": time.time(),
            })
            with open(self._METRICS_FILE_CONSUMER, "a") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.debug("Failed to write consumer metrics JSONL: %s", e)

    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._build_kv_layer_groups()
        self._manager.post_init()

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        Metric capture (per ``tasks/对齐Metrics.md``):

        Prefill (producer):
          - ENTRY: ``cuda.synchronize()`` then record
            ``t_prefill_start_load_kv_start`` (drains any GPU work that
            would otherwise smear across this boundary).
          - LOOP: retrieve cached KV (producer-with-prefix-cache path).
          - EXIT: record ``e_prefill_start`` on default stream (its
            GPU-side completion = "forward about to begin"); then
            record ``t_prefill_start_load_kv_end`` (no sync — retrieve's
            internal ``load_stream.synchronize()`` has already fenced
            any enqueued H2D memcpy).

        Decode (consumer):
          - ENTRY: ``cuda.synchronize()`` then record
            ``t_kv_mnck_out_start`` (per spec, decode side needs sync
            before stamping the "KV pull start" moment).
          - LOOP: retrieve KV from Mooncake (H2D memcpy on load_stream,
            internally synced by retrieve() via
            batched_to_gpu → load_stream.synchronize()).
          - EXIT (non-layerwise): record ``t_kv_end = time.time()`` with
            NO sync (retrieve already fenced H2D); emit consumer JSONL.

        ``setdefault`` is used on all per-request dicts so chunked prefill
        (multiple start_load_kv calls for the same request) keeps the
        FIRST chunk's boundary.
        """
        is_prefill_producer = self.kv_role == "kv_producer"
        is_kv_consumer = self.kv_role == "kv_consumer"

        # ── Entry-time cuda.synchronize() + CPU stamp ─────────────────
        # Spec: "t_prefill_start_load_kv_start: ... 需要在记录时先cuda sync"
        #       "t_kv_mnck_out_start:          ... 需要在记录时先cuda sync"
        # Both prefill and decode sides sync before taking the stamp.
        if (is_prefill_producer or is_kv_consumer) and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_entry = time.time()
        self._ensure_timing_anchor()

        self.current_layer = 0

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        # Stash the CPU entry stamp per-request (setdefault preserves the
        # FIRST chunk's value across chunked prefill).
        if is_prefill_producer:
            for request in metadata.requests:
                self._t_sl_kv_start.setdefault(request.req_id, t_entry)
        if is_kv_consumer:
            for request in metadata.requests:
                if (
                    request.load_spec is not None
                    and request.load_spec.can_load
                ):
                    self._t_kv_mnck_out_start.setdefault(
                        request.req_id, t_entry
                    )

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None

        self.layerwise_retrievers = []
        # Parallel list: (req_id, correlation_id) for each retriever
        # appended below. wait_for_layer_load uses this on the last
        # layer to emit the consumer-side JSONL metric line, keying by
        # correlation_id when available so producer+consumer records
        # join on a single key (= client's first-chunk id).
        self._layerwise_retriever_req_ids: list[tuple[str, Optional[str]]] = []

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            last_idx = idx

        # Collect per-request retrieve outcomes for the non-layerwise
        # consumer JSONL emit at end-of-function.
        non_layerwise_retrieved_reqs: list[tuple["ReqMeta", torch.Tensor, int]] = []

        for idx, request in enumerate(metadata.requests):
            # Update metrics for all requests that have a load_spec
            if request.load_spec is not None:
                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(
                    len(request.token_ids)
                )

            if request.load_spec is None or not request.load_spec.can_load:
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.to(self.device)
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
                    self._layerwise_retriever_req_ids.append(
                        (request.req_id, request.correlation_id)
                    )
            else:
                # retrieve() internally calls
                #   gpu_connector.batched_to_gpu(...)
                #     → load_stream.synchronize()
                # so when this call returns, the H2D memcpy is already
                # complete on the GPU side. No explicit sync needed in
                # the adapter.
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )
                non_layerwise_retrieved_reqs.append(
                    (request, ret_token_mask, lmcache_cached_tokens)
                )

        # ── Consumer (decode) non-layerwise JSONL emit ────────────────
        # retrieve() already synced load_stream, so t_kv_end = time.time()
        # directly (per spec: "使用了retrieve函数的no overlap模式，不需要
        # cuda sync").
        if is_kv_consumer and non_layerwise_retrieved_reqs:
            t_kv_end = time.time()
            for (req, ret_token_mask, lmcache_cached_tokens) in \
                    non_layerwise_retrieved_reqs:
                t_mo_start = self._t_kv_mnck_out_start.pop(req.req_id, t_kv_end)
                jsonl_id = (
                    req.correlation_id
                    or self._normalize_jsonl_req_id(req.req_id)
                )
                self._write_consumer_metric(
                    jsonl_id, "non_layerwise",
                    t_mo_start, t_kv_end,
                )
                # Sanity check on retrieved-token count (preserves the
                # pre-existing failure-block recovery path).
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - req.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        req.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    token_mask = torch.ones(len(req.token_ids), dtype=torch.bool)
                    masked_token_count = (
                        req.load_spec.vllm_cached_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )
                    token_mask[:masked_token_count] = False
                    slot_mapping = req.slot_mapping.to(self.device)
                    missing_blocks = self.record_failed_blocks(
                        req.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                    )
                    self._invalid_block_ids.update(missing_blocks)

        # ── Producer: record e_prefill_start + t_sl_kv_end ────────────
        # Spec: "e_prefill_start: prefill开始的cuda事件，可以放在prefill的
        #        start_load_kv函数最后"
        # Single event per iteration (shared across all reqs in this
        # forward batch), bound to each req via setdefault so chunked
        # prefill preserves the first chunk's event = "first forward
        # about to begin".
        if is_prefill_producer and torch.cuda.is_available():
            e_start = torch.cuda.Event(enable_timing=True)
            e_start.record()
            for request in metadata.requests:
                self._e_prefill_start.setdefault(request.req_id, e_start)

        if is_prefill_producer:
            # Spec: "t_prefill_start_load_kv_end: ... 不需要cuda sync，
            #        因为retrieve函数已经同步了to GPU的时间".
            t_sl_kv_end = time.time()
            for request in metadata.requests:
                self._t_sl_kv_end.setdefault(request.req_id, t_sl_kv_end)

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        if not torch.any(missing_mask):
            return set()

        missing_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        if slot_mapping_cpu.shape[0] > missing_mask.shape[0]:
            slot_mapping_cpu = slot_mapping_cpu[: missing_mask.shape[0]]

        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // self._block_size
        )
        missing_blocks = {int(block.item()) for block in missing_blocks_tensor}

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.layerwise_retrievers:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        is_last_layer = (
            self.layerwise_retrievers
            and self.current_layer == self.num_layers - 1
        )

        # Wait for the layer to be loaded
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")

        if is_last_layer and self.kv_role == "kv_consumer":
            # The retrieve_layer generator's final iteration calls
            # `current_stream.wait_stream(load_stream)` inside
            # gpu_connector.batched_to_gpu (at cache_engine.py:1042
            # `next(mem_obj_consumer)`). That's a stream-level
            # ordering fence — the decode forward kernel queued next
            # on current_stream will observe the finished H2D copy —
            # but it doesn't block the host thread. For t_kv_end to
            # reflect host time at which the KV is actually resident
            # in the GPU paged buffer we want a host-blocking sync
            # here. Use load_stream.synchronize() rather than a
            # device-wide cuda.synchronize() because overlap mode
            # may have other streams doing unrelated work (prefill-
            # stream, compute-stream, NCCL-stream), and
            # cuda.synchronize() on the device sometimes interacts
            # poorly with enforcing kernels / CUDA graphs in the
            # forward path and stalls indefinitely.
            try:
                if self.gpu_connector is not None and hasattr(
                    self.gpu_connector, "load_stream"
                ):
                    self.gpu_connector.load_stream.synchronize()
            except Exception as e:
                logger.debug("load_stream.synchronize() failed: %s", e)
            t_kv_end = time.time()
            for req_id, correlation_id in self._layerwise_retriever_req_ids:
                t_mo_start = self._t_kv_mnck_out_start.pop(
                    req_id, t_kv_end
                )
                # Join key = proxy's correlation_id (= prefill's
                # completion id) if present, else vLLM's req_id
                # normalized to match producer's key format.
                jsonl_id = (
                    correlation_id
                    or self._normalize_jsonl_req_id(req_id)
                )
                self._write_consumer_metric(
                    jsonl_id, "layerwise",
                    t_mo_start, t_kv_end,
                )
            self._layerwise_retriever_req_ids = []

        if self.layerwise_retrievers:
            self.current_layer += 1

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        kvcaches = list(self.kv_caches.values())
        is_first = True

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            layerwise_storer = self._layerwise_save_storers.get(request.req_id)
            if layerwise_storer is None:
                token_ids = request.token_ids
                assert isinstance(token_ids, list)

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.to(self.device)

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    assert save_spec is not None
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.debug(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                    req_id=request.req_id,
                    transfer_spec=request.disagg_spec,
                )
                self._layerwise_save_storers[request.req_id] = layerwise_storer
                if is_first:
                    is_first = False

            # Record t_kv_start for overlap mode: "第一层KV开始发送的时间,
            # 需要在记录时先cuda sync进行一次同步" (per
            # tasks/对齐Metrics.md). We only stamp this once per request
            # (setdefault) so the sync only fires on the first
            # save_kv_layer call for that request — aligning the CPU
            # clock with the GPU "first KV layer about to ship" moment.
            # Subsequent layers stamp nothing.
            if request.req_id not in self._t_kv_pipeline_start:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._t_kv_pipeline_start[request.req_id] = time.time()

            next(layerwise_storer)

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer.

        Metric capture (per ``tasks/对齐Metrics.md``):

        Prefill (producer):
          - ENTRY: record ``e_prefill_end`` on default stream (queues
            behind any pending forward kernels; its GPU-side completion
            = "forward last kernel done"). Then ``cuda.synchronize()``
            to drain all GPU work. Then record
            ``t_prefill_wait_for_save_start = time.time()`` — the
            authoritative "forward done on host" boundary.
          - NON-OVERLAP: ``t_kv_start = t_prefill_wait_for_save_start``
            (spec: "对于no overlap模式而言，直接等于
            t_prefill_wait_for_save_start").
          - LOOP: submit ``lmcache_engine.store()`` calls (async; they
            enqueue D2H on store_stream and submit RDMA Put through
            asyncio).
          - PHASE 2: for each submitted req, ``wait_put_done()`` until
            the Put's RDMA CQE is observed → ``t_kv_mnck_in_end``.
          - COMPUTE GPU-anchored wall-clocks for ``t_prefill_start`` /
            ``t_prefill_end`` via the anchor + ``cuda.Event`` machinery,
            and authoritative ``d_prefill_ms`` via direct
            ``e_prefill_start.elapsed_time(e_prefill_end)``.

        Decode (consumer): save is a no-op; just unpin the KV pins
        balancing ``contains()``.
        """
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            # But still need to unpin the kv caches according to req_id
            # to balance the pin count from contains()
            assert self.lmcache_engine is not None, (
                "LMCacheEngine must be initialized to unpin requests."
            )
            for request in connector_metadata.requests:
                self.lmcache_engine.lookup_unpin(request.req_id)

            return

        # ── ENTRY: e_prefill_end marker + sync + CPU boundary stamp ───
        # Spec: e_prefill_end is recorded at the very beginning of
        # wait_for_save on prefill. Record on default stream first; it
        # queues BEHIND any pending forward kernels and fires when
        # default stream drains (= "forward done on GPU"). Then
        # cuda.synchronize() drains default stream on the host thread so
        # t_prefill_wait_for_save_start truly marks "forward done on CPU".
        e_prefill_end: Optional[torch.cuda.Event] = None
        if torch.cuda.is_available():
            e_prefill_end = torch.cuda.Event(enable_timing=True)
            e_prefill_end.record()
            torch.cuda.synchronize()
        t_prefill_wait_for_save_start = time.time()

        if self.use_layerwise:
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    next(layerwise_storer)
                # unpin the kv caches according to req_id
                self.lmcache_engine.lookup_unpin(request.req_id)

            # After all generators flushed, RDMA Puts have been SUBMITTED
            # but not necessarily completed. Block per-request on the
            # remote backend's Put-completion tracker to recover the
            # real wall-clock "last-CQE-done" for t_kv_mnck_in_end.
            remote_backend = self._get_remote_backend()
            for request in connector_metadata.requests:
                t_sl_kv_start = self._t_sl_kv_start.pop(
                    request.req_id, t_prefill_wait_for_save_start
                )
                t_sl_kv_end = self._t_sl_kv_end.pop(
                    request.req_id, t_prefill_wait_for_save_start
                )
                # Overlap mode: t_kv_start = first save_kv_layer stamp
                # (captured with a preceding cuda.sync per spec).
                # Fallback: wait_for_save entry boundary.
                t_kv_start = self._t_kv_pipeline_start.pop(
                    request.req_id, t_prefill_wait_for_save_start
                )
                t_kv_mnck_in_end: Optional[float] = None
                if remote_backend is not None:
                    _, t_kv_mnck_in_end = remote_backend.wait_put_done(
                        request.req_id
                    )
                if t_kv_mnck_in_end is None:
                    t_kv_mnck_in_end = time.time()
                # GPU-anchored prefill span
                e_start = self._e_prefill_start.pop(request.req_id, None)
                t_prefill_start_gpu = self._event_wall_clock(e_start)
                t_prefill_end_gpu   = self._event_wall_clock(e_prefill_end)
                d_prefill_ms: Optional[float] = None
                if e_start is not None and e_prefill_end is not None:
                    try:
                        e_start.synchronize()
                        e_prefill_end.synchronize()
                        d_prefill_ms = e_start.elapsed_time(e_prefill_end)
                    except Exception:
                        d_prefill_ms = None
                self._write_producer_metric(
                    self._normalize_jsonl_req_id(request.req_id),
                    "layerwise",
                    t_prefill_start_load_kv_start = t_sl_kv_start,
                    t_prefill_start_load_kv_end   = t_sl_kv_end,
                    t_prefill_wait_for_save_start = t_prefill_wait_for_save_start,
                    t_kv_start        = t_kv_start,
                    t_kv_mnck_in_end  = t_kv_mnck_in_end,
                    t_prefill_start   = t_prefill_start_gpu,
                    t_prefill_end     = t_prefill_end_gpu,
                    d_prefill_ms      = d_prefill_ms,
                )
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        # Phase 1: submit all stores. Don't block per-request — `store()`
        # already submits async via remote_backend, so serializing the
        # wait inside this loop would stall subsequent requests behind
        # each one's RDMA. Collect req_ids that actually got Puts
        # submitted; Phase 2 then awaits them all (in parallel in the
        # asyncio loop) and writes metrics.
        submitted_req_ids: list[str] = []
        for request in connector_metadata.requests:
            # unpin the kv caches according to req_id
            self.lmcache_engine.lookup_unpin(request.req_id)

            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.to(self.device)

            skip_leading_tokens = save_spec.skip_leading_tokens
            # shared storage disaggregation will not have a disagg_spec passed in
            if self.kv_role == "kv_producer" and request.disagg_spec:
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.debug(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            # Non-overlap path: strip `receiver_rdma_host` before the
            # engine forwards the spec to the Mooncake remote backend.
            #
            # Rationale: `receiver_rdma_host` was added for the KV-overlap
            # experiment (commit 30e050e) so prefill's per-Put RDMA WRITE
            # targets decode's Mooncake segment directly, allowing the Put
            # to overlap with the remaining forward-pass compute. In
            # no-overlap we call this `store()` only once, AFTER prefill
            # forward finishes — there is no compute left to overlap with,
            # so forcing preferred_segment=decode_IP just shifts wall-clock
            # time from d_kv_mnck_out (decode Get) to d_kv_mnck_in (prefill
            # Put) without any end-to-end benefit, and inflates the Put-side
            # measurement in a way that obscures the per-NIC bandwidth
            # comparison. Upstream LMCache `dev` leaves preferred_segment at
            # its init-time value (= local hostname if
            # `mooncake_prefer_local_alloc=true`, else master's free-space
            # pick); here we match that by nulling out the per-Put override.
            no_overlap_spec = (
                replace(request.disagg_spec, receiver_rdma_host=None)
                if request.disagg_spec is not None
                else None
            )
            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=no_overlap_spec,
                request_configs=request.request_configs,
                req_id=request.req_id,
            )
            submitted_req_ids.append(request.req_id)

            # Update skip_leading_tokens only on last rank to ensure
            # each PP stage stores its own KV cache
            if get_pp_group().is_last_rank:
                # NOTE(Jiayi): We assume all tokens are saved
                save_spec.skip_leading_tokens = len(token_ids)
                if request.disagg_spec:
                    request.disagg_spec.num_transferred_tokens = len(token_ids)

        # Phase 2: for every request whose Puts were submitted, wait for
        # real RDMA completion and emit the producer JSONL line.
        #
        # Non-overlap: by spec, t_kv_start = t_prefill_wait_for_save_start
        # (no per-layer streaming; KV pipeline starts exactly when
        # wait_for_save begins, which also coincides with "GPU forward
        # finished on host" thanks to the entry-time cuda.synchronize).
        remote_backend = self._get_remote_backend()
        # GPU-anchored t_prefill_end: same event for every req in this
        # iteration — compute its wall-clock once, reuse.
        t_prefill_end_gpu = self._event_wall_clock(e_prefill_end)
        for req_id in submitted_req_ids:
            t_sl_kv_start = self._t_sl_kv_start.pop(
                req_id, t_prefill_wait_for_save_start
            )
            t_sl_kv_end = self._t_sl_kv_end.pop(
                req_id, t_prefill_wait_for_save_start
            )
            e_start = self._e_prefill_start.pop(req_id, None)
            t_prefill_start_gpu = self._event_wall_clock(e_start)
            # Authoritative d_prefill: direct event-pair elapsed_time in
            # milliseconds. Because we sync'd at wait_for_save entry,
            # e_prefill_end has long since completed — but we still
            # narrow-sync it defensively.
            d_prefill_ms: Optional[float] = None
            if e_start is not None and e_prefill_end is not None:
                try:
                    e_start.synchronize()
                    e_prefill_end.synchronize()
                    d_prefill_ms = e_start.elapsed_time(e_prefill_end)
                except Exception:
                    d_prefill_ms = None
            t_kv_mnck_in_end: Optional[float] = None
            if remote_backend is not None:
                _, t_kv_mnck_in_end = remote_backend.wait_put_done(req_id)
            if t_kv_mnck_in_end is None:
                t_kv_mnck_in_end = time.time()
            self._write_producer_metric(
                self._normalize_jsonl_req_id(req_id),
                "non_layerwise",
                t_prefill_start_load_kv_start = t_sl_kv_start,
                t_prefill_start_load_kv_end   = t_sl_kv_end,
                t_prefill_wait_for_save_start = t_prefill_wait_for_save_start,
                t_kv_start        = t_prefill_wait_for_save_start,
                t_kv_mnck_in_end  = t_kv_mnck_in_end,
                t_prefill_start   = t_prefill_start_gpu,
                t_prefill_end     = t_prefill_end_gpu,
                d_prefill_ms      = d_prefill_ms,
            )

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    @_lmcache_nvtx_annotate
    def shutdown(self):
        """Shutdown the connector by delegating to LMCacheManager."""
        logger.info("Starting LMCacheConnector shutdown...")
        self._manager.stop_services()

    ###################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # Ignore DP attention mock requests
        if request.request_id.startswith("mock_req"):
            return 0
        # to handle preempted requests, we want `get_num_new_matched_tokens` to be
        # idempotent under the condition that `update_state_after_alloc` is NOT called
        # then the two side-effects that must be idempotent are:
        # 1. lookup_client caches a result
        #     uncached in `update_state_after_alloc` if this request can be scheduled
        # 2. cache engine will pin the KV caches for the request
        #     unpinned in `wait_for_save` if this request can be scheduled
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        req_id = request.request_id

        # lookup_client is always initialized for scheduler role
        assert self.lookup_client is not None

        if (
            num_external_hit_tokens := self.lookup_client.lookup_cache(lookup_id=req_id)
        ) != -1:
            # -1 means no result cached
            # None or int means ongoing (async) or cached result
            logger.debug(
                f"Found {num_external_hit_tokens} hit tokens for request"
                f" {req_id} in the lookup cache."
            )
        else:
            logger.debug(f"Looking up cache for the first time for request {req_id}!")
            self._requests_priority[req_id] = getattr(request, "priority", 0)

            # token_ids = request.prompt_token_ids
            # all token ids covers the preemption case
            token_ids = request.all_token_ids

            # If the request has multimodal hashes, apply them to the token ids
            mm_hashes, mm_positions = extract_mm_features(request)
            if mm_hashes and mm_positions:
                # TODO(Jiayi): Optimize this
                token_ids = torch.tensor(request.prompt_token_ids)
                apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
                token_ids = token_ids.tolist()

            request_configs = extract_request_configs(request.sampling_params)
            if self.skip_last_n_tokens > 0:
                token_ids = token_ids[: -self.skip_last_n_tokens]

            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids,
                lookup_id=req_id,
                request_configs=request_configs,
            )

        if num_external_hit_tokens is None:
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: None.",
                req_id,
                request.num_tokens,
                num_computed_tokens,
            )
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        # Check if hit tokens meet the minimum for retrieve
        # If below minimum, skip retrieve but still record hit tokens
        # for skip_leading_tokens to avoid re-storing existing chunks
        min_retrieve = self.config.min_retrieve_tokens
        below_min_retrieve = min_retrieve > 0 and need_to_allocate < min_retrieve

        if below_min_retrieve:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, but need to load: %d < min_retrieve %d, "
                "skip retrieve but record for save skip",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
                min_retrieve,
            )
        else:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, need to load: %d",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
            )

        self.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        if below_min_retrieve or need_to_allocate <= 0:
            return 0

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        assert self.lookup_client is not None
        self.lookup_client.clear_lookup_status(request.request_id)

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
                receiver_rdma_host=req_disagg_spec.get(
                    "receiver_rdma_host"
                ),
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec

        # Decode-side: capture the proxy's correlation_id (= prefill's
        # completion id, i.e., the first-chunk id the client sees) into
        # a module-level staging dict, keyed by vLLM's req_id. The
        # worker process (different from the scheduler process running
        # this hook) will pick it up when RequestTracker is built via
        # from_new_request_data, then pass it through ReqMeta → worker.
        # This is the only correlation path that survives the
        # scheduler→worker boundary.
        if (
            self.kv_role == "kv_consumer"
            and kv_transfer_params is not None
            and "correlation_id" in kv_transfer_params
        ):
            tmp_correlation_id_tracker[request.request_id] = (
                kv_transfer_params["correlation_id"]
            )

        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        recalc_last = (
            1
            if (
                self.load_specs[request.request_id].lmcache_cached_tokens
                == request.num_tokens
            )
            else 0
        )
        assert (
            num_external_tokens
            == self.load_specs[request.request_id].lmcache_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
            - recalc_last
        ), (
            f"Mismatch in tokens to load: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} "
            "(tokens in lmcache) - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens} "
            "(tokens in vllm) - "
            f"{recalc_last} "
            "(full lmcache hits subtracts last token to recalculate logits)"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        # We should load KV for:
        # 1. new requests
        # 2. preempted requests (once per recovery)
        # can_load will only be True if `update_state_after_alloc` has been called
        # which only happens when vLLM's KV manager has space to receive KV from LMCache
        for request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if request.req_id.startswith("mock_req"):
                continue
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                load_spec = self.load_specs.pop(req.req_id, None)
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                request_tracker = self._request_trackers[req.req_id]

                # Pass all_token_ids for preempted requests to restore
                # token_ids correctly for chunk key computation
                all_token_ids = None
                if req.resumed_from_preemption:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    assert vllm_request is not None, (
                        f"Preempted request {req.req_id} not found "
                        "in _unfinished_requests"
                    )
                    all_token_ids = list(vllm_request.all_token_ids)

                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                )

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self.config.save_decode_cache,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: this is a dangerous reference to the request object inside vllm
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = request.num_computed_tokens
                # tracker_len < num_computed_tokens during decode
                #   (important for save_decode_cache).
                # num_computed_tokens < tracker_len after preemption.
                tracker_len = len(request_tracker.token_ids)
                slice_base = min(num_current_tokens, tracker_len)
                new_token_ids = request.all_token_ids[
                    slice_base : slice_base + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            load_spec = self.load_specs.pop(req_id, None)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens

            # Handle both old and new versions of CachedRequestData
            if hasattr(cached_reqs, "resumed_req_ids"):
                # New version with resumed_req_ids
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                # Old version with resumed_from_preemption
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
            if preempted:
                assert load_spec is not None, (
                    f"Request {req_id} is preempted but was not given a load spec"
                )
                # num_computed_tokens should be reset to 0 during preemption
                # and then set to the number of already cached tokens (maxxing
                # prefix caching and lmcache)
                # this assumption is crucial for the update() call of RequestTracker
                # On full cache hit, get_num_new_matched_tokens subtracts 1
                # to force last-token recomputation. This only affects
                # num_computed_tokens when lmcache has all tokens AND
                # provides more than vLLM's local cache.
                expected = max(lmcache_cached_tokens, load_spec.vllm_cached_tokens)
                full_hit_adj = (
                    lmcache_cached_tokens == len(request.all_token_ids)
                    and lmcache_cached_tokens > load_spec.vllm_cached_tokens
                )
                if full_hit_adj:
                    expected -= 1
                assert request.num_computed_tokens == expected, (
                    f"Preempted request {req_id} has "
                    f"num_computed_tokens {request.num_computed_tokens} "
                    f"but expected {expected} "
                    f"(full_hit_adj={full_hit_adj})"
                )

            # When retrieve fail, vllm will call _handle_invalid_blocks to
            # reset request.num_computed_tokens, this will lead to
            # request_tracker.token_ids being not matched with vllm
            if num_current_tokens < len(request_tracker.token_ids):
                logger.warning(
                    "Request %s rolled back from %d to %d tokens; "
                    "truncating tracker state.",
                    req_id,
                    len(request_tracker.token_ids),
                    num_current_tokens,
                )
                num_token_slots = (
                    len(request_tracker.allocated_block_ids) * self._block_size
                )
                tokens_to_keep = num_current_tokens
                if num_token_slots < num_current_tokens:
                    logger.warning(
                        "Request %s tracker has %d token slots but %d tokens; "
                        "capping token_ids to slot capacity.",
                        req_id,
                        num_token_slots,
                        num_current_tokens,
                    )
                    tokens_to_keep = num_token_slots

                request_tracker.token_ids = list(request.all_token_ids[:tokens_to_keep])
                request_tracker.num_saved_tokens = min(
                    request_tracker.num_saved_tokens, tokens_to_keep
                )

            # Pass all_token_ids for preempted requests to restore
            # token_ids correctly for chunk key computation
            all_token_ids = list(request.all_token_ids) if preempted else None

            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Layerwise save uses request-scoped generators. If request finishes
        # without entering wait_for_save (abort/error/evict path), make sure
        # we release the generator entry to avoid leaking state.
        if getattr(self, "use_layerwise", False) and hasattr(
            self, "_layerwise_save_storers"
        ):
            self._layerwise_save_storers.pop(request.request_id, None)

        # Cleanup if request was aborted
        if request.status == RequestStatus.FINISHED_ABORTED and self.async_loading:
            # Cancel any ongoing async lookup and prefetch tasks on workers
            lookup_id = request.request_id
            assert self.lookup_client is not None
            self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]

        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        if self.config.get_extra_config_value(
            "enable_cache_usage_details_in_response", False
        ):
            request_tracker = self._request_trackers.get(request.request_id)
            if request_tracker:
                return_params = return_params or {}
                return_params["num_lmcache_cached_tokens"] = (
                    request_tracker.num_lmcache_cached_tokens
                )

        return False, return_params

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []
