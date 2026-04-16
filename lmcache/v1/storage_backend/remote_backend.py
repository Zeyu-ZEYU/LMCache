# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, TimeoutError
from typing import Any, Callable, List, Optional, Sequence, Set
import asyncio
import copy
import threading
import time

# Third Party
import yaml

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.kv_routing import route_kv_chunk
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.connector import CreateConnector
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.naive_serde import CreateSerde

logger = init_logger(__name__)


class RemoteBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional[LocalCPUBackend],
        dst_device: str = "cuda",
    ):
        super().__init__(dst_device=dst_device)
        self.put_tasks: Set[CacheEngineKey] = set()
        self.lock = threading.Lock()

        assert config.remote_url is not None

        self.remote_url = config.remote_url

        self.local_cpu_backend = local_cpu_backend

        self.loop = loop
        self.config = config
        self.metadata = metadata

        # Re-establish connection only when the connection
        # has been lost for 10 secs
        self.connection: Optional[RemoteConnector] = None
        self.min_reconnect_interval = 10
        self.failure_time = -1000000.0
        self.init_connection()

        assert config.remote_serde is not None
        self.serializer, self.deserializer = CreateSerde(
            config.remote_serde, metadata, config
        )

        # Precompute MLA mode status
        self._mla_worker_id_as0_mode = (
            config.get_extra_config_value(
                "remote_enable_mla_worker_id_as0", metadata.use_mla
            )
            and metadata.use_mla
            and metadata.world_size > 1
            and metadata.worker_id != 0
        )
        logger.info(f"metadata={metadata}")
        logger.info(
            f"Connected to remote storage at {config.remote_url}, "
            f"remote_mla_worker_id_as_0 mode: {self._mla_worker_id_as0_mode}"
        )

        # TODO(Jiayi): If we want to have cache admission policies,
        # we must make decision (whether to send or not) at the local side

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        # --- Head NIC splitting (lazy init to avoid port conflicts at startup) ---
        self.head_connection: Optional[RemoteConnector] = None
        self._head_nic_enabled = (
            config.enable_head_nic_split and config.head_nic_config_file
        )
        self._head_nic_init_args = (config, metadata, loop, local_cpu_backend)

        # NOTE: Health monitoring is now handled at the LMCacheEngine level
        # through HealthMonitor. RemoteBackend no longer manages its own
        # health monitoring. The HealthMonitor in LMCacheEngine will
        # register RemoteBackendHealthCheck for each RemoteBackend.

        self._setup_metrics()

        self._get_blocking_failed_count = 0
        self._put_failed_count = 0

    def _ensure_head_connection(self) -> Optional[RemoteConnector]:
        """Lazy-init: create head NIC connector on first use."""
        if self.head_connection is not None:
            return self.head_connection
        if not self._head_nic_enabled:
            return None
        self._init_head_nic_connector(*self._head_nic_init_args)
        return self.head_connection

    def _init_head_nic_connector(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional[LocalCPUBackend],
    ):
        """Create a second Mooncake connector for head NIC (mlx5_0)."""
        try:
            with open(config.head_nic_config_file, "r") as f:
                head_cfg = yaml.safe_load(f)

            # Build a modified config for the head NIC connector
            head_config = copy.copy(config)
            head_url = head_cfg.get("remote_url", config.remote_url)
            head_config.remote_url = head_url

            # Override extra_config with head NIC params
            head_extra = head_cfg.get("extra_config", {})
            head_config.extra_config = head_extra

            self.head_connection = CreateConnector(
                head_url,
                loop,
                local_cpu_backend,
                head_config,
                metadata,
            )
            # Ensure head connector uses zero-copy (not put_parts).
            # copy.copy of metaclass config may not propagate extra_config
            # correctly, causing save_chunk_meta to default to wrong value.
            inner = getattr(self.head_connection, "_connector", None)
            if inner and hasattr(inner, "save_chunk_meta"):
                inner.save_chunk_meta = False
            logger.info(
                "Head NIC connector initialized: url=%s device=%s",
                head_url,
                head_extra.get("device_name", "unknown"),
            )
        except Exception:
            logger.exception("Failed to initialize head NIC connector")
            self.head_connection = None

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.remote_put_task_num.set_function(
                lambda: len(self.put_tasks)
            )
            prometheus_logger.get_blocking_failed_count.set_function(
                lambda: self._get_blocking_failed_count
            )
            prometheus_logger.put_failed_count.set_function(
                lambda: self._put_failed_count
            )

    def __str__(self):
        return self.__class__.__name__

    def init_connection(self):
        # Initialize connection
        if self.connection is not None:
            return
        if (time.time() - self.failure_time) < self.min_reconnect_interval:
            logger.warning(
                "Connection will not be re-established yet "
                "since it has not been long enough since "
                "the last failure"
            )
            return
        try:
            assert self.config.remote_url is not None
            self.connection = CreateConnector(
                self.config.remote_url,
                self.loop,
                self.local_cpu_backend,
                self.config,
                self.metadata,
            )
            logger.info(
                f"Connection initialized/re-established at {self.config.remote_url}"
            )
        except IrrecoverableException:
            logger.error("Irrecoverable error during connection initialization")
            raise
        except Exception as e:
            with self.lock:
                self.failure_time = time.time()
            logger.warning(f"Failed to initialize/re-establish remote connection: {e}")
            self.connection = None

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        if self.connection is None:
            logger.warning("Connection is None in contains, returning False")
            return False

        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)

        try:
            if self.config.extra_config is not None and self.config.extra_config.get(
                "use_exists_sync", False
            ):
                return self.connection.exists_sync(key)
            else:
                future = asyncio.run_coroutine_threadsafe(
                    self.connection.exists(key), self.loop
                )
                res = future.result()
                return res
        except Exception as e:
            logger.warning(f"Remote connection failed in contains: {e}")
            logger.warning("Returning False")
            return False

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if self.connection is None:
            logger.warning("Connection is None in batched_contains, returning 0")
            return 0

        if not self.connection.support_batched_contains():
            return super().batched_contains(keys, pin)

        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        try:
            return self.connection.batched_contains(keys)
        except Exception as e:
            logger.warning(f"Remote connection failed in batched_contains: {e}")
            return 0

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.lock:
            return key in self.put_tasks

    def put_callback(self, future: Future, key: CacheEngineKey):
        with self.lock:
            self.put_tasks.discard(key)
        try:
            future.result()
        except Exception as e:
            self._put_failed_count += 1
            logger.error(f"Put task failed for key {key}: {e}")

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """
        Submit a put task to store KV cache to remote storage asynchronously.

        :param on_complete_callback: Optional callback invoked after the remote
            write completes. Callback exceptions are caught and logged.
        """

        def create_immediate_empty_future() -> Future:
            f: Future = Future()
            f.set_result(None)
            return f

        if self.connection is None:
            logger.warning("Connection is None in submit_put_task, returning None")
            return create_immediate_empty_future()

        # If MLA worker id as 0 mode is enabled, skip put tasks
        if self._mla_worker_id_as0_mode:
            return create_immediate_empty_future()

        if self.exists_in_put_tasks(key):
            return create_immediate_empty_future()

        memory_obj.ref_count_up()

        with self.lock:
            self.put_tasks.add(key)

        compressed_memory_obj = self.serializer.serialize(memory_obj)
        memory_obj.ref_count_down()

        def put_done_callback(f: Future) -> None:
            self.put_callback(f, key)
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(f"on_complete_callback failed for key {key}: {e}")

        # NOTE: No need to do error handling here
        # since the `future` is never waited
        future = asyncio.run_coroutine_threadsafe(
            self.connection.put(key, compressed_memory_obj), self.loop
        )
        future.add_done_callback(put_done_callback)
        return future

    def batched_put_callback(self, future: Future, keys: List[CacheEngineKey]):
        """
        Callback function for batched put tasks.
        """
        with self.lock:
            self.put_tasks.difference_update(keys)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
        layer_id: Optional[int] = None,
        num_layers: int = 0,
    ) -> None:
        """
        Submit batched put tasks to store KV caches to remote storage.

        :param on_complete_callback: Optional callback invoked once per key
            after that key's write completes (not once per batch).
        :param layer_id: Layer index for head NIC routing (optional).
        :param num_layers: Total layers for head NIC routing.
        """
        if self.connection is None:
            logger.warning(
                "Connection is None in batched_submit_put_task, returning None"
            )
            return
        if self.connection.support_batched_put():
            if self._mla_worker_id_as0_mode:
                return

            # First, increment reference counts for all objects
            for memory_obj in memory_objs:
                memory_obj.ref_count_up()

            compressed_memory_objs = []
            try:
                for memory_obj in memory_objs:
                    compressed_memory_objs.append(self.serializer.serialize(memory_obj))
            finally:
                # Always decrement reference counts for all objects,
                # regardless of whether serialization succeeded or failed
                for memory_obj in memory_objs:
                    memory_obj.ref_count_down()

            kv_put_start = time.perf_counter()

            def batched_done_callback(f: Future) -> None:
                elapsed_ms = (time.perf_counter() - kv_put_start) * 1000
                logger.info(
                    "KV transfer: %d chunks, %.1f ms (layer=%s)",
                    len(keys), elapsed_ms, layer_id,
                )
                self.batched_put_callback(f, list(keys))
                # Invoke per-key callback for each key in the batch
                if on_complete_callback is not None:
                    for key in keys:
                        try:
                            on_complete_callback(key)
                        except Exception as e:
                            logger.warning(
                                f"on_complete_callback failed for key {key}: {e}"
                            )

            # Extract preferred_segment from transfer_spec for
            # targeted RDMA WRITE (e.g., to decode node's segment).
            # Only Mooncake connector supports preferred_segment.
            preferred_segment = None
            if transfer_spec is not None and hasattr(
                transfer_spec, "receiver_rdma_host"
            ):
                preferred_segment = getattr(
                    transfer_spec, "receiver_rdma_host", None
                )

            put_kwargs: dict = {}
            if preferred_segment:
                put_kwargs["preferred_segment"] = preferred_segment

            # --- Head NIC routing (lazy init) ---
            if (
                self._head_nic_enabled
                and layer_id is not None
                and self._ensure_head_connection() is not None
            ):
                num_chunks = len(keys)
                head_idx = []
                tail_idx = []
                for ci in range(num_chunks):
                    nic = route_kv_chunk(
                        layer_id, ci, num_layers, num_chunks
                    )
                    if nic == "head":
                        head_idx.append(ci)
                    else:
                        tail_idx.append(ci)

                if head_idx:
                    h_keys = [keys[i] for i in head_idx]
                    h_objs = [compressed_memory_objs[i] for i in head_idx]
                    hf = asyncio.run_coroutine_threadsafe(
                        self.head_connection.batched_put(h_keys, h_objs),
                        self.loop,
                    )
                    hf.add_done_callback(batched_done_callback)
                if tail_idx:
                    t_keys = [keys[i] for i in tail_idx]
                    t_objs = [compressed_memory_objs[i] for i in tail_idx]
                    tf = asyncio.run_coroutine_threadsafe(
                        self.connection.batched_put(
                            t_keys, t_objs, **put_kwargs
                        ),
                        self.loop,
                    )
                    tf.add_done_callback(batched_done_callback)
            else:
                # Default: all chunks via tail (main connection)
                future = asyncio.run_coroutine_threadsafe(
                    self.connection.batched_put(
                        keys, compressed_memory_objs, **put_kwargs
                    ),
                    self.loop,
                )
                future.add_done_callback(batched_done_callback)
        else:
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                self.submit_put_task(
                    key, memory_obj, on_complete_callback=on_complete_callback
                )

    @_lmcache_nvtx_annotate
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in get_blocking "
                "(likely scheduler role), returning None"
            )
            return None

        if self.connection is None:
            logger.warning("Connection is None in get_blocking, returning None")
            return None
        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)
        t1 = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)

        try:
            memory_obj = future.result(self.config.blocking_timeout_secs)
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.warning("get blocking timeout, trigger cancel the future task")
                future.cancel()
            logger.warning("Error occurred in get_blocking: %s, return None", e)
            memory_obj = None

        t2 = time.perf_counter()
        self.stats_monitor.update_interval_remote_time_to_get_sync((t2 - t1) * 1000)
        if memory_obj is None:
            self._get_blocking_failed_count += 1
            return None
        decompressed_memory_obj = self.deserializer.deserialize(memory_obj)
        t3 = time.perf_counter()
        logger.debug(
            "Get takes %.6f msec, deserialization takes %.6f msec",
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
        )
        return decompressed_memory_obj

    @property
    def get_blocking_failed_count(self):
        return self._get_blocking_failed_count

    @property
    def put_failed_count(self):
        return self._put_failed_count

    def _batched_get_from_connection(
        self,
        conn: RemoteConnector,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Helper: batched get from a specific connector."""
        if conn.support_batched_get():
            future = asyncio.run_coroutine_threadsafe(
                conn.batched_get(keys), self.loop
            )
            try:
                return future.result(self.config.blocking_timeout_secs)
            except Exception as e:
                if isinstance(e, TimeoutError):
                    future.cancel()
                logger.warning("_batched_get_from_connection error: %s", e)
                return [None] * len(keys)
        return [None] * len(keys)

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
        num_layers: int = 0,
    ) -> List[Optional[MemoryObj]]:
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in batched_get_blocking "
                "(likely scheduler role), returning None list"
            )
            return [None] * len(keys)

        if self.connection is None:
            logger.warning("Connection is None in batched_get_blocking, returning None")
            return [None] * len(keys)

        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        t1 = time.perf_counter()

        # --- Head NIC routing for retrieve (lazy init) ---
        if self._head_nic_enabled and keys and self._ensure_head_connection() is not None:
            layer_id = getattr(keys[0], "layer_id", None)
            if layer_id is not None:
                num_chunks = len(keys)
                head_idx = []
                tail_idx = []
                for ci in range(num_chunks):
                    nic = route_kv_chunk(
                        layer_id, ci, num_layers, num_chunks
                    )
                    if nic == "head":
                        head_idx.append(ci)
                    else:
                        tail_idx.append(ci)

                if head_idx and not tail_idx:
                    # All head — retrieve from head connector
                    return self._batched_get_from_connection(
                        self.head_connection, keys
                    )
                elif tail_idx and not head_idx:
                    pass  # Fall through to normal path
                elif head_idx and tail_idx:
                    # Mixed: query both, merge results
                    results: list[Optional[MemoryObj]] = [None] * num_chunks
                    h_keys = [keys[i] for i in head_idx]
                    t_keys = [keys[i] for i in tail_idx]
                    h_objs = self._batched_get_from_connection(
                        self.head_connection, h_keys
                    )
                    t_objs = self._batched_get_from_connection(
                        self.connection, t_keys
                    )
                    for i, idx in enumerate(head_idx):
                        results[idx] = h_objs[i] if h_objs else None
                    for i, idx in enumerate(tail_idx):
                        results[idx] = t_objs[i] if t_objs else None
                    return results

        # batched get (default: tail only)
        if self.connection.support_batched_get():
            future = asyncio.run_coroutine_threadsafe(
                self.connection.batched_get(keys), self.loop
            )
            try:
                memory_objs = future.result(self.config.blocking_timeout_secs)
            except Exception as e:
                if isinstance(e, TimeoutError):
                    logger.warning(
                        "batched get blocking timeout, trigger cancel the future task"
                    )
                    future.cancel()
                else:
                    logger.warning(
                        f"Error occurred in batched_get_blocking: {e}, "
                        f"returning None list"
                    )
                memory_objs = [None] * len(keys)
        else:
            remote_backend_individual_get_stats: dict[
                CacheEngineKey, dict[str, float]
            ] = {}
            retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
            if retrieve_stats is not None:
                retrieve_stats.detailed_metrics[
                    "remote_backend_individual_get_stats"
                ] = remote_backend_individual_get_stats

            futures = [
                asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)
                for key in keys
            ]
            memory_objs = []
            failed = False
            for fut in futures:
                if not failed:
                    try:
                        memory_obj = fut.result(self.config.blocking_timeout_secs)
                    except Exception as e:
                        failed = True
                        if isinstance(e, TimeoutError):
                            logger.warning(
                                "get blocking timeout, trigger cancel the future task"
                            )
                            fut.cancel()
                        else:
                            logger.warning(
                                f"Error occurred in get_blocking: {e}, returning None"
                            )
                        memory_obj = None
                    memory_objs.append(memory_obj)
                else:
                    memory_objs.append(None)
                    fut.cancel()

        t2 = time.perf_counter()
        duration = t2 - t1
        self.stats_monitor.update_interval_remote_time_to_get_sync(duration * 1000)

        retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
        if retrieve_stats is not None:
            retrieve_stats.detailed_metrics[
                "remote_backend_batched_get_blocking_time"
            ] = (
                retrieve_stats.detailed_metrics.get(
                    "remote_backend_batched_get_blocking_time", 0.0
                )
                + duration
            )
        decompressed_memory_objs: list[Optional[MemoryObj]] = []
        error_happened = False
        for memory_obj in memory_objs:
            if memory_obj is None:
                error_happened = True
                decompressed_memory_objs.append(None)
            else:
                decompressed_memory_objs.append(
                    self.deserializer.deserialize(memory_obj)
                )
        if error_happened:
            self._get_blocking_failed_count += 1

        assert len(decompressed_memory_objs) == len(keys), (
            f"keys length: {len(keys)}, "
            f"decompressed memory objs length: {len(decompressed_memory_objs)}"
        )
        return decompressed_memory_objs

    async def support_batched_async_contains(self) -> bool:
        return (
            self.connection is not None
            and self.connection.support_batched_async_contains()
        )

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if self.connection is None:
            logger.warning("Connection is None in batched_async_contains, returning 0")
            return 0
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        try:
            assert self.connection.support_batched_async_contains(), (
                f"Connector {self.connection} does not support batched async contains"
            )
            # warning, this timeout will not actually stop the
            # scheduler from waiting for the result
            return await asyncio.wait_for(
                self.connection.batched_async_contains(lookup_id, keys, pin),
                self.config.blocking_timeout_secs,
            )
        except asyncio.TimeoutError:
            logger.warning("batched_async_contains timed out")
            return 0
        except Exception as e:
            logger.warning(f"Error occurred in batched_async_contains: {e}")
            return 0

    async def support_batched_get_non_blocking(self) -> bool:
        return (
            self.connection is not None
            and self.connection.support_batched_get_non_blocking()
        )

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> List[MemoryObj]:
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in batched_get_non_blocking "
                "(likely scheduler role), returning empty list"
            )
            return []

        if self.connection is None:
            logger.warning(
                "Connection is None in batched_get_non_blocking, returning empty list"
            )
            return []
        try:
            # warning, this timeout will not actually stop the
            # scheduler from waiting for the result
            return await asyncio.wait_for(
                self.connection.batched_get_non_blocking(lookup_id, keys),
                self.config.blocking_timeout_secs,
            )
        except asyncio.TimeoutError:
            logger.warning("batched_get_non_blocking timed out")
            return []
        except Exception as e:
            logger.warning(f"Error occurred in batched_get_non_blocking: {e}")
            return []

    def pin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support pin. "
            "This method is a no-op and will return True."
        )
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support unpin. "
            "This method is a no-op and will return True."
        )
        return True

    def remove(self, key, force=True):
        if self.connection is None:
            logger.warning("Connection is None in remove, returning False")
            return False

        try:
            return self.connection.remove_sync(key)
        except Exception as e:
            logger.exception(
                f"Failed to remove key {key} from remote backend, error: {e}"
            )
            return False

    def get_allocator_backend(self):
        assert self.local_cpu_backend is not None, (
            "local_cpu_backend is required for get_allocator_backend, "
            "should not be called in scheduler role"
        )
        return self.local_cpu_backend

    def close(self):
        try:
            assert self.connection is not None
            future = asyncio.run_coroutine_threadsafe(
                self.connection.close(), self.loop
            )
            future.result()
            logger.info("Remote backend closed.")
        except Exception as e:
            logger.warning(f"Error occurred when closing remote connection: {e}")
