import asyncio
import concurrent.futures
import gc
import hashlib
import json
import logging
import os
import time
import uuid
import warnings
from copy import deepcopy
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryType
from mem0.configs.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    generate_additive_extraction_prompt,
)
from mem0.exceptions import ValidationError as Mem0ValidationError
from mem0.memory.base import MemoryBase
from mem0.memory.setup import mem0_dir, setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import MEM0_TELEMETRY, capture_event
from mem0.memory.notices import (
    PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS,
    detect_scale_threshold_from_add_result,
    detect_scale_threshold_from_top_k,
    detect_decay_usage_from_delete,
    detect_decay_usage_from_delete_all,
    detect_temporal_usage_from_metadata,
    detect_temporal_usage_from_search,
    display_decay_usage_notice,
    display_decay_usage_notice_async,
    display_first_run_notice,
    display_first_run_notice_async,
    display_performance_slow_query_notice,
    display_performance_slow_query_notice_async,
    display_scale_threshold_notice,
    display_scale_threshold_notice_async,
    display_temporal_usage_notice,
    display_temporal_usage_notice_async,
    get_decay_feature_error_message,
    get_decay_feature_error_message_async,
    get_temporal_feature_error_message,
    get_temporal_feature_error_message_async,
)
from mem0.memory.utils import (
    extract_json,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from mem0.utils.entity_extraction import extract_entities, extract_entities_batch
from mem0.utils.factory import (
    EmbedderFactory,
    LlmFactory,
    RerankerFactory,
    VectorStoreFactory,
)
from mem0.utils.lemmatization import lemmatize_for_bm25
from mem0.utils.scoring import (
    ENTITY_BOOST_WEIGHT,
    get_bm25_params,
    normalize_bm25,
    score_and_rank,
)
from mem0.vector_stores.base import VectorStoreBase

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)


def _vector_store_list_rows(listed):
    if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list):
        return listed[0]
    if isinstance(listed, (list, tuple)):
        return listed
    return []


# Fields that hold runtime auth/connection objects and must be preserved.
# These are non-serializable objects (e.g. AWSV4SignerAuth, RequestsHttpConnection)
# needed by clients like OpenSearch — not sensitive strings to redact.
_RUNTIME_FIELDS = frozenset({
    "http_auth",
    "auth",
    "connection_class",
    "ssl_context",
})

# Fields that are known to contain sensitive secrets and must be redacted.
_SENSITIVE_FIELDS_EXACT = frozenset({
    "api_key",
    "secret_key",
    "private_key",
    "access_key",
    "password",
    "credentials",
    "credential",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "auth_token",
    "session_token",
    "client_secret",
    "auth_client_secret",
    "azure_client_secret",
    "service_account_json",
    "aws_session_token",
})

# Suffixes that indicate a field likely holds a secret value.
_SENSITIVE_SUFFIXES = (
    "_password",
    "_secret",
    "_token",
    "_credential",
    "_credentials",
)

# Entity parameters that must be passed via filters, not top-level kwargs
ENTITY_PARAMS = frozenset({"user_id", "agent_id", "run_id"})


def _reject_top_level_entity_params(kwargs: Dict[str, Any], method_name: str) -> None:
    """Reject top-level entity parameters - must use filters instead."""
    invalid_keys = ENTITY_PARAMS & set(kwargs.keys())
    if invalid_keys:
        raise ValueError(
            f"Top-level entity parameters {invalid_keys} are not supported in {method_name}(). "
            f"Use filters={{'user_id': '...'}} instead."
        )


def _validate_and_trim_entity_id(value: Optional[str], name: str) -> Optional[str]:
    """
    Validates and normalizes an entity ID.
    - Trims leading/trailing whitespace
    - Rejects empty or whitespace-only strings
    - Rejects strings containing internal whitespace

    Args:
        value: The entity ID value to validate
        name: The parameter name (for error messages)

    Returns:
        The trimmed entity ID, or None if input is None

    Raises:
        ValueError: If entity ID is invalid
    """
    if value is None:
        return None
    trimmed = value.strip()
    if trimmed == "":
        raise ValueError(
            f"Invalid {name}: cannot be empty or whitespace-only. Provide a valid identifier."
        )
    if any(c.isspace() for c in trimmed):
        raise ValueError(
            f"Invalid {name}: cannot contain whitespace. Provide a valid identifier without spaces."
        )
    return trimmed


def _validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    """
    Validates search parameters.

    Args:
        threshold: Similarity threshold (must be between 0 and 1)
        top_k: Number of results to return (must be non-negative integer)

    Raises:
        ValueError: If threshold or top_k are invalid
    """
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a valid number")
        if threshold < 0 or threshold > 1:
            raise ValueError(
                f"Invalid threshold: {threshold}. Must be between 0 and 1 (inclusive)."
            )
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a valid integer")
        if top_k < 0:
            raise ValueError(
                f"Invalid top_k: {top_k}. Must be a non-negative integer."
            )


def _validate_and_trim_search_query(query: str) -> str:
    """
    Validates and normalizes a search query before embedding/vector search.

    Raises:
        ValueError: If query is not a string or is empty/whitespace-only.
    """
    if not isinstance(query, str):
        raise ValueError("Invalid query: must be a non-empty string.")
    trimmed = query.strip()
    if not trimmed:
        raise ValueError("Invalid query: cannot be empty or whitespace-only.")
    return trimmed


def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field should be redacted for telemetry safety.

    Uses a layered approach:
    1. Runtime fields (allowlist) — always preserved, highest priority.
    2. Exact deny list — known secret field names.
    3. Suffix deny list — catches patterns like db_password, auth_secret, etc.
    """
    name = field_name.lower().strip()
    if name in _RUNTIME_FIELDS:
        return False
    if name in _SENSITIVE_FIELDS_EXACT:
        return True
    return any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def _safe_deepcopy_config(config):
    """Safely deepcopy config, falling back to dict-based cloning for non-serializable objects."""
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"Deepcopy failed, using dict-based cloning: {e}")

        config_class = type(config)

        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump()
            except Exception:
                clone_dict = dict(config.__dict__)
        else:
            clone_dict = dict(config.__dict__)

        # Restore runtime fields, redact sensitive ones
        for field_name in list(clone_dict.keys()):
            if field_name in _RUNTIME_FIELDS and hasattr(config, field_name):
                clone_dict[field_name] = getattr(config, field_name)
            elif _is_sensitive_field(field_name):
                clone_dict[field_name] = None

        try:
            return config_class(**clone_dict)
        except Exception:
            logger.debug("Config reconstruction failed, returning shallow dict clone")
            return type("Config", (), clone_dict)()


def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    根据会话作用域和 actor 信息，构造“写入用 metadata”和“查询用 filters”。

    这个方法的核心作用是把输入拆成两份：

    1. `base_metadata_template`：给后续写入记忆时使用的元数据模板。
       它会保留传入的 `input_metadata`，并补上 `user_id` / `agent_id` / `run_id`。
    2. `effective_query_filters`：给后续查询已有记忆时使用的过滤条件。
       它会保留传入的 `input_filters`，并补上同样的作用域字段。

    如果同时传了 `actor_id`，它会作为额外的查询过滤条件附加进去。
    这个 `actor_id` 只用于查询，不会写回到 `base_metadata_template`，
    因为存储侧的 actor 往往会在后续消息解析阶段再确定。

    Args:
        user_id (Optional[str]): 用户作用域 ID，用来隔离不同用户的记忆。
        agent_id (Optional[str]): Agent 作用域 ID，用来隔离某个 agent 的记忆。
        run_id (Optional[str]): 运行/会话作用域 ID，用来隔离某次执行过程的记忆。
        actor_id (Optional[str]): 显式指定的 actor ID。
            如果传入，会作为额外的查询过滤条件使用。
        input_metadata (Optional[Dict[str, Any]]): 写入时要携带的基础元数据。
            这里会被拷贝后再补充作用域字段，避免直接修改外部对象。
        input_filters (Optional[Dict[str, Any]]): 查询时要携带的基础过滤条件。
            这里也会被拷贝后再补充作用域字段和 actor 条件。

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: 返回两个字典：
            - `base_metadata_template`：写入记忆时使用的元数据模板。
            - `effective_query_filters`：查询已有记忆时使用的最终过滤条件。
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- 校验并写入所有传入的作用域 ID ----------
    session_ids_provided = []

    # 先统一清理 ID：去掉首尾空白，并拒绝空字符串/含空格的非法值。
    user_id = _validate_and_trim_entity_id(user_id, "user_id")
    agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
    run_id = _validate_and_trim_entity_id(run_id, "run_id")

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- 可选的 actor 过滤 ----------
    # actor_id 的优先级：显式传入的 actor_id > input_filters 里的 actor_id。
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


def _build_session_scope(filters):
    """
    把过滤条件里的实体 ID 拼成一个确定性的会话作用域字符串。

    比如 {"user_id": "alice", "agent_id": "bot"} → "agent_id=bot&user_id=alice"。
    固定按字母序排序，保证同一批过滤条件无论传入顺序如何，都能生成相同的 scope key，
    用于 SQLite 历史记录和消息上下文的存取。
    """
    parts = []
    for key in sorted(["user_id", "agent_id", "run_id"]):
        val = filters.get(key)
        if val:
            parts.append(f"{key}={val}")
    return "&".join(parts)


def _entity_collection_name(provider: str, collection_name: str) -> str:
    """
    根据向量库 provider 和主 collection 名称，生成实体库的 collection 名称。

    s3_vectors 用连字符（AWS S3 不允许下划线），其他向量库统一用下划线分隔。
    例如 ("pgvector", "mem0_memories") → "mem0_memories_entities"
    """
    separator = "-" if provider == "s3_vectors" else "_"
    return f"{collection_name}{separator}entities"


def _normalize_expiration_date(value: Any) -> Optional[str]:
    """
    把各种形态的过期时间归一化成 YYYY-MM-DD 字符串。

    支持三种输入格式：
    - datetime 对象：取 .date() 后转字符串
    - date 对象：直接转字符串
    - str：尝试按 ISO 格式解析，失败则抛 ValueError

    返回 None 表示不设置过期时间（透传 None）。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError("expiration_date must be a valid date in YYYY-MM-DD format.") from exc
    raise ValueError("expiration_date must be a date string in YYYY-MM-DD format.")


def _payload_is_expired(payload: Optional[Dict[str, Any]]) -> bool:
    """
    判断一条记忆的 payload 是否已过期。

    读取 payload 里的 `expiration_date` 字段（YYYY-MM-DD 格式），
    与当前 UTC 日期做比较，早于今天则视为过期。
    - payload 为空、或不含 expiration_date，视为永不过期，返回 False。
    - 日期格式无法解析时，同样返回 False（容错处理）。
    """
    if not payload:
        return False
    expiration_date = payload.get("expiration_date")
    if not expiration_date:
        return False
    try:
        return date.fromisoformat(str(expiration_date)) < datetime.now(timezone.utc).date()
    except ValueError:
        return False


setup_config()
logger = logging.getLogger(__name__)

_UNSET = object()
_PROJECT_UPDATE_UNSUPPORTED_ERROR = "Project updates are not supported by the OSS Memory SDK."


class _OSSProject:
    def update(
        self,
        custom_instructions: Optional[str] = None,
        custom_categories: Optional[list] = None,
        retrieval_criteria: Optional[list] = None,
        multilingual: Optional[bool] = None,
        decay: Optional[bool] = None,
    ):
        if decay is True:
            raise ValueError(get_decay_feature_error_message("sync", "project.update", "decay"))
        raise ValueError(_PROJECT_UPDATE_UNSUPPORTED_ERROR)


class _AsyncOSSProject:
    async def update(
        self,
        custom_instructions: Optional[str] = None,
        custom_categories: Optional[list] = None,
        retrieval_criteria: Optional[list] = None,
        multilingual: Optional[bool] = None,
        decay: Optional[bool] = None,
    ):
        if decay is True:
            raise ValueError(await get_decay_feature_error_message_async("async", "project.update", "decay"))
        raise ValueError(_PROJECT_UPDATE_UNSUPPORTED_ERROR)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        # Entity store is initialized lazily on first use
        self._entity_store = None

        if MEM0_TELEMETRY:
            # Create telemetry config manually to avoid deepcopy issues with thread locks
            telemetry_config_dict = {}
            if hasattr(self.config.vector_store.config, 'model_dump'):
                # For pydantic models
                telemetry_config_dict = self.config.vector_store.config.model_dump()
            else:
                # For other objects, manually copy common attributes
                for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                    if hasattr(self.config.vector_store.config, attr):
                        telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

            # Override collection name for telemetry
            telemetry_config_dict['collection_name'] = "mem0migrations"

            # Set path for file-based vector stores
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config_dict['path'] = os.path.join(mem0_dir, provider_path)
                os.makedirs(telemetry_config_dict['path'], exist_ok=True)

            # Create the config object using the same class as the original
            telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
            self._telemetry_vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, telemetry_config
            )
        if getattr(type(self.vector_store), "keyword_search", None) is VectorStoreBase.keyword_search:
            logger.warning(
                "The '%s' vector store does not support keyword search. "
                "Hybrid (BM25) scoring will be disabled and search will use "
                "semantic similarity only. To enable hybrid search, switch to a "
                "store with keyword_search support (e.g. qdrant, elasticsearch, pgvector).",
                self.config.vector_store.provider,
            )

        capture_event("mem0.init", self, {"sync_type": "sync"})

    @property
    def project(self):
        return _OSSProject()

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            entity_collection = _entity_collection_name(self.config.vector_store.provider, self.collection_name)
            # Set collection name on the cloned config
            if hasattr(entity_config, 'collection_name'):
                entity_config.collection_name = entity_collection
            elif isinstance(entity_config, dict):
                entity_config['collection_name'] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                if hasattr(entity_config, "client"):
                    entity_config.client = self.vector_store.client
                elif isinstance(entity_config, dict):
                    entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    @staticmethod
    def _normalize_entity_text(value: str) -> str:
        return " ".join(value.strip().lower().split())

    def _existing_entities_by_text(self, filters):
        """Return existing entity rows keyed by normalized payload data."""
        try:
            listed = self.entity_store.list(filters=filters, top_k=10000)
        except Exception as e:
            logger.debug(f"Exact entity lookup failed, falling back to semantic dedup: {e}")
            return {}

        rows_by_text = {}
        for row in _vector_store_list_rows(listed):
            payload = getattr(row, "payload", None) or {}
            text = payload.get("data")
            if not isinstance(text, str):
                continue
            normalized = self._normalize_entity_text(text)
            if normalized and normalized not in rows_by_text:
                rows_by_text[normalized] = row
        return rows_by_text

    def _upsert_entity(self, entity_text, entity_type, memory_id, filters):
        """
        把一个实体写入实体库，并将其与指定的 memory_id 关联。

        查找逻辑（按优先级）：
        1. 精确匹配：先在本地缓存的实体列表里按规范化文本查找，避免重复向量搜索。
        2. 语义匹配：精确匹配未命中时，执行向量相似度搜索，相似度 >= 0.95 视为同一实体。

        匹配到已有实体 → 将 memory_id 追加到 linked_memory_ids，更新 payload。
        未匹配到 → 新建实体记录，payload 包含 data/entity_type/linked_memory_ids 和作用域字段。

        任何异常均以 warning 级别记录并吞掉，不影响主流程。
        """
        try:
            # 生成实体的向量表示，用于后续语义检索
            entity_embedding = self.embedding_model.embed(entity_text, "add")
            # 只保留有效的作用域字段，用于过滤隔离（不同 user/agent/run 的实体互不影响）
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
            # 先尝试精确文本匹配（O(1) 缓存查找，避免每次向量搜索）
            exact_match = self._existing_entities_by_text(search_filters).get(self._normalize_entity_text(entity_text))

            existing = []
            if exact_match is None:
                existing = self.entity_store.search(
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=1,
                    filters=search_filters,
                )

            semantic_match = existing[0] if existing and existing[0].score >= 0.95 else None
            match = exact_match or semantic_match
            if match:
                # Update existing entity's linked_memory_ids
                payload = match.payload or {}
                linked_ids = payload.get("linked_memory_ids", [])
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    self.entity_store.update(
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                # Create new entity
                entity_id = str(uuid.uuid4())
                entity_payload = {
                    "data": entity_text,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                self.entity_store.insert(
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}': {e}")

    def _remove_memory_from_entity_store(self, memory_id, filters):
        """
        从实体库中移除指定 memory_id 的所有关联关系。

        遍历当前作用域（user_id/agent_id/run_id）下的全部实体：
        - 若实体的 linked_memory_ids 包含 memory_id：
          - 移除后链接列表为空 → 直接删除该实体记录（孤立实体无保留价值）
          - 链接列表仍有其他记忆 → 重新 embed 实体文本并更新 payload
            （更新向量库时必须同时提供新向量，不能只更新 payload）

        设计原则：
        - 实体库未初始化（self._entity_store is None）时静默跳过
        - 单条实体的操作失败以 debug 级别记录，不中断整个清理循环
        - 外层异常以 warning 级别记录，永远不影响主删除/更新流程
        """
        if self._entity_store is None:
            return
        # 只使用作用域相关字段做过滤，避免误操作其他 user/agent 的实体
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = self.entity_store.list(filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        try:
                            self.entity_store.delete(vector_id=row.id)
                        except Exception as e:
                            logger.debug(f"Entity delete failed for id={row.id}: {e}")
                    else:
                        entity_text = payload.get("data")
                        if not isinstance(entity_text, str) or not entity_text:
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup")
                            continue
                        try:
                            vec = self.embedding_model.embed(entity_text, "update")
                        except Exception as e:
                            logger.debug(f"Entity re-embed failed for '{entity_text}': {e}")
                            continue
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            self.entity_store.update(
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id}: {e}")
                except Exception as e:
                    logger.debug(f"Entity cleanup error: {e}")
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id}: {e}")

    def _link_entities_for_memory(self, memory_id, text, filters):
        """
        从 text 中提取实体，并将它们与 memory_id 关联写入实体库。

        这是 add() Phase 7 的单记忆简化版本：
        1. 调用 extract_entities() 从文本中识别实体（类型 + 文本）
        2. 对每个唯一实体调用 _upsert_entity() 建立关联
        3. 用规范化文本去重（seen set），避免同名实体重复写入
        4. 任何单个实体的关联失败以 debug 级别记录，不影响其他实体
        5. 整体失败以 warning 级别记录，永远不影响主流程
        """
        try:
            entities = extract_entities(text)
            if not entities:
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = self._normalize_entity_text(entity_text)
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    self._upsert_entity(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}': {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id}: {e}")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        expiration_date: Optional[Any] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str | list[dict]): 要写入记忆的原始消息。
                既可以传一段字符串，也可以传单条消息 dict，或者传消息列表。
                最终会统一整理成消息序列，再进入抽取/写库流程。
            user_id (str, optional): 记忆所属用户的作用域 ID。
                这是最常见的隔离维度，写入和搜索时都会用它来限定范围。
            agent_id (str, optional): 记忆所属 agent 的作用域 ID。
                当你想按 agent 维度存储或检索记忆时使用。
            run_id (str, optional): 记忆所属一次运行/会话的作用域 ID。
                适合把某次执行过程中的记忆单独隔离出来。
            metadata (dict, optional): 附加元数据。
                会被一起写入存储层，常用于记录来源、标签、模型上下文等业务信息。
            timestamp (Any, optional): 平台侧的时间控制参数。
                OSS 版本不支持；如果传入会直接报错，避免误以为生效。
            expiration_date (Any, optional): 过期日期，格式通常是 `YYYY-MM-DD`。
                设置后，过期记忆默认不会在 search/get_all 中返回，除非显式开启 show_expired。
            infer (bool, optional): 是否让 LLM 先“理解再写入”。
                True 时会抽取关键事实并判断新增/更新/删除；False 时直接把原消息当作记忆写入。
            memory_type (str, optional): 记忆类型。
                目前只显式支持 `procedural_memory`，表示程序性/操作性记忆；其它值会被拒绝。
            prompt (str, optional): 自定义抽取提示词。
                用来覆盖默认提示词，影响 LLM 看到的上下文和抽取策略。


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            Mem0ValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """
        if timestamp is not None:
            raise ValueError(get_temporal_feature_error_message("sync", "add", "timestamp"))

        # 1) 先把时间/过期时间等平台能力做拦截与归一化，OSS 只保留支持的那部分。
        normalized_expiration_date = _normalize_expiration_date(expiration_date)
        # 1b) 探测 metadata 中是否携带了时间/日期语义的字段（key 名如date/timestamp/created_at，
        #     值如ISO格式、相对时间词、Unix时间戳等）。如果命中，OSS 不会报错（与 timestamp 参数不同），
        #     而是返回一个标记，后续通过 display_temporal_usage_notice 引导用户了解平台侧的时间功能。
        #     这不是硬拦截，只是"软性提示"：OSS 不支持时间语义，但 SDK 会尽量告知用户替代方案。
        temporal_usage_notice = detect_temporal_usage_from_metadata(metadata)
        # 2) 统一构造元数据和作用域过滤条件：user_id / agent_id / run_id 会被折叠进同一套过滤器里。
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )
        if normalized_expiration_date is not None:
            processed_metadata["expiration_date"] = normalized_expiration_date

        # 3) memory_type 目前只显式支持 procedural_memory，其它值直接拒绝，避免静默走错分支。
        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        # 4) 入参兼容三种形态：字符串、单条 dict、list[dict]；统一转成消息列表方便后续处理。
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        # 5) agent + procedural_memory 走专用路径：直接生成程序性记忆，不进入通用的向量抽取流水线。
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            scale_threshold_notice = detect_scale_threshold_from_add_result(self, results)
            if temporal_usage_notice:
                display_temporal_usage_notice(self, "sync", "add", *temporal_usage_notice)
            elif scale_threshold_notice:
                display_scale_threshold_notice(self, "sync", "add", *scale_threshold_notice)
            else:
                display_first_run_notice(self, "sync", "add")
            return results

        # 6) 如果开启视觉能力，先把图片/多模态消息转成模型可消费的结构；否则只做普通消息规范化。
        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        # 7) 通用记忆写入：先让 LLM 抽取增量事实，再写入向量库和历史记录。
        vector_store_result = self._add_to_vector_store(messages, processed_metadata, effective_filters, infer, prompt=prompt)
        scale_threshold_notice = detect_scale_threshold_from_add_result(self, vector_store_result)
        if temporal_usage_notice:
            display_temporal_usage_notice(self, "sync", "add", *temporal_usage_notice)
        elif scale_threshold_notice:
            display_scale_threshold_notice(self, "sync", "add", *scale_threshold_notice)
        else:
            display_first_run_notice(self, "sync", "add")
        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer, prompt=None):
        """
        将输入消息转成可持久化的记忆，并写入向量库/历史库（同步版）。

        Args:
            messages (list): 原始对话消息列表（role/content/name 等字段）。
            metadata (dict): 基础元数据模板，会按消息或动作扩展后写入 payload。
            filters (dict): 作用域过滤条件（如 user_id/agent_id/run_id）。
            infer (bool): 是否启用 LLM 推理抽取。False 时直接逐条写入消息文本。
            prompt (str, optional): 自定义抽取提示词，优先级高于实例级 custom_instructions。

        Returns:
            list: 记忆变更结果列表（ADD/UPDATE/DELETE 等动作）。
        """
        if not infer:
            # 非推理模式：把每条用户消息直接当成一条记忆写入，不做 LLM 抽取。
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                mem_id = self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # === V3 PHASED BATCH PIPELINE ===

        # Phase 0: 上下文采集
        # 先取最近对话历史，再让 LLM 在“当前输入 + 历史上下文”里判断该新增/更新/删除什么。
        session_scope = _build_session_scope(filters)
        # 从历史库取最近 10 条上下文，帮助 LLM 做“增量判断”而不是脱离上下文地抽取
        last_messages = self.db.get_last_messages(session_scope, limit=10)
        # 把结构化消息拼成统一文本，作为向量检索 query 和抽取提示词输入
        parsed_messages = parse_messages(messages)

        # Phase 1: 召回已有记忆
        # 先查一次向量库，拿到当前范围内的已有记忆，作为 LLM 去重和增量判断的参照。
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        query_embedding = self.embedding_model.embed(parsed_messages, "search")
        existing_results = self.vector_store.search(
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # 为了降低 LLM 幻觉，把真实 UUID 映射成短整数 ID 再交给模型描述。
        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM 抽取（单次调用）
        # 根据当前作用域决定提示词：agent 作用域会额外注入 agent 语境。
        is_agent_scoped = bool(filters.get("agent_id")) and not filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX

        # 用户可自定义提示词；没有就用实例级 custom_instructions。
        custom_instr = prompt or self.custom_instructions

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
        )

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

        # 解析 LLM 返回：先去代码块，再按 JSON 解析，解析失败就尝试从文本中再提取 JSON。
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            extracted_memories = []

        if not extracted_memories:
            # 即使没有抽出任何记忆，也要保存原始消息历史，方便下次 add/search 复用上下文。
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 3: 批量向量化
        # 把抽出的记忆文本尽量批量 embedding，失败再降级为逐条 embedding。
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        except Exception:
            # 批量 embedding 不可用时，逐条补救，保证可用性优先。
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = self.embedding_model.embed(text, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed memory text: {e}")

        # Phase 4/5: 逐条处理 + 哈希去重
        # 先收集已有哈希，再过滤当前批次重复文本，避免重复写入向量库。
        existing_hashes = set()
        for mem in existing_results:
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            if h:
                existing_hashes.add(h)

        records = []  # (memory_id, text, embedding, payload)
        seen_hashes = set()  # dedup within the current batch
        for mem in extracted_memories:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue

            # 同一条文本在“已存在结果”和“本批次内部”都要去重。
            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match): {text[:50]}")
                continue
            seen_hashes.add(mem_hash)

            # 组织最终落库 payload：保留原文、词形归一结果、hash、时间戳等。
            text_lemmatized = lemmatize_for_bm25(text)

            memory_id = str(uuid.uuid4())
            mem_metadata = deepcopy(metadata)
            mem_metadata["data"] = text
            mem_metadata["text_lemmatized"] = text_lemmatized
            mem_metadata["hash"] = mem_hash
            if "created_at" not in mem_metadata:
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            if mem.get("attributed_to"):
                mem_metadata["attributed_to"] = mem["attributed_to"]

            records.append((memory_id, text, embed_map[text], mem_metadata))

        if not records:
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]

        try:
            self.vector_store.insert(
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        except Exception:
            # Fallback: insert one by one
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                try:
                    self.vector_store.insert(vectors=[vec], ids=[mid], payloads=[pay])
                except Exception as e:
                    logger.error(f"Failed to insert memory {mid}: {e}")

        # Batch history
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in records
        ]
        try:
            self.db.batch_add_history(history_records)
        except Exception:
            # Fallback: add one by one
            for hr in history_records:
                try:
                    self.db.add_history(hr["memory_id"], None, hr["new_memory"], "ADD", created_at=hr.get("created_at"))
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']}: {e}")

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in records]
            all_entities = extract_entities_batch(all_texts)

            # 7a: Global dedup — collect unique entities across all memories
            global_entities = {}  # normalized_key -> (entity_type, entity_text, set of memory_ids)
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text in entities:
                    key = self._normalize_entity_text(entity_text)
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Single batch embed for all unique entities
                try:
                    entity_embeddings = self.embedding_model.embed_batch(entity_texts, "add")
                except Exception:
                    # Fallback: embed individually, use None for failures
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(self.embedding_model.embed(t, "add"))
                        except Exception:
                            entity_embeddings.append(None)


                if len(entity_embeddings) != len(ordered_keys):
                    logger.warning(
                        "embed_batch returned %d vectors for %d entity texts — "
                        "padding/truncating to avoid dropping entity links",
                        len(entity_embeddings),
                        len(ordered_keys),
                    )
                    entity_embeddings = list(entity_embeddings[: len(ordered_keys)])
                    entity_embeddings += [None] * (len(ordered_keys) - len(entity_embeddings))

                # Filter out entities with failed embeddings
                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]
                    exact_matches = self._existing_entities_by_text(search_filters)

                    # 7c: Batch search for existing entities
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    existing_matches = self.entity_store.search_batch(
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    for j, key in enumerate(valid_keys):
                        entity_type, entity_text, memory_ids = global_entities[key]
                        matches = existing_matches[j] if j < len(existing_matches) else []
                        exact_match = exact_matches.get(key)

                        semantic_match = matches[0] if matches and matches[0].score >= 0.95 else None
                        match = exact_match or semantic_match
                        if match:
                            # Update existing entity
                            payload = match.payload or {}
                            linked = set(payload.get("linked_memory_ids", []))
                            linked |= memory_ids
                            payload["linked_memory_ids"] = sorted(linked)
                            try:
                                self.entity_store.update(
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}': {e}")
                        else:
                            # New entity — collect for batch insert
                            to_insert_vectors.append(valid_vectors[j])
                            to_insert_ids.append(str(uuid.uuid4()))
                            to_insert_payloads.append({
                                "data": entity_text,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **search_filters,
                            })

                    # 7e: Single batch insert for all new entities
                    if to_insert_vectors:
                        try:
                            self.entity_store.insert(
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        except Exception as e:
                            logger.warning(f"Batch entity insert failed: {e}")
        except Exception as e:
            logger.warning(f"Batch entity linking failed: {e}")

        # Phase 8: Save messages + return
        self.db.save_messages(messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            for r in records
        ]

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            display_first_run_notice(self, "sync", "get")
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        display_first_run_notice(self, "sync", "get")
        return result_item

    def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        show_expired: bool = False,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.
            show_expired (bool, optional): Include expired memories. Defaults to False.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        fetch_limit = limit if show_expired else max(limit * 4, 60)
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        all_memories_result = self._get_all_from_vector_store(effective_filters, fetch_limit, show_expired, limit)

        if scale_threshold_notice:
            display_scale_threshold_notice(self, "sync", "get_all", *scale_threshold_notice)
        else:
            display_first_run_notice(self, "sync", "get_all")
        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit, show_expired=False, output_limit=None):
        memories_result = self.vector_store.list(filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            if not show_expired and _payload_is_expired(mem.payload):
                continue
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)
            if output_limit is not None and len(formatted_memories) >= output_limit:
                break

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: bool = False,
        explain: bool = False,
        reference_date: Optional[Any] = None,
        show_expired: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): 要搜索的自然语言问题或关键词。
                它会同时用于语义向量检索和关键词检索，所以越像真实查询越好。
            top_k (int, optional): 最多返回多少条结果。
                既控制最终返回数量，也会影响内部召回池大小和性能提示判断。
            filters (dict): 过滤条件字典。
                至少要包含 `user_id` / `agent_id` / `run_id` 之一，用来限定搜索作用域。
                也可以附带额外的元数据过滤条件，例如 `{"user_id": "u1", "tag": "work"}`。

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): 最低得分阈值。
                低于这个分数的记忆会被过滤掉，默认 0.1。
            rerank (bool, optional): 是否启用二次排序。
                开启后会把初步召回结果再交给 reranker 精排一次。
            explain (bool, optional): 是否返回每条结果的打分细节。
                适合调试为什么某条记忆被召回。
            reference_date (Any, optional): 平台侧时间参考参数。
                OSS 版本不支持；如果传入会直接报错。
            show_expired (bool, optional): 是否把已过期记忆也返回。
                默认 False，开发调试时如果想看历史数据可以打开。

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`

        Raises:
            ValueError: 当 filters 缺少作用域 ID，或者 threshold/top_k 非法时抛出。
        """
        if reference_date is not None:
            raise ValueError(get_temporal_feature_error_message("sync", "search", "reference_date"))

        # 1) 顶层实体参数已废弃，统一要求通过 filters 传入，避免参数来源分裂。
        _reject_top_level_entity_params(kwargs, "search")

        # 2) 基础校验：先校验 top_k / threshold，再清洗 query 内容。
        _validate_search_params(threshold=threshold, top_k=top_k)
        query = _validate_and_trim_search_query(query)
        temporal_usage_notice = detect_temporal_usage_from_search(query, filters)

        # 3) 规范化 filters 中的实体 ID，并保证至少有一个作用域 ID。
        effective_filters = filters.copy() if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        # 4) 搜索默认返回 top_k 条，同时把 top_k 作为搜索性能提示的判断依据。
        limit = top_k
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        # 5) 如果 filters 使用了高级运算符（AND/OR/NOT、contains、gt 等），先转成向量库可识别格式。
        if self._has_advanced_operators(effective_filters):
            processed_filters = self._process_metadata_filters(effective_filters)
            # 这些逻辑键已经被转译过了，避免和转译后的结果重复生效。
            for logical_key in ("AND", "OR", "NOT"):
                effective_filters.pop(logical_key, None)
            for fk in list(effective_filters.keys()):
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    effective_filters.pop(fk, None)
            effective_filters.update(processed_filters)

        # 6) 埋点：记录本次搜索的过滤条件和参数，便于调试与产品分析。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
                "explain": explain,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        # 7) 真正的检索在 _search_vector_store 里完成：语义召回 + 关键词召回 + BM25 融合。
        search_start = time.perf_counter()
        original_memories = self._search_vector_store(
            query, effective_filters, limit, threshold, explain=explain, show_expired=show_expired
        )
        search_elapsed_seconds = time.perf_counter() - search_start

        # 8) 如果配置了 reranker，则对候选结果做二次排序，提升最终相关性。
        if rerank and self.reranker and original_memories:
            try:
                reranked_memories = self.reranker.rerank(query, original_memories, limit)
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        # 9) 根据本次搜索是否触发 temporal / 性能 / 首次使用提示，打印不同的 runtime notice。
        if temporal_usage_notice:
            display_temporal_usage_notice(self, "sync", "search", *temporal_usage_notice)
        elif scale_threshold_notice:
            display_scale_threshold_notice(self, "sync", "search", *scale_threshold_notice)
        elif search_elapsed_seconds > PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS:
            display_performance_slow_query_notice(
                self,
                "sync",
                "search",
                search_elapsed_seconds,
                top_k,
                len(original_memories),
            )
        else:
            display_first_run_notice(self, "sync", "search")
        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    target[key].update(value)
                else:
                    target[key] = value

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                merge_filters(processed_filters, process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.
        
        Args:
            filters: Dictionary of filters to check
            
        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False
            
        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _search_vector_store(self, query, filters, limit, threshold=0.1, explain=False, show_expired=False):
        """
        执行统一检索管线（同步版）：语义召回 + 关键词召回 + 实体增强 + 融合重排。

        该方法是 search() 的核心执行器，整体流程：
        1. 预处理 query（词形还原 + 实体抽取）
        2. 语义向量召回（过采样）
        3. 关键词 BM25 召回并分数归一化
        4. 基于实体库计算 entity boost
        5. 融合打分（semantic + bm25 + entity）并按阈值/limit 截断
        6. 输出标准 MemoryItem 结构

        Args:
            query (str): 用户查询文本。
            filters (dict): 作用域与元数据过滤条件。
            limit (int): 最终返回条数上限。
            threshold (float, optional): 融合后最低分阈值；None 时回退到 0.1。
            explain (bool, optional): 是否输出 score_details 解释字段。
            show_expired (bool, optional): 是否包含过期记忆。

        Returns:
            list[dict]: 标准化后的记忆结果列表。
        """
        # 兼容旧调用方：threshold 可能显式传 None。
        if threshold is None:
            threshold = 0.1

        # Step 1) Query 预处理
        # - lemmatize_for_bm25: 为关键词检索准备统一词形
        # - extract_entities: 提取实体，后续用于实体相关性加权
        query_lemmatized = lemmatize_for_bm25(query)
        query_entities = extract_entities(query)

        # Step 2) 计算 query 向量（semantic 搜索输入）
        embeddings = self.embedding_model.embed(query, "search")

        # Step 3) 语义召回（过采样）
        # internal_limit > limit：先多取候选，给后续融合排序留空间，避免早截断误杀。
        internal_limit = max(limit * 4, 60)
        semantic_results = self.vector_store.search(
            query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4) 关键词召回（走向量库的 keyword_search 能力，若实现支持）
        keyword_results = self.vector_store.keyword_search(
            query=query_lemmatized, top_k=internal_limit, filters=filters
        )

        # Step 5) BM25 分数标准化
        # 不同向量库/检索器原始分值尺度不同，统一归一化后再参与融合。
        bm25_scores = {}
        if keyword_results is not None:
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6) 实体增强分
        # 当 query 命中实体（人名/组织/地点等）时，优先提升与这些实体关联的记忆。
        entity_boosts = {}
        if query_entities:
            entity_boosts = self._compute_entity_boosts(query_entities, filters)

        # Step 7) 构造融合候选集（来自 semantic 召回）
        # show_expired=False 时先过滤掉已过期记忆，避免进入后续排序。
        candidates = []
        for mem in semantic_results:
            payload = mem.payload if hasattr(mem, 'payload') else {}
            if not show_expired and _payload_is_expired(payload):
                continue
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": payload,
            })

        # Step 8) 融合打分与重排
        # score_and_rank 内部会按权重组合 semantic / bm25 / entity 分数，并应用阈值和 top_k。
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
            explain=explain,
        )

        # Step 9) 输出格式标准化
        # - 核心字段提升到顶层（id/memory/hash/time/score 等）
        # - 其余 payload 字段收拢到 metadata，避免顶层字段污染
        # - explain=True 时附带 score_details，便于调参与排障
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}

            if not payload.get("data"):
                continue  # Skip candidates with no payload data

            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)
            if explain and "score_details" in scored:
                memory_item_dict["score_details"] = scored["score_details"]

            original_memories.append(memory_item_dict)

        return original_memories

    def _compute_entity_boosts(self, query_entities, filters):
        """Compute per-memory entity boosts from entity store search.

        For each extracted entity from the query:
        1. Embed the entity text
        2. Search the entity store (threshold >= 0.5)
        3. For each matched entity, boost its linked memories

        Returns:
            Dict mapping memory_id (str) -> max entity boost [0, 0.5].
        """
        # Deduplicate entities (max 8)
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = self._normalize_entity_text(entity_text)
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        try:
            entity_texts = [text for _, text in deduped]
            embeddings = self.embedding_model.embed_batch(entity_texts, "search")

            if len(embeddings) != len(entity_texts):
                logger.warning(
                    "embed_batch returned %d vectors for %d texts — skipping entity boost",
                    len(embeddings),
                    len(entity_texts),
                )
                return memory_boosts

            entity_store = self.entity_store

            def _search_entity(entity_text, embedding):
                return entity_store.search(
                    query=entity_text, vectors=embedding, top_k=500, filters=search_filters
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    pool.submit(_search_entity, text, emb): text
                    for text, emb in zip(entity_texts, embeddings)
                }

                for future in concurrent.futures.as_completed(futures):
                    try:
                        matches = future.result()
                    except Exception as e:
                        logger.warning("Entity boost search failed for one entity: %s", e)
                        continue

                    for match in matches:
                        similarity = match.score if hasattr(match, 'score') else 0.0
                        if similarity < 0.5:
                            continue

                        payload = match.payload if hasattr(match, 'payload') else {}
                        linked_memory_ids = payload.get("linked_memory_ids", [])
                        if not isinstance(linked_memory_ids, list):
                            continue

                        num_linked = max(len(linked_memory_ids), 1)
                        memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                        boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                        for memory_id in linked_memory_ids:
                            if memory_id:
                                memory_key = str(memory_id)
                                memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    def update(
        self,
        memory_id,
        data: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expiration_date: Any = _UNSET,
    ):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str, optional): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.
            expiration_date (Any, optional): Date in YYYY-MM-DD format, or None to clear it.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        if data is None and metadata is None and expiration_date is _UNSET:
            raise ValueError("At least one of data, metadata, or expiration_date must be provided.")

        update_metadata = deepcopy(metadata) if metadata is not None else None
        if expiration_date is not _UNSET:
            update_metadata = update_metadata or {}
            update_metadata["expiration_date"] = _normalize_expiration_date(expiration_date)

        existing_embeddings = {}
        if data is not None:
            existing_embeddings[data] = self.embedding_model.embed(data, "update")

        self._update_memory(memory_id, data, existing_embeddings, update_metadata)
        display_first_run_notice(self, "sync", "update")
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_memory = self.vector_store.get(vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        self._delete_memory(memory_id, existing_memory)
        decay_usage_notice = detect_decay_usage_from_delete()
        if decay_usage_notice:
            display_decay_usage_notice(self, "sync", "delete", *decay_usage_notice)
        else:
            display_first_run_notice(self, "sync", "delete")
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        user_id = _validate_and_trim_entity_id(user_id, "user_id")
        agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
        run_id = _validate_and_trim_entity_id(run_id, "run_id")

        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        # delete all vector memories and reset the collections
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        decay_usage_notice = detect_decay_usage_from_delete_all(len(memories))
        if decay_usage_notice:
            display_decay_usage_notice(self, "sync", "delete_all", *decay_usage_notice)
        else:
            display_first_run_notice(self, "sync", "delete_all")
        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        history = self.db.get_history(memory_id)
        display_first_run_notice(self, "sync", "history")
        return history

    def _create_memory(self, data, existing_embeddings, metadata=None):
        """
        创建一条新的记忆并写入向量库，同时记录历史变更。

        Args:
            data (str): 要写入记忆的文本内容。
            existing_embeddings (dict): 预先计算好的 embedding 缓存，key 为文本，value 为向量。
                命中缓存时可跳过重复 embedding 计算。
            metadata (dict, optional): 要附加到记忆 payload 的元数据。
                会在内部补齐 data/hash/created_at/updated_at/text_lemmatized 等字段。

        Returns:
            str: 新创建的 memory_id（UUID）。
        """
        logger.debug(f"Creating memory with {data=}")
        # 优先复用上游阶段已算好的向量，减少一次模型调用
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        # 预计算词形还原文本，供 BM25 检索使用
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            # Backing-store failure, not a bad memory_id: re-raise the original so the REST layer maps it to 5xx, not 4xx.
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")
        if data is None:
            data = prev_value
        if not isinstance(data, str):
            raise ValueError(f"Memory with id {memory_id} does not have text content to update")
        text_changed = data != prev_value

        new_metadata = deepcopy(existing_memory.payload)
        if metadata is not None:
            new_metadata.update(metadata)

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # actor_id is immutable after creation (issue #4490)
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        if text_changed:
            self._remove_memory_from_entity_store(memory_id, session_filters)
            self._link_entities_for_memory(memory_id, data, session_filters)

        return memory_id

    def _delete_memory(self, memory_id, existing_memory=None):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = self.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        # Entity-store cleanup: strip this memory's id from any entity records
        # that linked to it. Non-fatal — the helper swallows errors.
        self._remove_memory_from_entity_store(memory_id, session_filters)

        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        self.db.reset()
        self.db.close()
        self.db = SQLiteManager(self.config.history_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        # Reset entity store if initialized
        if self._entity_store is not None:
            try:
                self._entity_store.reset()
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            self._entity_store = None

        capture_event("mem0.reset", self, {"sync_type": "sync"})
        display_first_run_notice(self, "sync", "reset")

    def close(self):
        """Release resources held by this Memory instance (SQLite connections, etc.)."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions
        self._entity_store = None

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        if MEM0_TELEMETRY:
            telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
            telemetry_config.collection_name = "mem0migrations"
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config.path = os.path.join(mem0_dir, provider_path)
                os.makedirs(telemetry_config.path, exist_ok=True)
            self._telemetry_vector_store = VectorStoreFactory.create(self.config.vector_store.provider, telemetry_config)

        if getattr(type(self.vector_store), "keyword_search", None) is VectorStoreBase.keyword_search:
            logger.warning(
                "The '%s' vector store does not support keyword search. "
                "Hybrid (BM25) scoring will be disabled and search will use "
                "semantic similarity only. To enable hybrid search, switch to a "
                "store with keyword_search support (e.g. qdrant, elasticsearch, pgvector).",
                self.config.vector_store.provider,
            )

        capture_event("mem0.init", self, {"sync_type": "async"})

    @property
    def project(self):
        return _AsyncOSSProject()

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            entity_collection = _entity_collection_name(self.config.vector_store.provider, self.collection_name)
            if hasattr(entity_config, 'collection_name'):
                entity_config.collection_name = entity_collection
            elif isinstance(entity_config, dict):
                entity_config['collection_name'] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                if hasattr(entity_config, "client"):
                    entity_config.client = self.vector_store.client
                elif isinstance(entity_config, dict):
                    entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(
                self.config.vector_store.provider, entity_config
            )
        return self._entity_store

    @staticmethod
    def _normalize_entity_text(value: str) -> str:
        return " ".join(value.strip().lower().split())

    def _existing_entities_by_text(self, filters):
        """Return existing entity rows keyed by normalized payload data."""
        try:
            listed = self.entity_store.list(filters=filters, top_k=10000)
        except Exception as e:
            logger.debug(f"Exact entity lookup failed, falling back to semantic dedup: {e}")
            return {}

        rows_by_text = {}
        for row in _vector_store_list_rows(listed):
            payload = getattr(row, "payload", None) or {}
            text = payload.get("data")
            if not isinstance(text, str):
                continue
            normalized = self._normalize_entity_text(text)
            if normalized and normalized not in rows_by_text:
                rows_by_text[normalized] = row
        return rows_by_text

    async def _upsert_entity_async(self, entity_text, entity_type, memory_id, filters):
        """Async variant of `_upsert_entity` — per-entity search-then-update-or-insert."""
        try:
            entity_embedding = await asyncio.to_thread(self.embedding_model.embed, entity_text, "add")
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
            exact_match = (
                await asyncio.to_thread(self._existing_entities_by_text, search_filters)
            ).get(self._normalize_entity_text(entity_text))

            existing = []
            if exact_match is None:
                existing = await asyncio.to_thread(
                    self.entity_store.search,
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=1,
                    filters=search_filters,
                )

            semantic_match = existing[0] if existing and existing[0].score >= 0.95 else None
            match = exact_match or semantic_match
            if match:
                payload = match.payload or {}
                linked_ids = payload.get("linked_memory_ids", [])
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    await asyncio.to_thread(
                        self.entity_store.update,
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                entity_id = str(uuid.uuid4())
                entity_payload = {
                    "data": entity_text,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                await asyncio.to_thread(
                    self.entity_store.insert,
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}' (async): {e}")

    async def _bulk_clear_entity_store(self, filters):
        """Delete all entity records matching the given scope filters.

        Used by delete_all to avoid the race condition that occurs when
        concurrent _delete_memory coroutines each try to read-modify-write
        the same entity rows' linked_memory_ids lists.
        """
        if self._entity_store is None:
            return
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = await asyncio.to_thread(self.entity_store.list, filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    await asyncio.to_thread(self.entity_store.delete, vector_id=row.id)
                except Exception as e:
                    logger.debug(f"Bulk entity delete failed for id={row.id}: {e}")
        except Exception as e:
            logger.warning(f"Bulk entity store cleanup failed: {e}")

    async def _remove_memory_from_entity_store(self, memory_id, filters):
        """Async variant of `Memory._remove_memory_from_entity_store`."""
        if self._entity_store is None:
            return
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = await asyncio.to_thread(self.entity_store.list, filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        try:
                            await asyncio.to_thread(self.entity_store.delete, vector_id=row.id)
                        except Exception as e:
                            logger.debug(f"Entity delete failed for id={row.id} (async): {e}")
                    else:
                        entity_text = payload.get("data")
                        if not isinstance(entity_text, str) or not entity_text:
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup (async)")
                            continue
                        try:
                            vec = await asyncio.to_thread(self.embedding_model.embed, entity_text, "update")
                        except Exception as e:
                            logger.debug(f"Entity re-embed failed for '{entity_text}' (async): {e}")
                            continue
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            await asyncio.to_thread(
                                self.entity_store.update,
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id} (async): {e}")
                except Exception as e:
                    logger.debug(f"Entity cleanup error (async): {e}")
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id} (async): {e}")

    async def _link_entities_for_memory(self, memory_id, text, filters):
        """Async variant of `Memory._link_entities_for_memory`."""
        try:
            entities = await asyncio.to_thread(extract_entities, text)
            if not entities:
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = self._normalize_entity_text(entity_text)
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    await self._upsert_entity_async(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}' (async): {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id} (async): {e}")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        expiration_date: Optional[Any] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str | list[dict]): 要写入记忆的原始消息。
                字符串、单条 dict、列表三种形态都支持，内部会统一整理。
            user_id (str, optional): 用户作用域 ID。
                异步版与同步版语义一致，用于隔离不同用户的记忆。
            agent_id (str, optional): agent 作用域 ID。
                当记忆只属于某个 agent 时使用。
            run_id (str, optional): run / session 作用域 ID。
                适合把单次执行过程中的记忆单独保存。
            metadata (dict, optional): 附加元数据。
                可携带来源、标签、业务字段等信息。
            timestamp (Any, optional): 平台侧时间参数。
                OSS 不支持，传入会报错。
            expiration_date (Any, optional): 过期日期，格式通常是 `YYYY-MM-DD`。
                过期后默认不会在检索结果中出现。
            infer (bool, optional): 是否先让 LLM 判断再落库。
                True 走抽取式增量记忆流程，False 则直接把消息写入。
            memory_type (str, optional): 记忆类型。
                传 `procedural_memory` 时会走程序性记忆分支。
            prompt (str, optional): 自定义提示词。
                可用于改变抽取策略或注入额外上下文。
            llm (BaseChatModel, optional): 程序性记忆专用的 LLM。
                适合你传入 LangChain ChatModel，自定义 procedural memory 的生成模型。
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        if timestamp is not None:
            raise ValueError(await get_temporal_feature_error_message_async("async", "add", "timestamp"))

        # 1) 异步版同样先处理时间/过期字段，OSS 不支持的平台字段直接拦截。
        normalized_expiration_date = _normalize_expiration_date(expiration_date)
        temporal_usage_notice = detect_temporal_usage_from_metadata(metadata)
        # 2) 统一把 user/agent/run 作用域与 metadata 合并成有效过滤条件。
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )
        if normalized_expiration_date is not None:
            processed_metadata["expiration_date"] = normalized_expiration_date

        # 3) 只允许 procedural_memory 走这个显式分支，其他值要尽早拒绝。
        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        # 4) 消化消息输入形态，统一变成消息列表，后续逻辑才能复用。
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        # 5) agent + procedural_memory 直接生成程序性记忆，不进入通用抽取管线。
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            scale_threshold_notice = await asyncio.to_thread(detect_scale_threshold_from_add_result, self, results)
            if temporal_usage_notice:
                await display_temporal_usage_notice_async(self, "async", "add", *temporal_usage_notice)
            elif scale_threshold_notice:
                await display_scale_threshold_notice_async(self, "async", "add", *scale_threshold_notice)
            else:
                await display_first_run_notice_async(self, "async", "add")
            return results

        # 6) 多模态消息先转成统一结构，再进入向量化与 LLM 抽取流程。
        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        # 7) 统一走异步版增量抽取管线：LLM 抽取、批量 embedding、去重、写库。
        vector_store_result = await self._add_to_vector_store(messages, processed_metadata, effective_filters, infer, prompt=prompt)
        scale_threshold_notice = await asyncio.to_thread(detect_scale_threshold_from_add_result, self, vector_store_result)
        if temporal_usage_notice:
            await display_temporal_usage_notice_async(self, "async", "add", *temporal_usage_notice)
        elif scale_threshold_notice:
            await display_scale_threshold_notice_async(self, "async", "add", *scale_threshold_notice)
        else:
            await display_first_run_notice_async(self, "async", "add")
        return {"results": vector_store_result}

    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
        prompt: Optional[str] = None,
    ):
        """
        将输入消息转成可持久化的记忆，并写入向量库/历史库（异步版）。

        Args:
            messages (list): 原始对话消息列表（role/content/name 等字段）。
            metadata (dict): 基础元数据模板，会按消息或动作扩展后写入 payload。
            effective_filters (dict): 作用域过滤条件（如 user_id/agent_id/run_id）。
            infer (bool): 是否启用 LLM 推理抽取。False 时直接逐条写入消息文本。
            prompt (str, optional): 自定义抽取提示词，优先级高于实例级 custom_instructions。

        Returns:
            list: 记忆变更结果列表（ADD/UPDATE/DELETE 等动作）。
        """
        if not infer:
            # 非推理模式：逐条消息直接写记忆，保持和同步版一致的语义。
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format (async): {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = await asyncio.to_thread(self.embedding_model.embed, msg_content, "add")
                mem_id = await self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # === V3 PHASED BATCH PIPELINE (async) ===

        # Phase 0: 收集上下文
        session_scope = _build_session_scope(effective_filters)
        # 在线程池中读取最近 10 条历史上下文，避免阻塞事件循环
        last_messages = await asyncio.to_thread(self.db.get_last_messages, session_scope, 10)
        # 把结构化消息拼成统一文本，作为向量检索 query 和抽取提示词输入
        parsed_messages = parse_messages(messages)

        # Phase 1: 先召回已有记忆，作为 LLM 判断“新增/更新/删除”的参照。
        search_filters = {k: v for k, v in effective_filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        query_embedding = await asyncio.to_thread(self.embedding_model.embed, parsed_messages, "search")
        existing_results = await asyncio.to_thread(
            self.vector_store.search,
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # 把真实 UUID 映射成短数字 ID，减少 LLM 输出与真实数据结构耦合。
        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: 单次 LLM 抽取
        is_agent_scoped = bool(effective_filters.get("agent_id")) and not effective_filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX

        # 外部传入 prompt 优先，否则使用实例级 custom_instructions。
        custom_instr = prompt or self.custom_instructions

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            custom_instructions=custom_instr,
        )

        try:
            response = await asyncio.to_thread(
                self.llm.generate_response,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"LLM extraction failed (async): {e}")
            return []

        # 解析返回：优先按 JSON 读，失败则尝试从文本里再抓一次。
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        except Exception as e:
            logger.error(f"Error parsing extraction response (async): {e}")
            extracted_memories = []

        if not extracted_memories:
            # 没抽到任何记忆也要保存消息历史，避免上下文丢失。
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            return []

        # Phase 3: 批量 embedding
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            mem_embeddings_list = await asyncio.to_thread(self.embedding_model.embed_batch, mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        except Exception:
            # 批处理失败时逐条补救，保证最终结果尽量完整。
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = await asyncio.to_thread(self.embedding_model.embed, text, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed memory text (async): {e}")

        # Phase 4/5: 哈希去重 + 逐条组织最终写入记录。
        existing_hashes = set()
        for mem in existing_results:
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            if h:
                existing_hashes.add(h)

        records = []
        seen_hashes = set()
        for mem in extracted_memories:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue

            # 已存在或本批次重复的文本，直接跳过。
            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match, async): {text[:50]}")
                continue
            seen_hashes.add(mem_hash)

            # 组装最终元数据：原文、词形归一、hash、时间戳、归因信息。
            text_lemmatized = lemmatize_for_bm25(text)

            memory_id = str(uuid.uuid4())
            mem_metadata = deepcopy(metadata)
            mem_metadata["data"] = text
            mem_metadata["text_lemmatized"] = text_lemmatized
            mem_metadata["hash"] = mem_hash
            if "created_at" not in mem_metadata:
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            if mem.get("attributed_to"):
                mem_metadata["attributed_to"] = mem["attributed_to"]

            records.append((memory_id, text, embed_map[text], mem_metadata))

        if not records:
            await asyncio.to_thread(self.db.save_messages, messages, session_scope)
            return []

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]

        try:
            await asyncio.to_thread(
                self.vector_store.insert,
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        except Exception:
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                try:
                    await asyncio.to_thread(self.vector_store.insert, vectors=[vec], ids=[mid], payloads=[pay])
                except Exception as e:
                    logger.error(f"Failed to insert memory {mid} (async): {e}")

        # Batch history
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in records
        ]
        try:
            await asyncio.to_thread(self.db.batch_add_history, history_records)
        except Exception:
            for hr in history_records:
                try:
                    await asyncio.to_thread(
                        self.db.add_history, hr["memory_id"], None, hr["new_memory"], "ADD",
                        created_at=hr.get("created_at")
                    )
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']} (async): {e}")

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in records]
            all_entities = await asyncio.to_thread(extract_entities_batch, all_texts)

            # 7a: Global dedup
            global_entities = {}
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text in entities:
                    key = self._normalize_entity_text(entity_text)
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Batch embed entities
                try:
                    entity_embeddings = await asyncio.to_thread(self.embedding_model.embed_batch, entity_texts, "add")
                except Exception:
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(await asyncio.to_thread(self.embedding_model.embed, t, "add"))
                        except Exception:
                            entity_embeddings.append(None)

                if len(entity_embeddings) != len(ordered_keys):
                    logger.warning(
                        "embed_batch returned %d vectors for %d entity texts — "
                        "padding/truncating to avoid dropping entity links",
                        len(entity_embeddings),
                        len(ordered_keys),
                    )
                    entity_embeddings = list(entity_embeddings[: len(ordered_keys)])
                    entity_embeddings += [None] * (len(ordered_keys) - len(entity_embeddings))

                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]
                    exact_matches = await asyncio.to_thread(self._existing_entities_by_text, search_filters)

                    # 7c: Batch search for existing entities
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    existing_matches = await asyncio.to_thread(
                        self.entity_store.search_batch,
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    for j, key in enumerate(valid_keys):
                        entity_type, entity_text, memory_ids = global_entities[key]
                        matches = existing_matches[j] if j < len(existing_matches) else []
                        exact_match = exact_matches.get(key)

                        semantic_match = matches[0] if matches and matches[0].score >= 0.95 else None
                        match = exact_match or semantic_match
                        if match:
                            payload = match.payload or {}
                            linked = set(payload.get("linked_memory_ids", []))
                            linked |= memory_ids
                            payload["linked_memory_ids"] = sorted(linked)
                            try:
                                await asyncio.to_thread(
                                    self.entity_store.update,
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}' (async): {e}")
                        else:
                            to_insert_vectors.append(valid_vectors[j])
                            to_insert_ids.append(str(uuid.uuid4()))
                            to_insert_payloads.append({
                                "data": entity_text,
                                "entity_type": entity_type,
                                "linked_memory_ids": sorted(memory_ids),
                                **search_filters,
                            })

                    # 7e: Batch insert new entities
                    if to_insert_vectors:
                        try:
                            await asyncio.to_thread(
                                self.entity_store.insert,
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        except Exception as e:
                            logger.warning(f"Batch entity insert failed (async): {e}")
        except Exception as e:
            logger.warning(f"Batch entity linking failed (async): {e}")

        # Phase 8: Save messages + return
        await asyncio.to_thread(self.db.save_messages, messages, session_scope)

        returned_memories = [
            {"id": r[0], "memory": r[1], "event": "ADD"}
            for r in records
        ]

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        return returned_memories

    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "async"})
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if not memory:
            await display_first_run_notice_async(self, "async", "get")
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        await display_first_run_notice_async(self, "async", "get")
        return result_item

    async def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        show_expired: bool = False,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.
            show_expired (bool, optional): Include expired memories. Defaults to False.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        fetch_limit = limit if show_expired else max(limit * 4, 60)
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"}
        )

        all_memories_result = await self._get_all_from_vector_store(effective_filters, fetch_limit, show_expired, limit)

        if scale_threshold_notice:
            await display_scale_threshold_notice_async(self, "async", "get_all", *scale_threshold_notice)
        else:
            await display_first_run_notice_async(self, "async", "get_all")
        return {"results": all_memories_result}

    async def _get_all_from_vector_store(self, filters, limit, show_expired=False, output_limit=None):
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            if not show_expired and _payload_is_expired(mem.payload):
                continue
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)
            if output_limit is not None and len(formatted_memories) >= output_limit:
                break

        return formatted_memories

    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: bool = False,
        explain: bool = False,
        reference_date: Optional[Any] = None,
        show_expired: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): 搜索词。
                会被同时用于语义检索和关键词检索。
            top_k (int, optional): 最多返回多少条结果。
                影响最终结果数量和内部候选池大小。
            filters (dict): 过滤条件字典。
                必须包含 `user_id` / `agent_id` / `run_id` 之一。
                还可以叠加额外元数据条件，例如 `{"user_id": "u1", "tag": "work"}`。

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): 最低得分阈值。
                低于该值的候选结果会被过滤掉。
            rerank (bool, optional): 是否启用 reranker。
                开启后会用重排器对初筛结果再排序。
            explain (bool, optional): 是否返回打分解释。
                方便调试召回为什么命中。
            reference_date (Any, optional): 平台侧时间参数。
                OSS 版本不支持。
            show_expired (bool, optional): 是否包含过期记忆。
                默认 False。

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`

        Raises:
            ValueError: 当 filters 缺少作用域 ID，或者 threshold/top_k 非法时抛出。
        """
        if reference_date is not None:
            raise ValueError(
                await get_temporal_feature_error_message_async("async", "search", "reference_date")
            )

        # 1) 统一要求通过 filters 传实体参数，避免旧参数位和新参数位并存。
        _reject_top_level_entity_params(kwargs, "search")

        # 2) 搜索参数校验：先保证 top_k / threshold 合法，再清理 query。
        _validate_search_params(threshold=threshold, top_k=top_k)
        query = _validate_and_trim_search_query(query)
        temporal_usage_notice = detect_temporal_usage_from_search(query, filters)

        # 3) 规范化过滤器中的实体 ID，并要求至少存在一个作用域。
        effective_filters = filters.copy() if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(
                effective_filters["user_id"], "user_id"
            )
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(
                effective_filters["agent_id"], "agent_id"
            )
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(
                effective_filters["run_id"], "run_id"
            )

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. "
                "Example: filters={'user_id': 'u1'}"
            )

        # 4) top_k 既是返回条数，也是性能提示阈值的参考。
        limit = top_k
        scale_threshold_notice = detect_scale_threshold_from_top_k(top_k)

        # 5) 高级过滤器（AND/OR/NOT/比较运算符）先转译成向量库可识别格式。
        if self._has_advanced_operators(effective_filters):
            processed_filters = self._process_metadata_filters(effective_filters)
            # 已经转译过的逻辑键删掉，避免重复解释。
            for logical_key in ("AND", "OR", "NOT"):
                effective_filters.pop(logical_key, None)
            for fk in list(effective_filters.keys()):
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(effective_filters.get(fk), dict):
                    effective_filters.pop(fk, None)
            effective_filters.update(processed_filters)

        # 6) 记录一次搜索埋点，方便排查过滤条件、耗时和召回情况。
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
                "explain": explain,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        # 7) 异步搜索同样走“语义召回 + 关键词召回 + BM25 融合”的统一检索管线。
        search_start = time.perf_counter()
        original_memories = await self._search_vector_store(
            query, effective_filters, limit, threshold, explain=explain, show_expired=show_expired
        )
        search_elapsed_seconds = time.perf_counter() - search_start

        # 8) 可选 rerank：如果配置了重排器，则再做一次相关性排序。
        if rerank and self.reranker and original_memories:
            try:
                reranked_memories = await asyncio.to_thread(
                    self.reranker.rerank, query, original_memories, limit
                )
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        # 9) 按不同情况输出运行提示：temporal / 性能慢查询 / 首次运行。
        if temporal_usage_notice:
            await display_temporal_usage_notice_async(self, "async", "search", *temporal_usage_notice)
        elif scale_threshold_notice:
            await display_scale_threshold_notice_async(self, "async", "search", *scale_threshold_notice)
        elif search_elapsed_seconds > PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS:
            await display_performance_slow_query_notice_async(
                self,
                "async",
                "search",
                search_elapsed_seconds,
                top_k,
                len(original_memories),
            )
        else:
            await display_first_run_notice_async(self, "async", "search")
        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    target[key].update(value)
                else:
                    target[key] = value

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                merge_filters(processed_filters, process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    async def _search_vector_store(self, query, filters, limit, threshold=0.1, explain=False, show_expired=False):
        """
        执行统一检索管线（异步版）：语义召回 + 关键词召回 + 实体增强 + 融合重排。

        与同步版逻辑一致，区别在于：
        - CPU/IO 可能阻塞的步骤通过 asyncio.to_thread 下沉到线程池
        - 实体增强阶段使用异步并发（受 semaphore 限流）
        """
        if threshold is None:
            threshold = 0.1

        # Step 1) Query 预处理（CPU-bound，放线程池）
        query_lemmatized = await asyncio.to_thread(lemmatize_for_bm25, query)
        query_entities = await asyncio.to_thread(extract_entities, query)

        # Step 2) 计算 query 向量
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")

        # Step 3) 语义召回（过采样）
        internal_limit = max(limit * 4, 60)
        semantic_results = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, top_k=internal_limit, filters=filters
        )

        # Step 4) 关键词召回（若向量库实现支持）
        keyword_results = await asyncio.to_thread(
            self.vector_store.keyword_search, query=query_lemmatized, top_k=internal_limit, filters=filters
        )

        # Step 5) BM25 分数标准化
        bm25_scores = {}
        if keyword_results is not None:
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, 'id') else str(mem.get('id', ''))
                raw_score = mem.score if hasattr(mem, 'score') else mem.get('score', 0)
                if raw_score and raw_score > 0:
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # Step 6) 实体增强分（异步并发版本）
        entity_boosts = {}
        if query_entities:
            entity_boosts = await self._compute_entity_boosts_async(query_entities, filters)

        # Step 7) 构造候选并过滤过期数据
        candidates = []
        for mem in semantic_results:
            payload = mem.payload if hasattr(mem, 'payload') else {}
            if not show_expired and _payload_is_expired(payload):
                continue
            mem_id = str(mem.id)
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": payload,
            })

        # Step 8) 融合打分与重排
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
            explain=explain,
        )

        # Step 9) 标准化输出结构（与同步版保持一致）
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
            "attributed_to",
            "expiration_date",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", "attributed_to", *promoted_payload_keys}

        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}
            if not payload.get("data"):
                continue

            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)
            if explain and "score_details" in scored:
                memory_item_dict["score_details"] = scored["score_details"]

            original_memories.append(memory_item_dict)

        return original_memories

    async def _compute_entity_boosts_async(self, query_entities, filters):
        """Async version of entity boost computation."""
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = self._normalize_entity_text(entity_text)
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        try:
            entity_texts = [text for _, text in deduped]
            embeddings = await asyncio.to_thread(self.embedding_model.embed_batch, entity_texts, "search")

            if len(embeddings) != len(entity_texts):
                logger.warning(
                    "embed_batch returned %d vectors for %d texts — skipping entity boost",
                    len(embeddings),
                    len(entity_texts),
                )
                return memory_boosts

            sem = asyncio.Semaphore(4)

            async def _search_entity(entity_text, embedding):
                async with sem:
                    return await asyncio.to_thread(
                        self.entity_store.search,
                        query=entity_text,
                        vectors=embedding,
                        top_k=500,
                        filters=search_filters,
                    )

            results = await asyncio.gather(
                *(_search_entity(text, emb) for text, emb in zip(entity_texts, embeddings)),
                return_exceptions=True,
            )

            for matches in results:
                if isinstance(matches, BaseException):
                    logger.warning("Entity boost search failed for one entity: %s", matches)
                    continue

                for match in matches:
                    similarity = match.score if hasattr(match, 'score') else 0.0
                    if similarity < 0.5:
                        continue

                    payload = match.payload if hasattr(match, 'payload') else {}
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    if not isinstance(linked_memory_ids, list):
                        continue

                    num_linked = max(len(linked_memory_ids), 1)
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    for memory_id in linked_memory_ids:
                        if memory_id:
                            memory_key = str(memory_id)
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    async def update(
        self,
        memory_id,
        data: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expiration_date: Any = _UNSET,
    ):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            data (str, optional): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.
            expiration_date (Any, optional): Date in YYYY-MM-DD format, or None to clear it.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> await m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "async"})

        if data is None and metadata is None and expiration_date is _UNSET:
            raise ValueError("At least one of data, metadata, or expiration_date must be provided.")

        update_metadata = deepcopy(metadata) if metadata is not None else None
        if expiration_date is not _UNSET:
            update_metadata = update_metadata or {}
            update_metadata["expiration_date"] = _normalize_expiration_date(expiration_date)

        existing_embeddings = {}
        if data is not None:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
            existing_embeddings[data] = embeddings

        await self._update_memory(memory_id, data, existing_embeddings, update_metadata)
        await display_first_run_notice_async(self, "async", "update")
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "async"})

        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        await self._delete_memory(memory_id, existing_memory)
        decay_usage_notice = detect_decay_usage_from_delete()
        if decay_usage_notice:
            await display_decay_usage_notice_async(self, "async", "delete", *decay_usage_notice)
        else:
            await display_first_run_notice_async(self, "async", "delete")
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        user_id = _validate_and_trim_entity_id(user_id, "user_id")
        agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
        run_id = _validate_and_trim_entity_id(run_id, "run_id")

        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)

        delete_tasks = []
        for memory in memories[0]:
            delete_tasks.append(self._delete_memory(memory.id, skip_entity_cleanup=True))

        results = await asyncio.gather(*delete_tasks, return_exceptions=True)

        if self._entity_store is not None:
            await self._bulk_clear_entity_store(filters)

        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            logger.warning("Failed to delete %d out of %d memories", len(errors), len(results))
            for err in errors:
                logger.warning("Delete error: %s", err)

        logger.info(f"Deleted {len(results) - len(errors)} memories")

        decay_usage_notice = detect_decay_usage_from_delete_all(len(memories[0]))
        if decay_usage_notice:
            await display_decay_usage_notice_async(self, "async", "delete_all", *decay_usage_notice)
        else:
            await display_first_run_notice_async(self, "async", "delete_all")
        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "async"})
        history = await asyncio.to_thread(self.db.get_history, memory_id)
        await display_first_run_notice_async(self, "async", "history")
        return history

    async def _create_memory(self, data, existing_embeddings, metadata=None):
        """
        异步创建一条新的记忆并写入向量库，同时记录历史变更。

        Args:
            data (str): 要写入记忆的文本内容。
            existing_embeddings (dict): 预先计算好的 embedding 缓存，key 为文本，value 为向量。
                命中缓存时可跳过重复 embedding 计算。
            metadata (dict, optional): 要附加到记忆 payload 的元数据。
                会在内部补齐 data/hash/created_at/updated_at/text_lemmatized 等字段。

        Returns:
            str: 新创建的 memory_id（UUID）。
        """
        logger.debug(f"Creating memory with {data=}")
        # 优先复用缓存向量；未命中时在线程池中执行 embedding，避免阻塞事件循环
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        # 预计算词形还原文本，供 BM25 检索使用
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        try:
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        except Exception:
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            raise

        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            if llm is not None:
                parsed_messages = convert_to_messages(parsed_messages)
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = remove_code_blocks(response.content)
            else:
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
                procedural_memory = remove_code_blocks(procedural_memory)
        
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata = {**metadata, "memory_type": MemoryType.PROCEDURAL.value}
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            # Backing-store failure, not a bad memory_id: re-raise the original so the REST layer maps it to 5xx, not 4xx.
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")
        if data is None:
            data = prev_value
        if not isinstance(data, str):
            raise ValueError(f"Memory with id {memory_id} does not have text content to update")
        text_changed = data != prev_value

        new_metadata = deepcopy(existing_memory.payload)
        if metadata is not None:
            new_metadata.update(metadata)

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # actor_id is immutable after creation (issue #4490)
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        if text_changed:
            await self._remove_memory_from_entity_store(memory_id, session_filters)
            await self._link_entities_for_memory(memory_id, data, session_filters)

        return memory_id

    async def _delete_memory(self, memory_id, existing_memory=None, skip_entity_cleanup=False):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}

        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        if not skip_entity_cleanup:
            await self._remove_memory_from_entity_store(memory_id, session_filters)

        return memory_id

    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        await asyncio.to_thread(self.vector_store.delete_col)

        gc.collect()

        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            await asyncio.to_thread(self.vector_store.client.close)

        await asyncio.to_thread(self.db.reset)
        await asyncio.to_thread(self.db.close)
        self.db = SQLiteManager(self.config.history_db_path)

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )

        if self._entity_store is not None:
            try:
                await asyncio.to_thread(self._entity_store.reset)
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            self._entity_store = None

        capture_event("mem0.reset", self, {"sync_type": "async"})
        await display_first_run_notice_async(self, "async", "reset")

    def close(self):
        """Release resources held by this AsyncMemory instance."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    async def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
