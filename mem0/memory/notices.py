import asyncio
import json
import re
import sys
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from mem0.memory import telemetry as telemetry_module
from mem0.memory.setup import _load_config, _write_config


FLAG_KEY = "mem0-oss-notices"
NOTICE_ID = "first_run"
TEMPORAL_FEATURE_NOTICE_ID = "temporal_stub"
TEMPORAL_USAGE_NOTICE_ID = "temporal_usage"
DECAY_FEATURE_NOTICE_ID = "decay_stub"
DECAY_USAGE_NOTICE_ID = "decay_usage"
SCALE_THRESHOLD_NOTICE_ID = "scale_threshold"
PERFORMANCE_SLOW_QUERY_NOTICE_ID = "performance_slow_query"
NOTICE_EVENT = "mem0.notice_displayed"
DISPLAYED_VARIANT = "displayed"
HOLDOUT_VARIANT = "holdout"
STATE_SECTION = "notice_state"
STATE_KEY = "first_run"
TEMPORAL_USAGE_STATE_KEY = "temporal_usage"
TEMPORAL_USAGE_CAP = 10
TEMPORAL_USAGE_WINDOW = timedelta(days=7)
DECAY_USAGE_STATE_KEY = "decay_usage"
DECAY_USAGE_CAP = 10
DECAY_USAGE_WINDOW = timedelta(days=7)
DECAY_USAGE_DELETE_THRESHOLD = 5
SCALE_THRESHOLD_STATE_KEY = "scale_threshold"
SCALE_THRESHOLD_CAP = 10
SCALE_THRESHOLD_WINDOW = timedelta(days=7)
SCALE_MEMORY_COUNT_THRESHOLD = 2000
SCALE_MEMORY_COUNT_CHECK_INTERVAL = 100
SCALE_TOP_K_THRESHOLD = 50
PERFORMANCE_SLOW_QUERY_STATE_KEY = "performance_slow_query"
PERFORMANCE_SLOW_QUERY_CAP = 10
PERFORMANCE_SLOW_QUERY_WINDOW = timedelta(days=7)
PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS = 2.0
FEATURE_ERROR_CAP = 10
FEATURE_ERROR_WINDOW = timedelta(days=7)
TEMPORAL_FEATURE_ERROR_MESSAGES = {
    "timestamp": "The timestamp parameter is not supported by the OSS Memory SDK.",
    "reference_date": "The reference_date parameter is not supported by the OSS Memory SDK.",
}
DECAY_FEATURE_ERROR_MESSAGE = "The decay parameter is not supported by the OSS Memory SDK."

_ISO_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)
_RELATIVE_TIME_RE = re.compile(
    r"\b("
    r"today|yesterday|tomorrow|"
    r"last\s+(?:night|week|month|year)|"
    r"this\s+(?:week|month|year)|"
    r"next\s+(?:week|month|year)|"
    r"(?:past|last)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)|"
    r"(?:since|before|after|until)\s+(?:today|yesterday|tomorrow|\d{4}-\d{2}-\d{2}|last\s+(?:week|month|year))"
    r")\b",
    re.IGNORECASE,
)
_RANGE_OPERATORS = {"gt", "gte", "lt", "lte"}

_state_lock = threading.Lock()
_first_run_claimed_in_process = False
_decay_usage_successful_delete_count_in_process = 0
_temporal_usage_capacity_reached_in_process = False
_decay_usage_capacity_reached_in_process = False
_scale_threshold_capacity_reached_in_process = False
_performance_slow_query_capacity_reached_in_process = False
_feature_error_capacity_reached_in_process = set()
_scale_memory_count_adds_since_check = 0
_scale_memory_count_checked_in_process = False
_scale_memory_count_threshold_evaluated_in_process = False


def display_first_run_notice(memory_instance, sync_type: str, trigger_function: str) -> None:
    """Best-effort first-run notice check. Never raises or writes unless displayed."""
    if not telemetry_module.MEM0_TELEMETRY:
        return

    if not _claim_first_run_notice(trigger_function):
        return

    variant = None
    try:
        telemetry = telemetry_module._get_oss_telemetry()
        if telemetry is None or telemetry.posthog is None or not telemetry.user_id:
            return

        flags = telemetry.posthog.evaluate_flags(telemetry.user_id, flag_keys=[FLAG_KEY])
        variant = flags.get_flag(FLAG_KEY)
        _update_first_run_variant(variant)

        if variant in (None, False):
            return

        payload = _coerce_mapping(flags.get_flag_payload(FLAG_KEY))
        notices = payload.get("notices", {})
        notice_config = _coerce_mapping(notices.get(NOTICE_ID) if isinstance(notices, dict) else {})
        notice_config_found = bool(notice_config)

        copy = notice_config.get("copy")
        enabled = notice_config.get("enabled", True) if notice_config_found else False
        notice_type = notice_config.get("notice_type", "log_line")

        disabled_reason = None
        bypass_reason = None
        if not notice_config_found:
            bypass_reason = "missing_notice_config"
        elif not enabled:
            disabled_reason = "payload_disabled"
            bypass_reason = disabled_reason
        elif not copy:
            bypass_reason = "missing_copy"
        elif variant != DISPLAYED_VARIANT:
            bypass_reason = "holdout" if variant == HOLDOUT_VARIANT else "not_displayed"

        displayed = variant == DISPLAYED_VARIANT and enabled and bool(copy)

        telemetry.capture_event(
            NOTICE_EVENT,
            {
                "notice_id": NOTICE_ID,
                "notice_type": notice_type,
                "flag_key": FLAG_KEY,
                "variant": variant,
                "displayed": displayed,
                "payload": copy,
                "bypass_reason": bypass_reason,
                "disabled_reason": disabled_reason,
                "notice_config_found": notice_config_found,
                "sync_type": sync_type,
                "trigger_function": trigger_function,
            },
            flags=flags,
        )

        if displayed:
            print(copy, file=sys.stderr)
    except Exception:
        if variant is not None:
            _update_first_run_variant(variant)


async def display_first_run_notice_async(memory_instance, sync_type: str, trigger_function: str) -> None:
    if not telemetry_module.MEM0_TELEMETRY or _first_run_claimed_in_process:
        return
    await asyncio.to_thread(display_first_run_notice, memory_instance, sync_type, trigger_function)


def display_temporal_usage_notice(
    memory_instance,
    sync_type: str,
    trigger_function: str,
    trigger_source: str,
    trigger_reason: str,
) -> None:
    """
    按实验开关与配额策略，尝试展示“时间语义使用提示”（temporal usage notice）。

    该函数是 best-effort：
    - 任意异常都会被吞掉，不影响主业务流程；
    - 只有命中 displayed 变体且配置可展示时，才会向 stderr 打印提示文案；
    - 无论是否展示，都会在符合条件时上报一次机会事件（用于实验分析）。

    Args:
        memory_instance: 当前 Memory/AsyncMemory 实例（用于关联上下文）。
        sync_type (str): 调用类型（如 "sync"/"async"）。
        trigger_function (str): 触发函数名（如 "add"/"search"）。
        trigger_source (str): 触发来源（如 payload 中哪个信号触发）。
        trigger_reason (str): 触发原因（用于遥测分析）。
    """
    if not telemetry_module.MEM0_TELEMETRY:
        return

    # 达到展示容量上限后，本进程内不再尝试该提示。
    if _temporal_usage_at_capacity():
        return

    try:
        telemetry = telemetry_module._get_oss_telemetry()
        if telemetry is None or telemetry.posthog is None or not telemetry.user_id:
            return

        flags = telemetry.posthog.evaluate_flags(telemetry.user_id, flag_keys=[FLAG_KEY])
        variant = flags.get_flag(FLAG_KEY)
        if variant in (None, False):
            return

        # 从 flag payload 中提取 temporal notice 配置（文案、开关、展示类型等）。
        payload = _coerce_mapping(flags.get_flag_payload(FLAG_KEY))
        notices = payload.get("notices", {})
        notice_config = _coerce_mapping(
            notices.get(TEMPORAL_USAGE_NOTICE_ID) if isinstance(notices, dict) else {}
        )
        notice_config_found = bool(notice_config)

        copy = notice_config.get("copy")
        enabled = notice_config.get("enabled", True) if notice_config_found else False
        notice_type = notice_config.get("notice_type", "log_line")

        disabled_reason = None
        bypass_reason = None
        if not notice_config_found:
            bypass_reason = "missing_notice_config"
        elif not enabled:
            disabled_reason = "payload_disabled"
            bypass_reason = disabled_reason
        elif not copy:
            bypass_reason = "missing_copy"
        elif variant != DISPLAYED_VARIANT:
            bypass_reason = "holdout" if variant == HOLDOUT_VARIANT else "not_displayed"

        # 仅 displayed 变体 + 开启 + 有文案时才真正展示到终端。
        displayed = variant == DISPLAYED_VARIANT and enabled and bool(copy)

        if not _record_temporal_usage_opportunity(
            variant=variant,
            sync_type=sync_type,
            trigger_function=trigger_function,
            trigger_source=trigger_source,
            trigger_reason=trigger_reason,
        ):
            return

        telemetry.capture_event(
            NOTICE_EVENT,
            {
                "notice_id": TEMPORAL_USAGE_NOTICE_ID,
                "notice_type": notice_type,
                "flag_key": FLAG_KEY,
                "variant": variant,
                "displayed": displayed,
                "payload": copy,
                "bypass_reason": bypass_reason,
                "disabled_reason": disabled_reason,
                "notice_config_found": notice_config_found,
                "sync_type": sync_type,
                "trigger_function": trigger_function,
                "trigger_source": trigger_source,
                "trigger_reason": trigger_reason,
            },
            flags=flags,
        )

        if displayed:
            print(copy, file=sys.stderr)
    except Exception:
        return


async def display_temporal_usage_notice_async(
    memory_instance,
    sync_type: str,
    trigger_function: str,
    trigger_source: str,
    trigger_reason: str,
) -> None:
    await asyncio.to_thread(
        display_temporal_usage_notice,
        memory_instance,
        sync_type,
        trigger_function,
        trigger_source,
        trigger_reason,
    )


def detect_decay_usage_from_delete() -> Optional[Tuple[str, str, Optional[int], Optional[int]]]:
    if not telemetry_module.MEM0_TELEMETRY:
        return None

    global _decay_usage_successful_delete_count_in_process
    try:
        with _state_lock:
            if _decay_usage_capacity_reached_in_process:
                return None
            _decay_usage_successful_delete_count_in_process += 1
            delete_count = _decay_usage_successful_delete_count_in_process

        if delete_count >= DECAY_USAGE_DELETE_THRESHOLD and not _decay_usage_at_capacity():
            return ("delete_count", "repeated_deletes", delete_count, None)
    except Exception:
        return None

    return None


def detect_decay_usage_from_delete_all(deleted_count: Any) -> Optional[Tuple[str, str, Optional[int], Optional[int]]]:
    if not telemetry_module.MEM0_TELEMETRY:
        return None

    deleted_count_value = _coerce_nonnegative_int(deleted_count, 0)
    if deleted_count_value <= 0:
        return None

    return ("delete_all", "bulk_delete", None, deleted_count_value)


def display_decay_usage_notice(
    memory_instance,
    sync_type: str,
    trigger_function: str,
    trigger_source: str,
    trigger_reason: str,
    delete_count: Optional[int] = None,
    deleted_count: Optional[int] = None,
) -> None:
    """Best-effort decay usage notice. Never raises or writes unless displayed."""
    if not telemetry_module.MEM0_TELEMETRY:
        return

    if _decay_usage_at_capacity():
        return

    try:
        telemetry = telemetry_module._get_oss_telemetry()
        if telemetry is None or telemetry.posthog is None or not telemetry.user_id:
            return

        flags = telemetry.posthog.evaluate_flags(telemetry.user_id, flag_keys=[FLAG_KEY])
        variant = flags.get_flag(FLAG_KEY)
        if variant in (None, False):
            return

        payload = _coerce_mapping(flags.get_flag_payload(FLAG_KEY))
        notices = payload.get("notices", {})
        notice_config = _coerce_mapping(
            notices.get(DECAY_USAGE_NOTICE_ID) if isinstance(notices, dict) else {}
        )
        notice_config_found = bool(notice_config)

        copy = notice_config.get("copy")
        enabled = notice_config.get("enabled", True) if notice_config_found else False
        notice_type = notice_config.get("notice_type", "log_line")

        disabled_reason = None
        bypass_reason = None
        if not notice_config_found:
            bypass_reason = "missing_notice_config"
        elif not enabled:
            disabled_reason = "payload_disabled"
            bypass_reason = disabled_reason
        elif not copy:
            bypass_reason = "missing_copy"
        elif variant != DISPLAYED_VARIANT:
            bypass_reason = "holdout" if variant == HOLDOUT_VARIANT else "not_displayed"

        displayed = variant == DISPLAYED_VARIANT and enabled and bool(copy)

        if not _record_decay_usage_opportunity(
            variant=variant,
            sync_type=sync_type,
            trigger_function=trigger_function,
            trigger_source=trigger_source,
            trigger_reason=trigger_reason,
            delete_count=delete_count,
            deleted_count=deleted_count,
        ):
            return

        telemetry.capture_event(
            NOTICE_EVENT,
            {
                "notice_id": DECAY_USAGE_NOTICE_ID,
                "notice_type": notice_type,
                "flag_key": FLAG_KEY,
                "variant": variant,
                "displayed": displayed,
                "payload": copy,
                "bypass_reason": bypass_reason,
                "disabled_reason": disabled_reason,
                "notice_config_found": notice_config_found,
                "sync_type": sync_type,
                "trigger_function": trigger_function,
                "trigger_source": trigger_source,
                "trigger_reason": trigger_reason,
                "delete_count": delete_count,
                "deleted_count": deleted_count,
            },
            flags=flags,
        )

        if displayed:
            print(copy, file=sys.stderr)
    except Exception:
        return


async def display_decay_usage_notice_async(
    memory_instance,
    sync_type: str,
    trigger_function: str,
    trigger_source: str,
    trigger_reason: str,
    delete_count: Optional[int] = None,
    deleted_count: Optional[int] = None,
) -> None:
    await asyncio.to_thread(
        display_decay_usage_notice,
        memory_instance,
        sync_type,
        trigger_function,
        trigger_source,
        trigger_reason,
        delete_count,
        deleted_count,
    )


def detect_scale_threshold_from_top_k(top_k: Any) -> Optional[Tuple[str, str, Optional[int], Optional[int], int]]:
    try:
        top_k_value = int(top_k)
    except (TypeError, ValueError):
        return None

    if top_k_value < SCALE_TOP_K_THRESHOLD:
        return None

    return ("top_k", "high_top_k", top_k_value, None, SCALE_TOP_K_THRESHOLD)


def detect_scale_threshold_from_add_result(
    memory_instance,
    add_result: Any,
) -> Optional[Tuple[str, str, Optional[int], Optional[int], int]]:
    """
    基于 add() 返回结果判断是否触发“规模阈值”提示（memory_count 维度）。

    触发条件（需全部满足）：
    1. Telemetry 已开启；
    2. 本次 add_result 中至少有 1 条 ADD 事件；
    3. 达到检查时机（首次，或累计新增达到检查间隔）；
    4. 当前进程和持久化状态都尚未判定过“阈值已评估”；
    5. 底层向量库的记忆总量 >= SCALE_MEMORY_COUNT_THRESHOLD；
    6. 成功写入“已评估”状态，避免重复提示。

    Returns:
        Optional[Tuple[str, str, Optional[int], Optional[int], int]]:
            命中时返回通知元组：
            ("memory_count", "memory_count_threshold", None, provider_count, threshold)
            未命中返回 None。
    """
    if not telemetry_module.MEM0_TELEMETRY:
        return None

    # 只统计真正新增的记忆数量；0 说明本次没有新增，不需要做规模阈值判断。
    added_count = _count_added_memories(add_result)
    if added_count == 0:
        return None

    global _scale_memory_count_adds_since_check
    global _scale_memory_count_checked_in_process
    global _scale_memory_count_threshold_evaluated_in_process
    try:
        with _state_lock:
            # 进程内已评估过阈值，直接跳过后续检查。
            if _scale_memory_count_threshold_evaluated_in_process:
                return None

            _scale_memory_count_adds_since_check += added_count
            # 控制检查频率：首次必查；之后按累计新增量到达间隔再查，减少昂贵的 provider 计数调用。
            should_check = (
                not _scale_memory_count_checked_in_process
                or _scale_memory_count_adds_since_check >= SCALE_MEMORY_COUNT_CHECK_INTERVAL
            )
            if not should_check:
                return None

            _scale_memory_count_checked_in_process = True
            _scale_memory_count_adds_since_check = 0

            config = _load_config()
            scale_state = _get_notice_state(config, SCALE_THRESHOLD_STATE_KEY)
            # 持久化状态已记录“阈值评估完成”，则本进程同步标记，避免重复检查。
            if scale_state.get("memory_count_threshold_evaluated"):
                _scale_memory_count_threshold_evaluated_in_process = True
                return None
    except Exception:
        return None

    # 走到这里才执行一次真实 provider 记忆数统计。
    provider_count = _get_provider_memory_count(memory_instance)
    if provider_count is None or provider_count < SCALE_MEMORY_COUNT_THRESHOLD:
        return None

    # 原子标记“已评估”；失败则不触发提示，避免并发下重复展示。
    if not _mark_scale_memory_count_threshold_evaluated():
        return None

    return (
        "memory_count",
        "memory_count_threshold",
        None,
        provider_count,
        SCALE_MEMORY_COUNT_THRESHOLD,
    )


def display_scale_threshold_notice(
    memory_instance,
    sync_type: str,
    trigger_function: str,
    trigger_source: str,
    trigger_reason: str,
    top_k: Optional[int] = None,
    memory_count: Optional[int] = None,
    threshold: Optional[int] = None,
) -> None:
    """
    按实验开关与配额策略，尝试展示“规模阈值提示”（scale threshold notice）。

    该提示通常由两类信号触发：
    - top_k 过大（高召回量可能影响性能/成本）
    - memory_count 达到阈值（数据规模进入新阶段）

    行为同样是 best-effort：
    - 任意异常不向上抛出；
    - 只有 displayed 变体且文案可用时才打印到 stderr；
    - 满足条件时会上报遥测事件，用于分析展示/旁路原因。

    Args:
        memory_instance: 当前 Memory/AsyncMemory 实例（用于关联上下文）。
        sync_type (str): 调用类型（如 "sync"/"async"）。
        trigger_function (str): 触发函数名（如 "add"/"search"）。
        trigger_source (str): 触发来源（如 "top_k" 或 "memory_count"）。
        trigger_reason (str): 触发原因（用于遥测分析）。
        top_k (Optional[int]): 触发时的 top_k 值（若来源为 top_k）。
        memory_count (Optional[int]): 触发时的总记忆数（若来源为 memory_count）。
        threshold (Optional[int]): 对应触发阈值。
    """
    if not telemetry_module.MEM0_TELEMETRY:
        return

    # 达到展示容量上限后，本进程内不再尝试该提示。
    if _scale_threshold_at_capacity():
        return

    try:
        telemetry = telemetry_module._get_oss_telemetry()
        if telemetry is None or telemetry.posthog is None or not telemetry.user_id:
            return

        flags = telemetry.posthog.evaluate_flags(telemetry.user_id, flag_keys=[FLAG_KEY])
        variant = flags.get_flag(FLAG_KEY)
        if variant in (None, False):
            return

        # 读取 scale notice 配置，并按触发来源选择对应文案模板。
        payload = _coerce_mapping(flags.get_flag_payload(FLAG_KEY))
        notices = payload.get("notices", {})
        notice_config = _coerce_mapping(
            notices.get(SCALE_THRESHOLD_NOTICE_ID) if isinstance(notices, dict) else {}
        )
        notice_config_found = bool(notice_config)

        copies = _coerce_mapping(notice_config.get("copies")) if notice_config_found else {}
        copy_key = "memory_count" if trigger_source == "memory_count" else "top_k"
        copy = _render_scale_copy(copies.get(copy_key), top_k=top_k, memory_count=memory_count)
        enabled = notice_config.get("enabled", True) if notice_config_found else False
        notice_type = notice_config.get("notice_type", "log_line")

        disabled_reason = None
        bypass_reason = None
        if not notice_config_found:
            bypass_reason = "missing_notice_config"
        elif not enabled:
            disabled_reason = "payload_disabled"
            bypass_reason = disabled_reason
        elif not copy:
            bypass_reason = "missing_copy"
        elif variant != DISPLAYED_VARIANT:
            bypass_reason = "holdout" if variant == HOLDOUT_VARIANT else "not_displayed"

        # 仅 displayed 变体 + 开启 + 有文案时才真正展示到终端。
        displayed = variant == DISPLAYED_VARIANT and enabled and bool(copy)

        if not _record_scale_threshold_opportunity(
            variant=variant,
            sync_type=sync_type,
            trigger_function=trigger_function,
            trigger_source=trigger_source,
            trigger_reason=trigger_reason,
            top_k=top_k,
            memory_count=memory_count,
            threshold=threshold,
        ):
            return

        telemetry.capture_event(
            NOTICE_EVENT,
            {
                "notice_id": SCALE_THRESHOLD_NOTICE_ID,
                "notice_type": notice_type,
                "flag_key": FLAG_KEY,
                "variant": variant,
                "displayed": displayed,
                "payload": copy,
                "bypass_reason": bypass_reason,
                "disabled_reason": disabled_reason,
                "notice_config_found": notice_config_found,
                "sync_type": sync_type,
                "trigger_function": trigger_function,
                "trigger_source": trigger_source,
                "trigger_reason": trigger_reason,
                "top_k": top_k,
                "memory_count": memory_count,
                "threshold": threshold,
            },
            flags=flags,
        )

        if displayed:
            print(copy, file=sys.stderr)
    except Exception:
        return


async def display_scale_threshold_notice_async(
    memory_instance,
    sync_type: str,
    trigger_function: str,
    trigger_source: str,
    trigger_reason: str,
    top_k: Optional[int] = None,
    memory_count: Optional[int] = None,
    threshold: Optional[int] = None,
) -> None:
    await asyncio.to_thread(
        display_scale_threshold_notice,
        memory_instance,
        sync_type,
        trigger_function,
        trigger_source,
        trigger_reason,
        top_k,
        memory_count,
        threshold,
    )


def display_performance_slow_query_notice(
    memory_instance,
    sync_type: str,
    trigger_function: str,
    elapsed_seconds: float,
    top_k: int,
    result_count: int,
) -> None:
    """Best-effort slow-query notice. Never raises or writes unless displayed."""
    if not telemetry_module.MEM0_TELEMETRY:
        return

    if _performance_slow_query_at_capacity():
        return

    try:
        telemetry = telemetry_module._get_oss_telemetry()
        if telemetry is None or telemetry.posthog is None or not telemetry.user_id:
            return

        flags = telemetry.posthog.evaluate_flags(telemetry.user_id, flag_keys=[FLAG_KEY])
        variant = flags.get_flag(FLAG_KEY)
        if variant in (None, False):
            return

        payload = _coerce_mapping(flags.get_flag_payload(FLAG_KEY))
        notices = payload.get("notices", {})
        notice_config = _coerce_mapping(
            notices.get(PERFORMANCE_SLOW_QUERY_NOTICE_ID) if isinstance(notices, dict) else {}
        )
        notice_config_found = bool(notice_config)

        copy = notice_config.get("copy")
        enabled = notice_config.get("enabled", True) if notice_config_found else False
        notice_type = notice_config.get("notice_type", "log_line")

        disabled_reason = None
        bypass_reason = None
        if not notice_config_found:
            bypass_reason = "missing_notice_config"
        elif not enabled:
            disabled_reason = "payload_disabled"
            bypass_reason = disabled_reason
        elif not copy:
            bypass_reason = "missing_copy"
        elif variant != DISPLAYED_VARIANT:
            bypass_reason = "holdout" if variant == HOLDOUT_VARIANT else "not_displayed"

        displayed = variant == DISPLAYED_VARIANT and enabled and bool(copy)
        trigger_reason = "slow_query"

        if not _record_performance_slow_query_opportunity(
            variant=variant,
            sync_type=sync_type,
            trigger_function=trigger_function,
            trigger_reason=trigger_reason,
        ):
            return

        telemetry.capture_event(
            NOTICE_EVENT,
            {
                "notice_id": PERFORMANCE_SLOW_QUERY_NOTICE_ID,
                "notice_type": notice_type,
                "flag_key": FLAG_KEY,
                "variant": variant,
                "displayed": displayed,
                "payload": copy,
                "bypass_reason": bypass_reason,
                "disabled_reason": disabled_reason,
                "notice_config_found": notice_config_found,
                "sync_type": sync_type,
                "trigger_function": trigger_function,
                "trigger_reason": trigger_reason,
                "elapsed_ms": round(elapsed_seconds * 1000),
                "threshold_ms": round(PERFORMANCE_SLOW_QUERY_THRESHOLD_SECONDS * 1000),
                "top_k": top_k,
                "result_count": result_count,
            },
            flags=flags,
        )

        if displayed:
            print(copy, file=sys.stderr)
    except Exception:
        return


async def display_performance_slow_query_notice_async(
    memory_instance,
    sync_type: str,
    trigger_function: str,
    elapsed_seconds: float,
    top_k: int,
    result_count: int,
) -> None:
    await asyncio.to_thread(
        display_performance_slow_query_notice,
        memory_instance,
        sync_type,
        trigger_function,
        elapsed_seconds,
        top_k,
        result_count,
    )


def get_temporal_feature_error_message(sync_type: str, trigger_function: str, trigger_parameter: str) -> str:
    """Return the temporal feature error copy and capture event when available."""
    return _get_feature_error_message(
        TEMPORAL_FEATURE_NOTICE_ID,
        TEMPORAL_FEATURE_ERROR_MESSAGES[trigger_parameter],
        sync_type,
        trigger_function,
        trigger_parameter,
    )


async def get_temporal_feature_error_message_async(
    sync_type: str,
    trigger_function: str,
    trigger_parameter: str,
) -> str:
    return await asyncio.to_thread(
        get_temporal_feature_error_message,
        sync_type,
        trigger_function,
        trigger_parameter,
    )


def get_decay_feature_error_message(sync_type: str, trigger_function: str, trigger_parameter: str) -> str:
    """Return the decay feature error copy and capture event when available."""
    return _get_feature_error_message(
        DECAY_FEATURE_NOTICE_ID,
        DECAY_FEATURE_ERROR_MESSAGE,
        sync_type,
        trigger_function,
        trigger_parameter,
    )


async def get_decay_feature_error_message_async(
    sync_type: str,
    trigger_function: str,
    trigger_parameter: str,
) -> str:
    return await asyncio.to_thread(
        get_decay_feature_error_message,
        sync_type,
        trigger_function,
        trigger_parameter,
    )


def _get_feature_error_message(
    notice_id: str,
    plain_error: str,
    sync_type: str,
    trigger_function: str,
    trigger_parameter: str,
) -> str:
    if not telemetry_module.MEM0_TELEMETRY:
        return plain_error

    if _feature_error_at_capacity(notice_id):
        return plain_error

    try:
        telemetry = telemetry_module._get_oss_telemetry()
        if telemetry is None or telemetry.posthog is None or not telemetry.user_id:
            return plain_error

        flags = telemetry.posthog.evaluate_flags(telemetry.user_id, flag_keys=[FLAG_KEY])
        variant = flags.get_flag(FLAG_KEY)
        if variant in (None, False):
            return plain_error

        payload = _coerce_mapping(flags.get_flag_payload(FLAG_KEY))
        notices = payload.get("notices", {})
        notice_config = _coerce_mapping(
            notices.get(notice_id) if isinstance(notices, dict) else {}
        )
        notice_config_found = bool(notice_config)

        copy = notice_config.get("copy")
        enabled = notice_config.get("enabled", True) if notice_config_found else False
        notice_type = notice_config.get("notice_type", "error")

        disabled_reason = None
        bypass_reason = None
        if not notice_config_found:
            bypass_reason = "missing_notice_config"
        elif not enabled:
            disabled_reason = "payload_disabled"
            bypass_reason = disabled_reason
        elif not copy:
            bypass_reason = "missing_copy"
        elif variant not in (DISPLAYED_VARIANT, HOLDOUT_VARIANT):
            bypass_reason = "not_displayed"

        displayed = variant in (DISPLAYED_VARIANT, HOLDOUT_VARIANT) and enabled and bool(copy)

        if not _record_feature_error_opportunity(
            notice_id=notice_id,
            variant=variant,
            sync_type=sync_type,
            trigger_function=trigger_function,
            trigger_parameter=trigger_parameter,
        ):
            return plain_error

        telemetry.capture_event(
            NOTICE_EVENT,
            {
                "notice_id": notice_id,
                "notice_type": notice_type,
                "flag_key": FLAG_KEY,
                "variant": variant,
                "displayed": displayed,
                "payload": copy,
                "bypass_reason": bypass_reason,
                "disabled_reason": disabled_reason,
                "notice_config_found": notice_config_found,
                "sync_type": sync_type,
                "trigger_function": trigger_function,
                "trigger_parameter": trigger_parameter,
            },
            flags=flags,
        )

        if displayed:
            return copy
    except Exception:
        return plain_error

    return plain_error


def detect_temporal_usage_from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    try:
        if not isinstance(metadata, dict):
            return None

        for key, value in _walk_mapping(metadata):
            temporal_key = _is_temporal_key(key)
            if temporal_key and _looks_temporal_value(value, allow_epoch=True):
                return ("metadata", "date_like_metadata")
    except Exception:
        return None
    return None


def detect_temporal_usage_from_search(
    query: Any,
    filters: Optional[Dict[str, Any]],
) -> Optional[Tuple[str, str]]:
    try:
        if isinstance(query, str):
            if _RELATIVE_TIME_RE.search(query):
                return ("query", "relative_phrase")
            if _ISO_DATE_RE.search(query):
                return ("query", "date_like_query")

        if _has_temporal_filter(filters):
            return ("filter", "date_range_filter")
    except Exception:
        return None
    return None


def _claim_first_run_notice(trigger_function: str) -> bool:
    global _first_run_claimed_in_process

    with _state_lock:
        if _first_run_claimed_in_process:
            return False

        config = _load_config()
        state = config.get(STATE_SECTION)
        if isinstance(state, dict):
            first_run = state.get(STATE_KEY)
            if isinstance(first_run, dict) and first_run.get("consumed"):
                _first_run_claimed_in_process = True
                return False

        if not isinstance(state, dict):
            state = {}

        state[STATE_KEY] = {
            "consumed": True,
            "consumed_at": datetime.now(timezone.utc).isoformat(),
            "trigger_function": trigger_function,
            "variant": None,
        }
        config[STATE_SECTION] = state
        _write_config(config)
        _first_run_claimed_in_process = True
        return True


def _update_first_run_variant(variant) -> None:
    try:
        with _state_lock:
            config = _load_config()
            state = config.get(STATE_SECTION)
            if not isinstance(state, dict):
                state = {}
            first_run = state.get(STATE_KEY)
            if not isinstance(first_run, dict):
                first_run = {"consumed": True}
            first_run["variant"] = variant
            state[STATE_KEY] = first_run
            config[STATE_SECTION] = state
            _write_config(config)
    except Exception:
        return


def _feature_error_at_capacity(notice_id: str) -> bool:
    if notice_id in _feature_error_capacity_reached_in_process:
        return True

    try:
        with _state_lock:
            config = _load_config()
            entries = _recent_feature_error_entries(config, notice_id, datetime.now(timezone.utc))
            at_capacity = len(entries) >= FEATURE_ERROR_CAP
            if at_capacity:
                _feature_error_capacity_reached_in_process.add(notice_id)
            return at_capacity
    except Exception:
        return True


def _record_feature_error_opportunity(
    *,
    notice_id: str,
    variant: str,
    sync_type: str,
    trigger_function: str,
    trigger_parameter: str,
) -> bool:
    try:
        with _state_lock:
            now = datetime.now(timezone.utc)
            config = _load_config()
            entries = _recent_feature_error_entries(config, notice_id, now)
            if len(entries) >= FEATURE_ERROR_CAP:
                _feature_error_capacity_reached_in_process.add(notice_id)
                return False

            entries.append(
                {
                    "evaluated_at": now.isoformat(),
                    "variant": variant,
                    "sync_type": sync_type,
                    "trigger_function": trigger_function,
                    "trigger_parameter": trigger_parameter,
                }
            )

            state = config.get(STATE_SECTION)
            if not isinstance(state, dict):
                state = {}
            feature_state = state.get(notice_id)
            if not isinstance(feature_state, dict):
                feature_state = {}
            feature_state["events"] = entries
            state[notice_id] = feature_state
            config[STATE_SECTION] = state
            _write_config(config)
            if len(entries) >= FEATURE_ERROR_CAP:
                _feature_error_capacity_reached_in_process.add(notice_id)
            return True
    except Exception:
        return False


def _recent_feature_error_entries(config: Dict[str, Any], notice_id: str, now: datetime):
    state = config.get(STATE_SECTION)
    if not isinstance(state, dict):
        return []

    feature_state = state.get(notice_id)
    if not isinstance(feature_state, dict):
        return []

    entries = feature_state.get("events")
    if not isinstance(entries, list):
        return []

    cutoff = now - FEATURE_ERROR_WINDOW
    recent = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        evaluated_at = _parse_datetime(entry.get("evaluated_at"))
        if evaluated_at is not None and evaluated_at >= cutoff:
            recent.append(entry)
    return recent


def _temporal_usage_at_capacity() -> bool:
    global _temporal_usage_capacity_reached_in_process
    if _temporal_usage_capacity_reached_in_process:
        return True

    try:
        with _state_lock:
            config = _load_config()
            entries = _recent_temporal_usage_entries(config, datetime.now(timezone.utc))
            at_capacity = len(entries) >= TEMPORAL_USAGE_CAP
            if at_capacity:
                _temporal_usage_capacity_reached_in_process = True
            return at_capacity
    except Exception:
        return True


def _record_temporal_usage_opportunity(
    *,
    variant: str,
    sync_type: str,
    trigger_function: str,
    trigger_source: str,
    trigger_reason: str,
) -> bool:
    global _temporal_usage_capacity_reached_in_process
    try:
        with _state_lock:
            now = datetime.now(timezone.utc)
            config = _load_config()
            entries = _recent_temporal_usage_entries(config, now)
            if len(entries) >= TEMPORAL_USAGE_CAP:
                _temporal_usage_capacity_reached_in_process = True
                return False

            entries.append(
                {
                    "evaluated_at": now.isoformat(),
                    "variant": variant,
                    "sync_type": sync_type,
                    "trigger_function": trigger_function,
                    "trigger_source": trigger_source,
                    "trigger_reason": trigger_reason,
                }
            )

            state = config.get(STATE_SECTION)
            if not isinstance(state, dict):
                state = {}
            temporal_state = state.get(TEMPORAL_USAGE_STATE_KEY)
            if not isinstance(temporal_state, dict):
                temporal_state = {}
            temporal_state["events"] = entries
            state[TEMPORAL_USAGE_STATE_KEY] = temporal_state
            config[STATE_SECTION] = state
            _write_config(config)
            if len(entries) >= TEMPORAL_USAGE_CAP:
                _temporal_usage_capacity_reached_in_process = True
            return True
    except Exception:
        return False


def _recent_temporal_usage_entries(config: Dict[str, Any], now: datetime):
    state = config.get(STATE_SECTION)
    if not isinstance(state, dict):
        return []

    temporal_state = state.get(TEMPORAL_USAGE_STATE_KEY)
    if not isinstance(temporal_state, dict):
        return []

    entries = temporal_state.get("events")
    if not isinstance(entries, list):
        return []

    cutoff = now - TEMPORAL_USAGE_WINDOW
    recent = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        evaluated_at = _parse_datetime(entry.get("evaluated_at"))
        if evaluated_at is not None and evaluated_at >= cutoff:
            recent.append(entry)
    return recent


def _decay_usage_at_capacity() -> bool:
    global _decay_usage_capacity_reached_in_process
    if _decay_usage_capacity_reached_in_process:
        return True

    try:
        with _state_lock:
            config = _load_config()
            entries = _recent_decay_usage_entries(config, datetime.now(timezone.utc))
            at_capacity = len(entries) >= DECAY_USAGE_CAP
            if at_capacity:
                _decay_usage_capacity_reached_in_process = True
            return at_capacity
    except Exception:
        return True


def _record_decay_usage_opportunity(
    *,
    variant: str,
    sync_type: str,
    trigger_function: str,
    trigger_source: str,
    trigger_reason: str,
    delete_count: Optional[int],
    deleted_count: Optional[int],
) -> bool:
    global _decay_usage_capacity_reached_in_process
    try:
        with _state_lock:
            now = datetime.now(timezone.utc)
            config = _load_config()
            entries = _recent_decay_usage_entries(config, now)
            if len(entries) >= DECAY_USAGE_CAP:
                _decay_usage_capacity_reached_in_process = True
                return False

            entry = {
                "evaluated_at": now.isoformat(),
                "variant": variant,
                "sync_type": sync_type,
                "trigger_function": trigger_function,
                "trigger_source": trigger_source,
                "trigger_reason": trigger_reason,
            }
            if delete_count is not None:
                entry["delete_count"] = delete_count
            if deleted_count is not None:
                entry["deleted_count"] = deleted_count
            entries.append(entry)

            state = config.get(STATE_SECTION)
            if not isinstance(state, dict):
                state = {}
            decay_state = state.get(DECAY_USAGE_STATE_KEY)
            if not isinstance(decay_state, dict):
                decay_state = {}
            decay_state["events"] = entries
            state[DECAY_USAGE_STATE_KEY] = decay_state
            config[STATE_SECTION] = state
            _write_config(config)
            if len(entries) >= DECAY_USAGE_CAP:
                _decay_usage_capacity_reached_in_process = True
            return True
    except Exception:
        return False


def _recent_decay_usage_entries(config: Dict[str, Any], now: datetime):
    state = config.get(STATE_SECTION)
    if not isinstance(state, dict):
        return []

    decay_state = state.get(DECAY_USAGE_STATE_KEY)
    if not isinstance(decay_state, dict):
        return []

    entries = decay_state.get("events")
    if not isinstance(entries, list):
        return []

    cutoff = now - DECAY_USAGE_WINDOW
    recent = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        evaluated_at = _parse_datetime(entry.get("evaluated_at"))
        if evaluated_at is not None and evaluated_at >= cutoff:
            recent.append(entry)
    return recent


def _scale_threshold_at_capacity() -> bool:
    global _scale_threshold_capacity_reached_in_process
    if _scale_threshold_capacity_reached_in_process:
        return True

    try:
        with _state_lock:
            config = _load_config()
            entries = _recent_scale_threshold_entries(config, datetime.now(timezone.utc))
            at_capacity = len(entries) >= SCALE_THRESHOLD_CAP
            if at_capacity:
                _scale_threshold_capacity_reached_in_process = True
            return at_capacity
    except Exception:
        return True


def _record_scale_threshold_opportunity(
    *,
    variant: str,
    sync_type: str,
    trigger_function: str,
    trigger_source: str,
    trigger_reason: str,
    top_k: Optional[int],
    memory_count: Optional[int],
    threshold: Optional[int],
) -> bool:
    global _scale_threshold_capacity_reached_in_process
    try:
        with _state_lock:
            now = datetime.now(timezone.utc)
            config = _load_config()
            entries = _recent_scale_threshold_entries(config, now)
            if len(entries) >= SCALE_THRESHOLD_CAP:
                _scale_threshold_capacity_reached_in_process = True
                return False

            entry = {
                "evaluated_at": now.isoformat(),
                "variant": variant,
                "sync_type": sync_type,
                "trigger_function": trigger_function,
                "trigger_source": trigger_source,
                "trigger_reason": trigger_reason,
            }
            if top_k is not None:
                entry["top_k"] = top_k
            if memory_count is not None:
                entry["memory_count"] = memory_count
            if threshold is not None:
                entry["threshold"] = threshold
            entries.append(entry)

            state = config.get(STATE_SECTION)
            if not isinstance(state, dict):
                state = {}
            scale_state = state.get(SCALE_THRESHOLD_STATE_KEY)
            if not isinstance(scale_state, dict):
                scale_state = {}
            scale_state["events"] = entries
            if trigger_source == "memory_count":
                scale_state["memory_count_threshold_evaluated"] = True
            state[SCALE_THRESHOLD_STATE_KEY] = scale_state
            config[STATE_SECTION] = state
            _write_config(config)
            if len(entries) >= SCALE_THRESHOLD_CAP:
                _scale_threshold_capacity_reached_in_process = True
            return True
    except Exception:
        return False


def _mark_scale_memory_count_threshold_evaluated() -> bool:
    global _scale_memory_count_threshold_evaluated_in_process
    try:
        with _state_lock:
            config = _load_config()
            state = config.get(STATE_SECTION)
            if not isinstance(state, dict):
                state = {}
            scale_state = state.get(SCALE_THRESHOLD_STATE_KEY)
            if not isinstance(scale_state, dict):
                scale_state = {}
            if scale_state.get("memory_count_threshold_evaluated"):
                _scale_memory_count_threshold_evaluated_in_process = True
                return False

            scale_state["memory_count_threshold_evaluated"] = True
            state[SCALE_THRESHOLD_STATE_KEY] = scale_state
            config[STATE_SECTION] = state
            _write_config(config)
            _scale_memory_count_threshold_evaluated_in_process = True
            return True
    except Exception:
        return False


def _recent_scale_threshold_entries(config: Dict[str, Any], now: datetime):
    state = config.get(STATE_SECTION)
    if not isinstance(state, dict):
        return []

    scale_state = state.get(SCALE_THRESHOLD_STATE_KEY)
    if not isinstance(scale_state, dict):
        return []

    entries = scale_state.get("events")
    if not isinstance(entries, list):
        return []

    cutoff = now - SCALE_THRESHOLD_WINDOW
    recent = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        evaluated_at = _parse_datetime(entry.get("evaluated_at"))
        if evaluated_at is not None and evaluated_at >= cutoff:
            recent.append(entry)
    return recent


def _performance_slow_query_at_capacity() -> bool:
    global _performance_slow_query_capacity_reached_in_process
    if _performance_slow_query_capacity_reached_in_process:
        return True

    try:
        with _state_lock:
            config = _load_config()
            entries = _recent_performance_slow_query_entries(config, datetime.now(timezone.utc))
            at_capacity = len(entries) >= PERFORMANCE_SLOW_QUERY_CAP
            if at_capacity:
                _performance_slow_query_capacity_reached_in_process = True
            return at_capacity
    except Exception:
        return True


def _record_performance_slow_query_opportunity(
    *,
    variant: str,
    sync_type: str,
    trigger_function: str,
    trigger_reason: str,
) -> bool:
    global _performance_slow_query_capacity_reached_in_process
    try:
        with _state_lock:
            now = datetime.now(timezone.utc)
            config = _load_config()
            entries = _recent_performance_slow_query_entries(config, now)
            if len(entries) >= PERFORMANCE_SLOW_QUERY_CAP:
                _performance_slow_query_capacity_reached_in_process = True
                return False

            entries.append(
                {
                    "evaluated_at": now.isoformat(),
                    "variant": variant,
                    "sync_type": sync_type,
                    "trigger_function": trigger_function,
                    "trigger_reason": trigger_reason,
                }
            )

            state = config.get(STATE_SECTION)
            if not isinstance(state, dict):
                state = {}
            performance_state = state.get(PERFORMANCE_SLOW_QUERY_STATE_KEY)
            if not isinstance(performance_state, dict):
                performance_state = {}
            performance_state["events"] = entries
            state[PERFORMANCE_SLOW_QUERY_STATE_KEY] = performance_state
            config[STATE_SECTION] = state
            _write_config(config)
            if len(entries) >= PERFORMANCE_SLOW_QUERY_CAP:
                _performance_slow_query_capacity_reached_in_process = True
            return True
    except Exception:
        return False


def _recent_performance_slow_query_entries(config: Dict[str, Any], now: datetime):
    state = config.get(STATE_SECTION)
    if not isinstance(state, dict):
        return []

    performance_state = state.get(PERFORMANCE_SLOW_QUERY_STATE_KEY)
    if not isinstance(performance_state, dict):
        return []

    entries = performance_state.get("events")
    if not isinstance(entries, list):
        return []

    cutoff = now - PERFORMANCE_SLOW_QUERY_WINDOW
    recent = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        evaluated_at = _parse_datetime(entry.get("evaluated_at"))
        if evaluated_at is not None and evaluated_at >= cutoff:
            recent.append(entry)
    return recent


def _parse_datetime(value: Any):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _get_notice_state(config: Dict[str, Any], state_key: str) -> Dict[str, Any]:
    state = config.get(STATE_SECTION)
    if not isinstance(state, dict):
        return {}

    notice_state = state.get(state_key)
    return notice_state if isinstance(notice_state, dict) else {}


def _count_added_memories(add_result: Any) -> int:
    results = add_result.get("results") if isinstance(add_result, dict) else add_result
    if not isinstance(results, list):
        return 0

    count = 0
    for item in results:
        if isinstance(item, dict) and item.get("event") == "ADD":
            count += 1
    return count


def _get_provider_memory_count(memory_instance) -> Optional[int]:
    vector_store = getattr(memory_instance, "vector_store", None)
    if vector_store is None:
        return None

    try:
        count = getattr(vector_store, "count", None)
        if callable(count):
            value = _coerce_nonnegative_int(count(), None)
            if value is not None:
                return value
    except Exception:
        pass

    try:
        col_info = getattr(vector_store, "col_info", None)
        if callable(col_info):
            collection_name = getattr(vector_store, "collection_name", None)
            if collection_name is None:
                schema = getattr(vector_store, "schema", None)
                if isinstance(schema, dict):
                    index = schema.get("index")
                    if isinstance(index, dict):
                        collection_name = index.get("name")
            if collection_name is not None:
                try:
                    info = col_info(collection_name)
                except TypeError:
                    info = col_info()
            else:
                info = col_info()
            value = _extract_count(info)
            if value is not None:
                return value

            client = getattr(vector_store, "client", None)
            client_count = (
                getattr(client, "count", None) if client is not None and collection_name is not None else None
            )
            if callable(client_count):
                return _extract_count(client_count(index=collection_name))
    except Exception:
        return None

    return None


def _extract_count(info: Any) -> Optional[int]:
    if info is None:
        return None

    if isinstance(info, dict):
        for key in ("count", "points_count", "vectors_count", "indexed_vectors_count", "num_docs"):
            value = _coerce_nonnegative_int(info.get(key), None)
            if value is not None:
                return value
        return None

    model_dump = getattr(info, "model_dump", None)
    if callable(model_dump):
        try:
            value = _extract_count(model_dump())
            if value is not None:
                return value
        except Exception:
            return None

    for attr in ("count", "points_count", "vectors_count", "indexed_vectors_count", "num_docs"):
        value = _coerce_nonnegative_int(getattr(info, attr, None), None)
        if value is not None:
            return value

    return None


def _render_scale_copy(copy_template: Any, *, top_k: Optional[int], memory_count: Optional[int]) -> Optional[str]:
    if not isinstance(copy_template, str) or not copy_template.strip():
        return None
    try:
        return copy_template.format(top_k=top_k, memory_count=memory_count)
    except Exception:
        return copy_template


def _coerce_nonnegative_int(value: Any, default: Optional[int]) -> Optional[int]:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _coerce_mapping(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _walk_mapping(value: Any, parent_key: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            yield key_text, child
            yield from _walk_mapping(child, key_text)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _walk_mapping(child, parent_key)


def _is_temporal_key(key: Any) -> bool:
    key_text = str(key).lower()
    exact_keys = {
        "date",
        "time",
        "timestamp",
        "datetime",
        "event_date",
        "reference_date",
        "created_at",
        "updated_at",
        "started_at",
        "ended_at",
        "expires_at",
    }
    return (
        key_text in exact_keys
        or key_text.endswith("_date")
        or key_text.endswith("_time")
        or key_text.endswith("_at")
        or "timestamp" in key_text
    )


def _looks_temporal_value(value: Any, allow_epoch: bool) -> bool:
    if isinstance(value, datetime):
        return True
    if isinstance(value, date):
        return True
    if isinstance(value, str):
        return bool(_ISO_DATE_RE.search(value) or _RELATIVE_TIME_RE.search(value))
    if allow_epoch and isinstance(value, (int, float)) and not isinstance(value, bool):
        return 946684800 <= value <= 4102444800 or 946684800000 <= value <= 4102444800000
    return False


def _has_temporal_filter(filters: Any) -> bool:
    if not isinstance(filters, dict):
        return False

    for key, value in filters.items():
        if key in {"AND", "OR", "NOT", "$and", "$or", "$not"}:
            if isinstance(value, list) and any(_has_temporal_filter(item) for item in value):
                return True
            if isinstance(value, dict) and _has_temporal_filter(value):
                return True
            continue

        temporal_key = _is_temporal_key(key)
        if isinstance(value, dict):
            range_values = [item for op, item in value.items() if op in _RANGE_OPERATORS]
            if range_values and (
                temporal_key
                or any(_looks_temporal_value(item, allow_epoch=temporal_key) for item in range_values)
            ):
                return True
            if _has_temporal_filter(value):
                return True
        elif temporal_key and _looks_temporal_value(value, allow_epoch=True):
            return True

    return False
