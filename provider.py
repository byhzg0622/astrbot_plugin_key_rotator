"""Key 轮询版 OpenAI Chat Completion 提供商适配器（注册新提供商类型，不改核心代码）。

设计要点
--------
1. 注册连接点：astrbot.core.provider.register.register_provider_adapter 是
   AstrBot 自己注册提供商类型的入口。本模块注册新类型
   openai_chat_completion_rotate，WebUI「服务提供商 -> 新增」的类型列表会自动
   出现它（dashboard/services/config_service.py 把 provider_registry 合并进
   config_template），配置表单由本模块提供的 DEFAULT_CONFIG_TMPL 生成。

2. 轮换挂接点：原生 ProviderOpenAIOfficial.text_chat() 每次调用都会执行
       available_api_keys = self.api_keys.copy()
       chosen_key = random.choice(available_api_keys)
   我们把 self.api_keys 换成 RotatingKeyRing（list 子类）并重写 copy()，
   让它返回「本次轮到的那个 Key」——单元素列表，random.choice 必然命中它。
   于是底层的选 Key、429 换 Key、上下文裁剪等逻辑一行都不用改。

3. 超时换 Key：原生对超时是「同 Key 重试若干次 -> 抛出 -> 交给回退模型链」。
   本类在 text_chat / text_chat_stream 外面包一层：超时、连接失败、429、
   鉴权/配额错误时换下一个 Key 再试（次数可配）。
   流式请求一旦已经吐出过 chunk 就不重试，避免内容重复。

4. 熔断冷却：某个 Key 连续失败达到阈值后冷却 cooldown_seconds 秒，轮换时自动
   跳过；全部 Key 都在冷却时降级为全量轮换（fail-open，绝不阻断请求）。

5. 配置取值：默认读插件全局配置（改动立即生效，无需重启）；只有当服务商配置
   里显式带了 rotate_override=true 时，才用该服务商自己的 rotate_* 字段。

6. 并发安全：copy() 是同步调用（前后没有 await），且「本次用哪个 Key」用
   contextvars 记录——每个 asyncio Task 各有一份上下文，并发请求不会串号。

7. 插件热重载会清理 sys.modules 后重新 import，装饰器会重复执行，
   所以注册处做了幂等保护（已存在则只更新类引用）。
"""

from __future__ import annotations

import asyncio
import contextvars
import random
import time
from collections import deque
from typing import Any, AsyncGenerator

from astrbot.api import logger
from astrbot.core.provider.register import provider_cls_map, register_provider_adapter
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.utils.network_utils import is_connection_error

PROVIDER_TYPE = "openai_chat_completion_rotate"
PROVIDER_DISPLAY_NAME = "OpenAI Compatible（多 Key 轮询）"

# 插件级设置：由 main.py 的 Star 类在加载/保存配置时写入；未加载时用默认值。
PLUGIN_SETTINGS: dict[str, Any] = {
    "enable": True,
    "log_key_usage": False,
    "rotate_mode": "round_robin",
    "rotate_on_timeout": True,
    "rotate_on_429": True,
    "switch_retries": 1,
    "cooldown_seconds": 60,
    "cooldown_threshold": 2,
    "rate_limit_per_min": 0,  # 每分钟总调用数；0 = 不限速
}

RATE_WINDOW_S = 60  # 限速窗口固定 1 分钟

# 本次 text_chat 调用实际分到的 Key（每个 asyncio Task 独立，防并发串号）
_current_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "astrbot_plugin_key_rotator_current_key", default=None
)

# 实时事件流（面板右下角动态）：use=成功一次 / fail=失败 / switch=换 Key 重试 / cooldown=进入冷却
EVENTS: deque = deque(maxlen=200)
EVENT_KIND_LABEL = {
    "use": "成功",
    "fail": "失败",
    "switch": "换 Key",
    "cooldown": "冷却",
    "quota": "额度满",
    "fallback": "移交备用",
}


class ModelSkippedError(RuntimeError):
    """本模型连续失败已进入跳过状态（模型级熔断）。

    立即抛出，让 Agent Runner 跳过这个模型去试下一个，全部模型不可用时
    自然落到回退链末端的备用 API——不在模型内部一个个慢慢试。
    """


class RateLimitExhaustedError(RuntimeError):
    """本线路本轮额度已用尽。

    抛给 Agent Runner 后会被当作普通请求异常，自动切换到「回退对话模型」
    （即备用模型），窗口重置后自动回到本线路。
    """


def push_event(
    kind: str,
    *,
    source: str = "",
    model: str = "",
    key: Any = None,
    detail: str = "",
) -> None:
    """往事件流里追加一条（单事件循环下同步写入，无锁）。"""
    EVENTS.append(
        {
            "t": time.time(),
            "kind": kind,
            "label": EVENT_KIND_LABEL.get(kind, kind),
            "source": source,
            "model": model,
            "key": mask_key(key) if key is not None else "",
            "detail": str(detail or "")[:200],
        }
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def mask_key(key: Any) -> str:
    """Key 脱敏，日志与命令输出统一用它。"""
    text = str(key or "")
    if not text:
        return "(空)"
    if len(text) <= 10:
        return text[:2] + "***"
    return f"{text[:6]}...{text[-4:]}"


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def _is_timeout_like(exc: BaseException) -> bool:
    if is_connection_error(exc):
        return True
    return type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}


def _is_rate_limited(exc: BaseException) -> bool:
    if _status_code(exc) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def is_switchable_error(exc: BaseException) -> bool:
    """换一个 Key 重试有意义的错误：线路问题、限流、鉴权与配额。"""
    if _is_timeout_like(exc):
        return True
    if _status_code(exc) in {401, 403, 408, 429, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "rate limit",
            "too many requests",
            "quota",
            "invalid api key",
            "incorrect api key",
            "invalid_api_key",
            "authentication",
            "unauthorized",
        )
    )


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Key 环：轮换 + 冷却 + 统计
# ---------------------------------------------------------------------------
class RotatingKeyRing(list):
    """覆盖 text_chat 内部的 api_keys.copy()，把「随机取」变成「轮到谁取谁」。"""

    def __init__(
        self,
        keys: list[str],
        *,
        source_id: str,
        source_config: dict,
        model_id: str = "",
    ) -> None:
        super().__init__(keys)
        self.source_id = source_id
        self.model_id = model_id  # 报错/事件按「模型」记，不按 api 记
        self.src_cfg = source_config or {}
        self._model_fail = 0
        self._model_skip_until = 0.0
        self.override_keys: list[str] = []
        self._i = 0
        self._ov_i = 0
        self._window_start = time.monotonic()
        self._last_rate = 0
        self.stats: dict[str, dict[str, Any]] = {
            key: {
                "assigned": 0,
                "ok": 0,
                "fail": 0,
                "consecutive_fail": 0,
                "cooldown_until": 0.0,
                "last_error": "",
                "used": 0,  # 本窗口已用额度
            }
            for key in self
        }

    # -- 配置（默认读插件全局，source 显式 override 时读自己） ------------
    def _get(self, key: str, default: Any) -> Any:
        if _as_bool(self.src_cfg.get("rotate_override"), False) and key in self.src_cfg:
            return self.src_cfg.get(key)
        return PLUGIN_SETTINGS.get(key, default)

    @property
    def mode(self) -> str:
        mode = str(self._get("rotate_mode", "round_robin"))
        return mode if mode in {"round_robin", "random"} else "round_robin"

    @property
    def rotate_on_timeout(self) -> bool:
        return _as_bool(self._get("rotate_on_timeout", True), True)

    @property
    def rotate_on_429(self) -> bool:
        return _as_bool(self._get("rotate_on_429", True), True)

    @property
    def switch_retries(self) -> int:
        return max(0, min(_as_int(self._get("switch_retries", 1), 1), 5))

    @property
    def cooldown_seconds(self) -> int:
        return max(0, _as_int(self._get("cooldown_seconds", 60), 60))

    @property
    def cooldown_threshold(self) -> int:
        return max(1, _as_int(self._get("cooldown_threshold", 2), 2))

    # -- 限速（配置里的限速数据：每分钟总条数，0 = 不限速） ----------------
    @property
    def rate_limit(self) -> int:
        return max(0, _as_int(self._get("rate_limit_per_min", 0), 0))

    @property
    def active_pool(self) -> list[str]:
        """当前实际参与轮换的 Key 集合（手动池优先）。"""
        return list(self.override_keys) if self.override_keys else list(self)

    @property
    def quota(self) -> int:
        """均分后的单 Key 额度（每窗口，舍去小数）。不限速时为 0。"""
        rate = self.rate_limit
        pool = self.active_pool
        if rate <= 0 or not pool:
            return 0
        return rate // len(pool)

    def _roll_window(self, now: float) -> None:
        """窗口到期或限速值变化 -> 清零用量。"""
        rate = self.rate_limit
        if rate != self._last_rate:
            self._last_rate = rate
            self._window_start = now
            for st in self.stats.values():
                st["used"] = 0
            return
        if rate > 0 and now - self._window_start >= RATE_WINDOW_S:
            self._window_start = now
            for st in self.stats.values():
                st["used"] = 0

    def _remaining_window(self, now: float) -> int:
        if self.rate_limit <= 0:
            return 0
        left = RATE_WINDOW_S - (now - self._window_start)
        return max(0, int(left) + (1 if left % 1 else 0))

    def _select_key(self, candidates: list[str], counter_name: str) -> str:
        """挑一个可用 Key；额度全部用尽时抛 RateLimitExhaustedError。"""
        now = time.monotonic()
        self._roll_window(now)

        def _cooling(k: str) -> bool:
            return self.stats.get(k, {}).get("cooldown_until", 0.0) > now

        # 1) 冷却中的 Key 一律跳过，绝不参与轮询
        available = [k for k in candidates if not _cooling(k)]
        if not available:
            wake_in = max(
                0,
                int(
                    min(
                        self.stats.get(k, {}).get("cooldown_until", now) - now
                        for k in candidates
                    )
                )
                + 1,
            )
            push_event(
                "fallback",
                source=self.source_id,
                model=self.model_id,
                detail=f"{len(candidates)} 个 Key 全部冷却中，最早 {wake_in}s 后恢复",
            )
            raise RateLimitExhaustedError(
                f"模型 {self.model_label} 的 {len(candidates)} 个 Key 全部处于冷却中"
                f"（连续失败 {self.cooldown_threshold} 次触发），"
                f"已跳过该模型，最早 {wake_in}s 后恢复轮换。"
            )

        # 2) 限速 < Key 数，均分后每 Key 额度为 0：一次都发不出去，直接移交
        quota = self.quota
        if self.rate_limit > 0 and quota <= 0:
            raise RateLimitExhaustedError(
                f"模型 {self.model_label}（线路 {self.source_id}）限速 "
                f"{self.rate_limit}/min 小于 Key 数 {len(candidates)}，"
                "均分额度为 0，无法发送，已跳过该模型。"
                "请调高限速或删掉几个模型。"
            )

        # 3) 额度控制：只在没满额的可用 Key 里挑
        if quota > 0:
            fresh = [
                k
                for k in available
                if self.stats.get(k, {}).get("used", 0) < quota
            ]
            if not fresh:
                push_event(
                    "fallback",
                    source=self.source_id,
                    model=self.model_id,
                    detail=(
                        f"额度用尽 {self.rate_limit}/min ÷ {len(candidates)} = "
                        f"{quota}/Key，跳过该模型，"
                        f"{self._remaining_window(now)}s 后重置"
                    ),
                )
                raise RateLimitExhaustedError(
                    f"模型 {self.model_label} 本窗口额度已用尽"
                    f"（{self.rate_limit}/min ÷ {len(candidates)} 个 Key = "
                    f"{quota}/Key），已跳过该模型，"
                    f"{self._remaining_window(now)}s 后重置。"
                )
            available = fresh

        return self._pick(available, counter_name)

    # -- 手动 Key 池（面板下发，用于配置里拿不到 Key 的场景） -------------
    @staticmethod
    def _new_stat() -> dict:
        return {
            "assigned": 0,
            "ok": 0,
            "fail": 0,
            "consecutive_fail": 0,
            "cooldown_until": 0.0,
            "last_error": "",
            "used": 0,
        }

    def set_override(self, keys: Any) -> int:
        """设置手动 Key 池：之后每次调用只在池内轮换。

        用途：当线路**配置里没有/拿不到 Key** 时，手动填多个进来直接开轮换。
        优先级最高（不受总开关、原 Key 列表影响），可随时撤销。

        Args:
            keys: 单个 Key 字符串，或 Key 列表。

        Returns:
            实际生效的 Key 数量。
        """
        if isinstance(keys, str):
            keys = [keys]
        cleaned: list[str] = []
        for raw in keys or []:
            key = str(raw or "").strip()
            if key and key not in cleaned:
                cleaned.append(key)
        if not cleaned:
            return 0
        self.override_keys = cleaned
        self._ov_i = 0
        for key in cleaned:
            if key not in self.stats:
                self.stats[key] = self._new_stat()
        return len(cleaned)

    def clear_override(self) -> None:
        """撤销手动池，回到线路配置里的 Key 列表轮换。"""
        self.override_keys = []
        self._ov_i = 0
        for key in [k for k in self.stats if k not in self]:
            self.stats.pop(key, None)

    def _pick(self, candidates: list[str], counter_name: str) -> str:
        """在候选里选一个：random 随机，round_robin 按索引轮；冷却中的跳过。"""
        now = time.monotonic()
        eligible = [
            k
            for k in candidates
            if self.stats.get(k, {}).get("cooldown_until", 0.0) <= now
        ]
        pool = eligible or list(candidates)  # 全部冷却 -> fail-open，绝不阻断
        if self.mode == "random":
            return random.choice(pool)

        pool_set = set(pool)
        start = getattr(self, counter_name)
        total = len(candidates)
        for step in range(total):
            cand = candidates[(start + step) % total]
            if cand in pool_set:
                setattr(self, counter_name, (start + step + 1) % total)
                return cand
        return pool[0]

    # -- 轮换核心 ---------------------------------------------------------
    def copy(self):  # noqa: A003 - 故意覆盖 list.copy
        """text_chat 每次调用都会执行这行，返回单元素列表即可锁定本次用的 Key。"""
        if not PLUGIN_SETTINGS.get("enable", True) and not self.override_keys:
            # 总开关关闭（手动池除外，它是显式指定的）-> 与原生一致：全量随机
            return list(self)

        # 模型级熔断：该模型连续失败已被跳过 -> 立刻抛出，让 runner 去试下一个模型
        if self.model_skipped:
            raise ModelSkippedError(
                f"模型 {self.model_label} 连续失败，已跳过 "
                f"{self.model_skip_remaining()}s，直接交由下一个模型/备用 API。"
            )

        pool = self.active_pool
        if not pool:
            return list(self)

        chosen = self._select_key(pool, "_ov_i" if self.override_keys else "_i")
        st = self.stats.get(chosen)
        if st:
            st["assigned"] += 1
            st["used"] = st.get("used", 0) + 1
        _current_key.set(chosen)
        if PLUGIN_SETTINGS.get("log_key_usage"):
            logger.debug(
                "[Key轮询] %s %s -> %s（第 %d/%d 个，本窗口 %s/%s）",
                self.source_id,
                "手动池" if self.override_keys else "轮换",
                mask_key(chosen),
                (self._ov_i if self.override_keys else self._i - 1)
                % len(pool)
                + 1,
                len(pool),
                st.get("used", 0) if st else 0,
                self.quota or "-",
            )
        return [chosen]

    # -- 模型级熔断：连续失败 N 次 -> 跳过整个模型 --------------------------
    @property
    def model_label(self) -> str:
        return self.model_id or self.source_id

    def model_skip_remaining(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        return max(0, int(self._model_skip_until - now) + (1 if (self._model_skip_until - now) % 1 else 0))

    @property
    def model_skipped(self) -> bool:
        return self._model_skip_until > time.monotonic()

    def note_model_success(self) -> None:
        self._model_fail = 0

    def note_model_failure(self) -> None:
        """一次完整的请求失败（不是单次尝试）。"""
        now = time.monotonic()
        self._model_fail += 1
        threshold = max(1, self.cooldown_threshold)
        if self._model_fail < threshold or now < self._model_skip_until:
            return
        duration = self.cooldown_seconds or 60
        self._model_skip_until = now + duration
        self._model_fail = 0
        push_event(
            "cooldown",
            source=self.source_id,
            model=self.model_id,
            detail=f"模型连续失败 {threshold} 次，跳过 {duration}s",
        )
        logger.warning(
            "[Key轮询] 模型 %s 连续失败，跳过 %d 秒，交由下一个模型/备用。",
            self.model_label,
            duration,
        )

    # -- 成败上报 ---------------------------------------------------------
    def report_success(self, key: Any) -> None:
        if not key:
            return
        st = self.stats.get(str(key))
        if not st:
            return
        st["ok"] += 1
        st["consecutive_fail"] = 0
        st["last_error"] = ""  # 成功一次即消除报错

    def report_failure(self, key: Any, exc: BaseException) -> None:
        if not key:
            return
        st = self.stats.get(str(key))
        if not st:
            return
        now = time.monotonic()
        st["fail"] += 1
        st["consecutive_fail"] += 1
        # 失败即记录并一直显示，直到该 Key 下一次成功才消除
        st["last_error"] = f"{type(exc).__name__}: {exc}"[:200]

        threshold = self.cooldown_seconds
        if threshold <= 0:
            return
        if (
            st["consecutive_fail"] >= self.cooldown_threshold
            and now >= st["cooldown_until"]
        ):
            st["cooldown_until"] = now + threshold
            st["consecutive_fail"] = 0
            push_event(
                "cooldown",
                source=self.source_id,
                model=self.model_id,
                key=key,
                detail=f"Key 连续失败 {self.cooldown_threshold} 次，冷却 {threshold} 秒",
            )
            logger.warning(
                "[Key轮询] %s 连续失败 %d 次，冷却 %d 秒。最后错误：%s",
                mask_key(key),
                self.cooldown_threshold,
                threshold,
                st["last_error"],
            )

    def should_switch(self, exc: BaseException) -> bool:
        if not is_switchable_error(exc):
            return False
        if _is_timeout_like(exc):
            return self.rotate_on_timeout
        if _is_rate_limited(exc):
            return self.rotate_on_429
        return True  # 401/403/quota 等 Key 本身的问题，始终换

    # -- 状态快照 ---------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        now = time.monotonic()
        self._roll_window(now)
        overridden = bool(self.override_keys)
        # 生效集合：有手动池就展示手动池——那才是真正被调用使用的 Key
        active: list[str] = list(self.override_keys) if overridden else list(self)
        rate = self.rate_limit
        quota = self.quota
        items = []
        for idx, key in enumerate(active, start=1):
            st = self.stats.get(key, {})
            remain = max(0.0, float(st.get("cooldown_until", 0.0)) - now)
            used = int(st.get("used", 0))
            exhausted = bool(quota and used >= quota)
            items.append(
                {
                    "index": idx,
                    "key": mask_key(key),
                    "manual": overridden,
                    "assigned": st.get("assigned", 0),
                    "ok": st.get("ok", 0),
                    "fail": st.get("fail", 0),
                    "cooldown": int(round(remain)),
                    "last_error": st.get("last_error", ""),  # 失败即显示，成功才消除
                    "used": used,
                    "quota": quota,
                    "exhausted": exhausted,
                    "skipped": bool(remain > 0) or exhausted,
                }
            )

        # 「下一个」必须指向真正会被用到的 Key：跳过冷却中与满额的
        available = {i["index"] for i in items if not i["skipped"]}
        next_index = 0
        if active and available:
            total = len(active)
            base = (self._ov_i if overridden else self._i) % total
            for step in range(total):
                pos = (base + step) % total + 1
                if pos in available:
                    next_index = pos
                    break
        return {
            "source_id": self.source_id,
            "mode": self.mode,
            "enabled": bool(PLUGIN_SETTINGS.get("enable", True)),
            "total": len(active),
            "source_key_count": len(self),
            "overridden": overridden,
            "override": [mask_key(k) for k in self.override_keys],
            "override_total": len(self.override_keys),
            "next_index": next_index,
            "model_id": self.model_id,
            "model_label": self.model_label,
            "model_skipped": self.model_skip_remaining(now),
            "model_fail": self._model_fail,
            "available": len(available),
            "all_unavailable": bool(active) and not available,
            "rate_limit": rate,
            "quota": quota,
            "limited": rate > 0,
            "window": RATE_WINDOW_S if rate > 0 else 0,
            "remaining": self._remaining_window(now) if rate > 0 else 0,
            "all_exhausted": bool(
                quota and active and all(i["exhausted"] for i in items)
            ),
            "keys": items,
        }


# ---------------------------------------------------------------------------
# 提供商本体
# ---------------------------------------------------------------------------
class RotatingOpenAIProvider(ProviderOpenAIOfficial):
    """多 Key 轮询 / 超时换 Key / 失败冷却 的 OpenAI 兼容提供商。"""

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        cfg = provider_config or {}

        raw = self.api_keys
        if isinstance(raw, str):
            raw = [raw]
        elif not isinstance(raw, list):
            raw = [raw]
        keys = [str(k).strip() for k in raw if str(k or "").strip()] or [""]

        self.key_ring = RotatingKeyRing(
            keys,
            source_id=str(cfg.get("provider_source_id") or cfg.get("id") or "unknown"),
            source_config=cfg,
            model_id=str(cfg.get("id") or ""),
        )
        # 底层全部通过 self.api_keys.copy() 取 Key，这里替换成环
        self.api_keys = self.key_ring

    # -- 非流式 -----------------------------------------------------------
    async def text_chat(self, *args: Any, **kwargs: Any) -> Any:
        attempts = self.key_ring.switch_retries
        last: BaseException | None = None
        for i in range(attempts + 1):
            # 关键：每次尝试都重置归因。否则第 2 次尝试若在 copy() 之前就抛
            # （例如全部冷却），失败会被记到上一次的 Key 头上 -> fail > assigned
            # -> KPI 成功次数变负数。
            _current_key.set(None)
            try:
                response = await super().text_chat(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last = exc
                self._on_failed(exc)
                if i >= attempts or not self.key_ring.should_switch(exc):
                    self._note_model_failure(exc)
                    raise
                push_event(
                    "switch",
                    source=self.key_ring.source_id,
                    model=self._model_id(),
                    key=_current_key.get(),
                    detail=f"{type(exc).__name__}: {exc}",
                )
                logger.warning(
                    "[Key轮询] %s 调用失败，换下一个 Key 重试（%d/%d）：%s: %s",
                    self.key_ring.source_id,
                    i + 1,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(0.3)
            else:
                self._on_ok()
                self.key_ring.note_model_success()
                return response
        raise last if last else RuntimeError("Key 轮询重试耗尽")

    # -- 流式 -------------------------------------------------------------
    async def text_chat_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        attempts = self.key_ring.switch_retries
        for i in range(attempts + 1):
            produced = False
            _current_key.set(None)  # 每次尝试重置归因，避免串到上一次的 Key
            try:
                async for chunk in super().text_chat_stream(*args, **kwargs):
                    produced = True
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                self._on_failed(exc)
                # 已经吐出 chunk 就不能重来，否则内容会重复
                if produced or i >= attempts or not self.key_ring.should_switch(exc):
                    self._note_model_failure(exc)
                    raise
                push_event(
                    "switch",
                    source=self.key_ring.source_id,
                    model=self._model_id(),
                    key=_current_key.get(),
                    detail=f"{type(exc).__name__}: {exc}",
                )
                logger.warning(
                    "[Key轮询] %s 流式请求失败，换下一个 Key 重试（%d/%d）：%s: %s",
                    self.key_ring.source_id,
                    i + 1,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(0.3)
                continue
            else:
                self._on_ok()
                self.key_ring.note_model_success()
                return

    # -- 成败归因 ---------------------------------------------------------
    def _model_id(self) -> str:
        """事件/报错统一用完整模型 id（与回退链里的 id 一致）。"""
        return str(
            (getattr(self, "provider_config", None) or {}).get("id") or self.get_model()
        )

    def _note_model_failure(self, exc: BaseException) -> None:
        """一次完整的请求失败才计模型级失败；已在跳过状态不重复计数。"""
        if isinstance(exc, ModelSkippedError):
            return
        self.key_ring.note_model_failure()

    def _on_ok(self) -> None:
        key = _current_key.get()
        if key:
            self.key_ring.report_success(key)
            push_event(
                "use",
                source=self.key_ring.source_id,
                model=self._model_id(),
                key=key,
            )

    def _on_failed(self, exc: BaseException) -> None:
        key = _current_key.get()
        if not key:
            return  # 这次尝试还没分配到 Key（例如 copy() 直接抛出），不记账
        self.key_ring.report_failure(key, exc)
        push_event(
            "fail",
            source=self.key_ring.source_id,
            model=self._model_id(),
            key=key,
            detail=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# 覆盖 Key 的接管：连没改成轮询型的原生实例也能接管
# ---------------------------------------------------------------------------
def adopt_key_override(inst: Any, keys: Any) -> bool:
    """把面板下发的**手动 Key 池**应用到任意 OpenAI 兼容提供商实例。

    依据：原生 text_chat / text_chat_stream 每次调用都只做
        available_api_keys = self.api_keys.copy()
        chosen_key = random.choice(available_api_keys)
    所以只要把 api_keys 换成「带 override 的环」，就能控制每次调用用哪个 Key——
    **不需要改服务商 type、不需要写配置、不需要重启**。

    Returns:
        True 表示已接管；False 表示该实例不支持（非 OpenAI 兼容实现）。
    """
    if not isinstance(inst, ProviderOpenAIOfficial):
        return False
    ring = getattr(inst, "api_keys", None)
    if not isinstance(ring, RotatingKeyRing):
        original = list(ring) if isinstance(ring, list) else [str(ring or "")]
        cfg = getattr(inst, "provider_config", None) or {}
        ring = RotatingKeyRing(
            original,
            source_id=str(cfg.get("provider_source_id") or cfg.get("id") or "unknown"),
            source_config=cfg,
            model_id=str(cfg.get("id") or ""),
        )
        inst.api_keys = ring
        # 与轮询型实例保持同名属性，便于上层统一读取
        inst.key_ring = ring
    return bool(ring.set_override(keys))


def release_key_override(inst: Any) -> bool:
    """撤销覆盖，回到该实例原本的 Key 列表。"""
    ring = getattr(inst, "api_keys", None)
    if isinstance(ring, RotatingKeyRing):
        ring.clear_override()
        return True
    return False


# ---------------------------------------------------------------------------
# 注册到 AstrBot 提供商注册表（幂等，兼容插件热重载）
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_TMPL: dict[str, Any] = {
    "id": "openai_chat_completion_rotate",
    "provider": "openai",
    "type": PROVIDER_TYPE,
    "provider_type": "chat_completion",
    "enable": True,
    "key": [],
    "api_base": "https://api.openai.com/v1",
    "timeout": 60,
    "proxy": "",
    "custom_headers": {},
    # 本插件扩展字段（仅当 rotate_override 为 true 时才会覆盖插件全局配置）
    "rotate_override": False,
    "rotate_mode": "round_robin",
    "rotate_on_timeout": True,
    "rotate_on_429": True,
    "switch_retries": 1,
    "cooldown_seconds": 60,
    "cooldown_threshold": 2,
}


def _register() -> None:
    meta = provider_cls_map.get(PROVIDER_TYPE)
    if meta is None:
        register_provider_adapter(
            PROVIDER_TYPE,
            "OpenAI API Chat Completion 提供商适配器（多 Key 轮询 / 超时换 Key）",
            provider_display_name=PROVIDER_DISPLAY_NAME,
            default_config_tmpl=dict(DEFAULT_CONFIG_TMPL),
        )(RotatingOpenAIProvider)
        logger.info("[Key轮询] 已注册提供商类型：%s", PROVIDER_TYPE)
    else:
        # 插件热重载时模块被重新 import，这里只更新类引用，避免命名冲突报错
        meta.cls_type = RotatingOpenAIProvider
        meta.default_config_tmpl = dict(DEFAULT_CONFIG_TMPL)
        meta.provider_display_name = PROVIDER_DISPLAY_NAME
        logger.info("[Key轮询] 提供商类型已热更新：%s", PROVIDER_TYPE)


_register()
