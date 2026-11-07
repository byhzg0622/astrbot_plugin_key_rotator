"""AstrBot 插件：同一服务商多 API Key 轮询（第一次用 1、第二次用 2……循环）。

本模块负责四件事（轮换核心在 provider.py）：

1. 把配置同步进 provider.PLUGIN_SETTINGS，让改动**立即生效**，不用重启；
2. 注册面板 Web API（status / settings / reset），供网页面板通过桥调用；
3. 提供 /keyrot 命令，聊天里也能看状态；
4. 面板设置持久化到插件数据目录 settings.json（优先级高于 AstrBot 插件配置）。

面板入口：插件管理 -> 本插件 -> 页面（pages/console/index.html）。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.web import error_response, json_response, request as web_request

from . import provider as provider_mod
from .provider import PROVIDER_TYPE, RotatingOpenAIProvider

PLUGIN_NAME = "astrbot_plugin_key_rotator"

_BOOL_KEYS = ("enable", "log_key_usage", "stats_command", "rotate_on_timeout", "rotate_on_429")
_STRING_KEYS = ("rotate_mode", "backup_provider_id")
_INT_KEYS = (
    "switch_retries",
    "cooldown_seconds",
    "cooldown_threshold",
    "rate_limit_per_min",
)
_INT_RANGE = {
    "switch_retries": (0, 5),
    "cooldown_seconds": (0, 600),
    "cooldown_threshold": (1, 10),
    "rate_limit_per_min": (0, 100000),  # 0 = 不限速
}
_SETTING_KEYS = _BOOL_KEYS + _STRING_KEYS + _INT_KEYS


def _coerce(key: str, value: Any) -> Any:
    if key in _BOOL_KEYS:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if key in _STRING_KEYS:
        text = str(value or "").strip()
        if key == "rotate_mode" and text not in {"round_robin", "random"}:
            return "round_robin"
        return text
    try:
        number = int(value)
    except (TypeError, ValueError):
        return provider_mod.PLUGIN_SETTINGS.get(key)
    low, high = _INT_RANGE.get(key, (0, 10**9))
    return max(low, min(number, high))

def _int_of(value: Any, default: int) -> int:
    """宽容取整（供线路/原生配置字段读写用），失败返回默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class KeyRotatorPlugin(Star):
    """多 Key 轮询插件主类：配置 + 面板接口 + 聊天命令。"""

    def __init__(self, context: Context, config: Any = None) -> None:
        super().__init__(context, config)
        # Star 基类只保存 context/logger，不会保存 config，必须在这里自己接住
        self.config = config
        self._data_dir = self._resolve_data_dir()
        self._file_settings = self._load_file_settings()
        self._overrides = self._load_overrides()
        self._sync_settings()
        self._register_routes()

    # ------------------------------------------------------------------
    # 数据目录与持久化
    # ------------------------------------------------------------------
    def _resolve_data_dir(self) -> Path:
        """数据目录：优先官方插件数据目录（防更新覆盖），回落插件内 data/。"""
        try:
            return Path(StarTools.get_data_dir(PLUGIN_NAME))
        except Exception:  # noqa: BLE001
            return Path(__file__).resolve().parent / "data"

    @property
    def _settings_file(self) -> Path:
        return self._data_dir / "settings.json"

    def _load_file_settings(self) -> dict:
        try:
            raw = self._settings_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if k in _SETTING_KEYS}
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 读取 settings.json 失败，使用插件配置： %s", PLUGIN_NAME, exc)
        return {}

    def _save_file_settings(self) -> bool:
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._settings_file.write_text(
                json.dumps(self._file_settings, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 写入 settings.json 失败：%s", PLUGIN_NAME, exc)
            return False

    # ------------------------------------------------------------------
    # 覆盖 Key：面板下发，直接替换某个模型每次调用使用的 API Key
    # ------------------------------------------------------------------
    @property
    def _keys_file(self) -> Path:
        return self._data_dir / "key_overrides.json"

    def _load_overrides(self) -> dict:
        try:
            data = json.loads(self._keys_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 读取 key_overrides.json 失败：%s", PLUGIN_NAME, exc)
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, Any] = {}
        for mid, item in data.items():
            if not isinstance(item, dict):
                continue
            keys = item.get("keys")
            if not isinstance(keys, list):
                single = item.get("key")
                keys = [single] if isinstance(single, str) else []
            keys = [str(k).strip() for k in keys if str(k or "").strip()]
            if keys:
                out[str(mid)] = {"keys": keys, "at": item.get("at", 0)}
        return out

    def _save_overrides(self) -> bool:
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._keys_file.write_text(
                json.dumps(self._overrides, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 写入 key_overrides.json 失败：%s", PLUGIN_NAME, exc)
            return False

    def _inst_map(self) -> dict:
        """实例表：模型 id -> 提供商实例（隔壁分组配置管家的同款主源）。"""
        try:
            pm = getattr(self.context, "provider_manager", None)
            inst_map = dict(getattr(pm, "inst_map", None) or {})
        except Exception:  # noqa: BLE001
            inst_map = {}
        if not inst_map:
            inst_map = {}
            for p in self.context.get_all_providers() or []:
                pid = getattr(p, "id", None) or (p.provider_config or {}).get("id", "")
                if pid:
                    inst_map[str(pid)] = p
        return inst_map

    def _apply_overrides(self) -> int:
        """把覆盖 Key 下发到实例（幂等；实例被 WebUI 重载后下条消息会重新接管）。"""
        if not self._overrides:
            return 0
        inst_map = self._inst_map()
        applied = 0
        for model_id, item in self._overrides.items():
            inst = inst_map.get(model_id)
            if inst is None:
                continue
            keys = item.get("keys") or [item.get("key")]
            if provider_mod.adopt_key_override(inst, keys):
                applied += 1
        return applied

    def _release_override(self, model_id: str) -> bool:
        inst = self._inst_map().get(model_id)
        if inst is None:
            return False
        return provider_mod.release_key_override(inst)

    @staticmethod
    def _ov_keys(item: dict | None) -> list[str]:
        """兼容旧的单 key 格式，统一返回 Key 列表。"""
        if not item:
            return []
        keys = item.get("keys")
        if not isinstance(keys, list):
            single = item.get("key")
            keys = [single] if single else []
        return [str(k) for k in keys if k]

    @staticmethod
    def _split_keys(raw: str) -> list[str]:
        """把粘贴的文本切成 Key 列表（支持换行 / 中英文逗号分号 / 空白分隔）。"""
        out: list[str] = []
        for part in re.split(r"[,，;；\s]+", str(raw or "").strip()):
            part = part.strip()
            if part and part not in out:
                out.append(part)
        return out

    def _overrides_status(self) -> dict:
        inst_map = self._inst_map()
        out: dict[str, Any] = {}
        for model_id, item in self._overrides.items():
            inst = inst_map.get(model_id)
            ring = getattr(inst, "api_keys", None) if inst is not None else None
            keys = [k for k in (item.get("keys") or [item.get("key")]) if k]
            active = (
                isinstance(ring, provider_mod.RotatingKeyRing)
                and bool(ring.override_keys)
                and list(ring.override_keys) == keys
            )
            keys = item.get("keys") or [item.get("key")]
            keys = [k for k in keys if k]
            out[model_id] = {
                "keys": [provider_mod.mask_key(k) for k in keys],
                "count": len(keys),
                "at": item.get("at", 0),
                "instantiated": inst is not None,
                "active": bool(active),
            }
        return out

    # ------------------------------------------------------------------
    # 配置同步（AstrBot 插件配置为底，面板 settings.json 覆盖）
    # ------------------------------------------------------------------
    def _cfg_get(self, key: str, default: Any = None) -> Any:
        cfg = getattr(self, "config", None)
        if cfg is None or not hasattr(cfg, "get"):
            return default
        try:
            value = cfg.get(key, default)
        except Exception:  # noqa: BLE001
            return default
        return default if value is None else value

    def _sync_settings(self) -> None:
        merged = {k: self._cfg_get(k) for k in _SETTING_KEYS}
        merged.update(self._file_settings)
        settings = provider_mod.PLUGIN_SETTINGS
        for key in _SETTING_KEYS:
            value = merged.get(key)
            if value is None:
                continue
            settings[key] = _coerce(key, value)

    def _apply_settings(self, incoming: dict) -> dict:
        """校验并应用面板提交的设置，返回应用后的完整设置。"""
        accepted = {}
        for key, value in incoming.items():
            if key not in _SETTING_KEYS:
                continue
            accepted[key] = _coerce(key, value)
            self._file_settings[key] = accepted[key]
            provider_mod.PLUGIN_SETTINGS[key] = accepted[key]
        self._save_file_settings()
        return dict(provider_mod.PLUGIN_SETTINGS)

    # ------------------------------------------------------------------
    # 面板 Web API
    # ------------------------------------------------------------------
    def _register_routes(self) -> None:
        register = getattr(self.context, "register_web_api", None)
        if not callable(register):
            logger.warning("[%s] 当前 AstrBot 无 context.register_web_api，网页面板不可用", PLUGIN_NAME)
            return
        routes = [
            ("status", self.api_status, "轮询状态"),
            ("settings", self.api_settings, "读取/保存设置"),
            ("reset", self.api_reset, "清空统计与事件"),
            ("keys", self.api_keys, "覆盖 Key 列表"),
            ("set_key", self.api_set_key, "设置覆盖 Key"),
            ("clear_key", self.api_clear_key, "撤销覆盖 Key"),
            ("set_type", self.api_set_type, "一键切换线路类型"),
            ("set_source", self.api_set_source, "修改线路参数（超时/地址/代理）"),
            ("delete_model", self.api_delete_model, "删除模型（退出轮换）"),
            ("restore_model", self.api_restore_model, "恢复模型"),
        ]
        ok = 0
        for name, handler, desc in routes:
            try:
                register(f"/{PLUGIN_NAME}/{name}", handler, ["GET", "POST"], f"Key轮询：{desc}")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] 注册面板接口 %s 失败：%s", PLUGIN_NAME, name, exc)
        try:
            register(
                f"/{PLUGIN_NAME}/<path:rest>",
                self._api_fallback,
                ["GET", "POST"],
                "Key轮询：兜底",
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 注册兜底路由失败：%s", PLUGIN_NAME, exc)
        logger.info("[%s] 面板接口已注册 %d 个", PLUGIN_NAME, ok)

    async def _params(self) -> dict:
        """合并 query 与 JSON body。"""
        merged: dict[str, Any] = {}
        if web_request is None:
            return merged
        try:
            for key in list(web_request.query.keys()):
                merged[key] = web_request.query.get(key)
        except Exception:  # noqa: BLE001
            pass
        try:
            body = await web_request.json()
        except Exception:  # noqa: BLE001
            body = None
        if isinstance(body, dict):
            for key, value in body.items():
                merged.setdefault(key, value)
        return merged

    async def _api_fallback(self, rest: str = "") -> Any:
        head = (rest or "").split("/", 1)[0]
        mapping = {
            "status": self.api_status,
            "settings": self.api_settings,
            "reset": self.api_reset,
            "keys": self.api_keys,
            "set_key": self.api_set_key,
            "clear_key": self.api_clear_key,
            "set_type": self.api_set_type,
            "set_source": self.api_set_source,
            "delete_model": self.api_delete_model,
            "restore_model": self.api_restore_model,
        }
        handler = mapping.get(head)
        if handler is None:
            return error_response(f"未知的面板接口：{rest}", status_code=404)
        return await handler()

    def _iter_rotating(self):
        try:
            providers = self.context.get_all_providers() or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 读取提供商列表失败：%s", PLUGIN_NAME, exc)
            return []
        return [p for p in providers if isinstance(p, RotatingOpenAIProvider)]

    def _read_model_config(self) -> dict:
        """学隔壁（分组配置管家 api_pools）：读出全部线路与模型配置。

        读的是**配置**而非实例，因此连 enable=false、type 没改成轮询型、
        甚至尚未实例化的条目都能看到；再用实例表标注"当前是否真的在轮换"。
        """
        sources_out: list[dict] = []
        models_out: list[dict] = []

        cfg = None
        try:
            get_config = getattr(self.context, "get_config", None)
            if callable(get_config):
                cfg = get_config()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] 读取 AstrBot 配置失败：%s", PLUGIN_NAME, exc)

        sources: list[dict] = []
        models: list[dict] = []
        if cfg is not None and hasattr(cfg, "get"):
            try:
                sources = list(cfg.get("provider_sources") or [])
                models = list(cfg.get("provider") or [])
            except Exception as exc:  # noqa: BLE001
                logger.debug("[%s] 解析模型配置失败：%s", PLUGIN_NAME, exc)

        # 实例表：id -> 实例（隔壁的主源，保证与底层 set_provider 同源）
        inst_map: dict = {}
        try:
            pm = getattr(self.context, "provider_manager", None)
            inst_map = dict(getattr(pm, "inst_map", None) or {})
        except Exception:  # noqa: BLE001
            inst_map = {}
        if not inst_map:
            inst_map = {
                getattr(p, "id", "") or (p.provider_config or {}).get("id", ""): p
                for p in (self.context.get_all_providers() or [])
            }

        rotating_source_ids = {
            str((p.provider_config or {}).get("provider_source_id") or "")
            for p in self._iter_rotating()
        }

        for src in sources:
            sid = str(src.get("id") or "")
            stype = str(src.get("type") or "")
            keys = src.get("key") or []
            key_count = len(keys) if isinstance(keys, list) else (1 if keys else 0)
            sources_out.append(
                {
                    "id": sid,
                    "type": stype,
                    "rotating": stype == PROVIDER_TYPE,
                    "instance_rotating": sid in rotating_source_ids,
                    "enable": bool(src.get("enable", True)),
                    "api_base": str(src.get("api_base") or ""),
                    "timeout": src.get("timeout", ""),
                    "proxy": str(src.get("proxy") or ""),
                    "key_count": key_count,
                    "model_count": sum(
                        1 for m in models if m.get("provider_source_id") == sid
                    ),
                }
            )

        src_by_id = {str(s.get("id") or ""): s for s in sources}
        for m in models:
            mid = str(m.get("id") or "")
            sid = str(m.get("provider_source_id") or "")
            src = src_by_id.get(sid, {})
            ov = self._overrides.get(mid)
            ov_keys = self._ov_keys(ov)
            inst = inst_map.get(mid)
            ring = getattr(inst, "api_keys", None) if inst is not None else None
            models_out.append(
                {
                    "id": mid,
                    "model": str(m.get("model") or ""),
                    "source_id": sid,
                    "source_type": str(src.get("type") or ""),
                    "rotating": sid in rotating_source_ids,
                    "instantiated": any(
                        key and (key == mid or key == sid) for key in inst_map
                    ),
                    "enable": bool(m.get("enable", True)),
                    "deleted": not bool(m.get("enable", True)),
                    "override": [provider_mod.mask_key(k) for k in ov_keys],
                    "override_count": len(ov_keys),
                    "override_active": bool(
                        ov_keys
                        and isinstance(ring, provider_mod.RotatingKeyRing)
                        and list(ring.override_keys) == ov_keys
                    ),
                }
            )

        return {"sources": sources_out, "models": models_out}

    def _build_status(self) -> dict:
        lines = []
        totals = {"keys": 0, "assigned": 0, "ok": 0, "failed": 0, "cooling": 0}
        for prov in self._iter_rotating():
            ring = getattr(prov, "key_ring", None)
            if ring is None:
                continue
            info = ring.describe()
            cfg = prov.provider_config or {}
            keys = info["keys"]
            line = {
                "model": prov.get_model(),
                "api_base": str(cfg.get("api_base", "")),
                "timeout": cfg.get("timeout", ""),
                "enable": bool(cfg.get("enable", True)),
                "cfg_override": bool(cfg.get("rotate_override", False)),
                **info,
            }
            lines.append(line)
            totals["keys"] += info["total"]
            totals["assigned"] += sum(k["assigned"] for k in keys)
            totals["ok"] += sum(k["ok"] for k in keys)
            totals["failed"] += sum(k["fail"] for k in keys)
            totals["cooling"] += sum(1 for k in keys if k["cooldown"] > 0)

        events = list(provider_mod.EVENTS)
        events.reverse()
        model_config = self._read_model_config()
        return {
            "ok": True,
            "plugin": PLUGIN_NAME,
            "registered": PROVIDER_TYPE in provider_mod.provider_cls_map,
            "provider_type": PROVIDER_TYPE,
            "settings": dict(provider_mod.PLUGIN_SETTINGS),
            "agent": self._read_agent_config(),
            "lines": lines,
            "model_config": model_config,
            "warnings": self._diagnose(model_config, len(lines)),
            "overrides": self._overrides_status(),
            "events": events[:80],
            "totals": {
                **totals,
                "lines": len(lines),
                "rotating": bool(provider_mod.PLUGIN_SETTINGS.get("enable", True)),
            },
        }

    def _diagnose(self, model_config: dict, rotating_lines: int) -> list[str]:
        """解释「配置」与「实例」不一致的情况——避免面板自相矛盾。"""
        warnings: list[str] = []
        for src in model_config.get("sources", []):
            sid = src.get("id", "?")
            kc = int(src.get("key_count") or 0)
            cfg_rot = bool(src.get("rotating"))
            inst_rot = bool(src.get("instance_rotating"))
            if cfg_rot and not inst_rot:
                warnings.append(
                    f"线路 {sid} 配置里已标成轮询型，但运行中的实例还是原生型——"
                    "在 WebUI 保存一次该服务商（或重启 AstrBot）后才会真正开始轮换。"
                )
            elif inst_rot and not cfg_rot:
                warnings.append(
                    f"线路 {sid} 的实例是轮询型，但配置类型已改回原生型——"
                    "保存一次该服务商让两者一致。"
                )
            elif not cfg_rot and kc > 1:
                warnings.append(
                    f"线路 {sid} 有 {kc} 个 Key，但类型是「原生型」，"
                    "这 {kc} 个 Key **不会轮换**。把类型改成"
                    "「OpenAI Compatible（多 Key 轮询）」即可开启；"
                    "或者在下方模型行「设置」手动填充 Key 池（不用改类型）。"
                )
            if kc == 0:
                warnings.append(
                    f"线路 {sid} 配置里没有任何 Key，直接调用会鉴权失败——"
                    "可在下方模型行「设置」手动填充 Key 池。"
                )
            rate = int(provider_mod.PLUGIN_SETTINGS.get("rate_limit_per_min", 0) or 0)
            if rate > 0 and kc > 0 and rate // kc == 0:
                warnings.append(
                    f"限速 {rate}/分钟 小于线路 {sid} 的 Key 数（{kc}），"
                    f"均分后每 Key 额度为 0，该线路会一直被移交给备用模型——"
                    "请调高限速或删掉几个模型。"
                )
        if rotating_lines == 0 and not warnings:
            warnings.append(
                "当前没有任何轮询型线路。把某条线路的「类型」改成"
                "「OpenAI Compatible（多 Key 轮询）」，或在下方模型行手动填充 Key 池。"
            )
        return warnings

    async def api_status(self) -> Any:
        return json_response(self._build_status())

    # ------------------------------------------------------------------
    # AstrBot 原生参数（agent_runner）：底层重试次数与回退链
    # ------------------------------------------------------------------
    def _read_agent_config(self) -> dict:
        """读 AstrBot 原生 agent_runner 配置（只读快照，供面板配对展示/编辑）。"""
        try:
            from astrbot.core import astrbot_config

            agent_cfg = astrbot_config.get("agent_runner") or {}
            cfg = agent_cfg.get("config") if isinstance(agent_cfg, dict) else None
            model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
            misc = cfg.get("misc") if isinstance(cfg, dict) else None
            model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
            misc = misc if isinstance(misc, dict) else {}
            return {
                "request_max_retries": _int_of(
                    model_cfg.get("request_max_retries"), 5
                ),
                "main_provider_id": str(model_cfg.get("provider_id") or ""),
                "fallback_chain": list(model_cfg.get("fallback_provider_ids") or []),
                "tool_call_timeout": _int_of(misc.get("tool_call_timeout"), 120),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] 读取 agent_runner 配置失败：%s", PLUGIN_NAME, exc)
            return {}

    async def _apply_agent_params(self, incoming: dict) -> dict:
        """把面板提交的 AstrBot 原生参数写回 cmd_config.json（保存即生效）。

        目前支持 request_max_retries（Agent Runner 的「异常重试次数」，
        即底层同 Key 传输重试，AstrBot 原生字段，含首次、最小 1）。
        它与本插件的「换 Key 重试次数」是两层重试，配对关系为：
            最坏耗时 ≈ 超时 × request_max_retries ×（换 Key 次数 + 1）
        """
        result: dict[str, Any] = {}
        if "request_max_retries" not in incoming:
            return result
        try:
            value = int(float(incoming.get("request_max_retries")))
        except (TypeError, ValueError):
            return {"error": "底层重试次数必须是整数"}
        # AstrBot 归一化时会把 <1 钳到 1，这里保持一致（0 也按 1 处理）
        value = max(1, min(value, 100))
        try:
            from astrbot.core import astrbot_config

            agent_cfg = astrbot_config.get("agent_runner") or {}
            cfg = agent_cfg.get("config") if isinstance(agent_cfg, dict) else None
            if not isinstance(cfg, dict):
                return {"error": "找不到 agent_runner.config 配置"}
            model_cfg = cfg.get("model")
            if not isinstance(model_cfg, dict):
                return {"error": "找不到 agent_runner.config.model 配置"}
            old = model_cfg.get("request_max_retries")
            model_cfg["request_max_retries"] = value
            if _int_of(old, -1) != value:
                await astrbot_config.save_config_async()
            result.update(
                {"request_max_retries": value, "old": old, "saved": True}
            )
            logger.info(
                "[%s] 底层请求重试次数：%s -> %s（原生 agent_runner 配置）",
                PLUGIN_NAME,
                old,
                value,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 写入 agent_runner 重试次数失败：%s", PLUGIN_NAME, exc)
            result["error"] = str(exc)
        return result

    async def _apply_backup_chain(self, settings: dict) -> dict:
        """把「最后兜底」模型追加到回退链末位，保证全报错时一定有备用可转交。

        回退链配置每次请求都会现读，因此写完立即生效，无需重启。
        """
        backup = str(settings.get("backup_provider_id") or "").strip()
        result: dict[str, Any] = {"backup": backup, "added": False, "chain": []}
        if not backup:
            return result

        from astrbot.core import astrbot_config

        try:
            agent_cfg = astrbot_config.get("agent_runner") or {}
            model_cfg = (agent_cfg.get("config") or {}).get("model")
            if not isinstance(model_cfg, dict):
                result["error"] = "找不到 agent_runner.config.model 配置"
                return result
            chain = list(model_cfg.get("fallback_provider_ids") or [])
            result["chain"] = chain
            if backup == str(model_cfg.get("provider_id") or ""):
                result["note"] = "该模型就是主模型，无需加入回退链"
                return result
            ids = {str(m.get("id")) for m in (astrbot_config.get("provider") or [])}
            if backup not in ids:
                result["error"] = f"找不到模型：{backup}"
                return result
            if backup in chain:
                result["note"] = "已在回退链中"
                return result
            chain.append(backup)
            model_cfg["fallback_provider_ids"] = chain
            await astrbot_config.save_config_async()
            result["added"] = True
            result["chain"] = chain
            logger.info(
                "[%s] 已把 %s 追加为回退链末位兜底（当前链：%s）",
                PLUGIN_NAME,
                backup,
                " -> ".join(chain),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 写入兜底回退链失败：%s", PLUGIN_NAME, exc)
            result["error"] = str(exc)
        return result

    async def api_settings(self) -> Any:
        params = await self._params()
        if params:
            # 原生底层重试次数（agent_runner）与插件设置同一次提交、一起保存
            agent_result = await self._apply_agent_params(params)
            settings = self._apply_settings(params)
            backup_result = await self._apply_backup_chain(settings)
            logger.info(
                "[%s] 面板保存设置：%s（原生参数：%s）",
                PLUGIN_NAME,
                settings,
                agent_result,
            )
            return json_response(
                {
                    "ok": True,
                    "settings": settings,
                    "backup": backup_result,
                    "agent": agent_result,
                    "agent_config": self._read_agent_config(),
                }
            )
        return json_response(
            {
                "ok": True,
                "settings": dict(provider_mod.PLUGIN_SETTINGS),
                "agent_config": self._read_agent_config(),
            }
        )

    async def api_keys(self) -> Any:
        return json_response({"ok": True, "overrides": self._overrides_status()})

    async def api_set_key(self) -> Any:
        params = await self._params()
        model_id = str(params.get("model_id") or "").strip()
        raw = str(params.get("keys") or params.get("key") or "")
        if not model_id:
            return error_response("缺少 model_id", status_code=400)
        keys = self._split_keys(raw)
        if not keys:
            return error_response(
                "请粘贴要使用的 API Key（每行一个，也支持逗号/空格分隔）",
                status_code=400,
            )
        short = [i + 1 for i, k in enumerate(keys) if len(k) < 8]
        if short:
            return error_response(
                f"第 {short[0]} 个 Key 太短（{len(keys[short[0] - 1])} 字符），"
                "请粘贴完整的 API Key",
                status_code=400,
            )

        self._overrides[model_id] = {"keys": keys, "at": int(time.time())}
        self._save_overrides()
        applied = self._apply_overrides()
        logger.info(
            "[%s] 为 %s 设置手动 Key 池（%d 个：%s）；下发状态：%s",
            PLUGIN_NAME,
            model_id,
            len(keys),
            "、".join(provider_mod.mask_key(k) for k in keys),
            "已生效" if applied else "实例未加载，将在下一条消息时接管",
        )
        return json_response(
            {
                "ok": True,
                "applied": applied > 0,
                "count": len(keys),
                "overrides": self._overrides_status(),
            }
        )

    async def api_clear_key(self) -> Any:
        params = await self._params()
        model_id = str(params.get("model_id") or "").strip()
        if not model_id:
            return error_response("缺少 model_id", status_code=400)
        existed = model_id in self._overrides
        self._overrides.pop(model_id, None)
        self._save_overrides()
        released = self._release_override(model_id)
        logger.info("[%s] 撤销 %s 的覆盖 Key（存在=%s，已释放=%s）",
                    PLUGIN_NAME, model_id, existed, released)
        return json_response(
            {
                "ok": True,
                "removed": existed,
                "released": released,
                "overrides": self._overrides_status(),
            }
        )

    # ------------------------------------------------------------------
    # 配置写入：一键改类型 / 删除·恢复模型（唯一允许写 cmd_config 的地方）
    # ------------------------------------------------------------------
    async def _persist_and_reload(self, models: list[dict]) -> bool:
        """落盘 + 重建受影响的模型实例。"""
        from astrbot.core import astrbot_config

        try:
            await astrbot_config.save_config_async()
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 配置落盘失败：%s", PLUGIN_NAME, exc)
            return False

        pm = getattr(self.context, "provider_manager", None)
        if pm is None or not hasattr(pm, "reload"):
            logger.warning("[%s] 无法重建实例：provider_manager 不可用", PLUGIN_NAME)
            return False
        for model in models:
            try:
                await pm.reload(model)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] 重建模型实例 %s 失败：%s",
                    PLUGIN_NAME,
                    model.get("id"),
                    exc,
                )
        return True

    async def api_set_type(self) -> Any:
        """一键切换线路类型（原生型 <-> 轮询型）。"""
        from astrbot.core import astrbot_config

        params = await self._params()
        source_id = str(params.get("source_id") or "").strip()
        target = str(params.get("type") or "").strip() or PROVIDER_TYPE
        if not source_id:
            return error_response("缺少 source_id", status_code=400)
        if target not in {PROVIDER_TYPE, "openai_chat_completion"}:
            return error_response(f"不支持的目标类型：{target}", status_code=400)
        if target == PROVIDER_TYPE and PROVIDER_TYPE not in provider_mod.provider_cls_map:
            return error_response(
                "插件未注册轮询类型（插件可能被禁用），无法切换",
                status_code=400,
            )

        sources = astrbot_config.get("provider_sources") or []
        source = next(
            (s for s in sources if str(s.get("id") or "") == source_id), None
        )
        if source is None:
            return error_response(f"找不到线路：{source_id}", status_code=404)

        old = str(source.get("type") or "")
        if old == target:
            return json_response(
                {"ok": True, "changed": False, "source_id": source_id, "type": target}
            )

        source["type"] = target
        affected = [
            m
            for m in (astrbot_config.get("provider") or [])
            if str(m.get("provider_source_id") or "") == source_id
        ]
        saved = await self._persist_and_reload(affected)
        logger.info(
            "[%s] 线路 %s 类型：%s -> %s（影响 %d 个模型，落盘=%s）",
            PLUGIN_NAME,
            source_id,
            old,
            target,
            len(affected),
            saved,
        )
        return json_response(
            {
                "ok": True,
                "changed": True,
                "saved": saved,
                "source_id": source_id,
                "type": target,
                "reloaded": len(affected),
            }
        )

    async def api_set_source(self) -> Any:
        """修改线路参数（服务商原生字段）：写 cmd_config.json + 重建受影响实例。

        支持字段（白名单，均为 AstrBot 服务商原生配置）：
        - timeout：超时秒数（1~3600），与插件的「换 Key 重试次数」配对，
          决定最坏耗时 ≈ 超时 × 底层重试 ×（换 Key 次数 + 1）
        - api_base：接口地址（必须 http/https）
        - proxy：代理地址（可清空）
        - enable：线路启停

        与 set_type 相同的写路径：改内存配置 -> save_config_async 落盘 ->
        provider_manager.reload 重建该线路的模型实例，无需重启 AstrBot。
        """
        from astrbot.core import astrbot_config

        params = await self._params()
        source_id = str(params.get("source_id") or "").strip()
        if not source_id:
            return error_response("缺少 source_id", status_code=400)

        sources = astrbot_config.get("provider_sources") or []
        source = next(
            (s for s in sources if str(s.get("id") or "") == source_id), None
        )
        if source is None:
            return error_response(f"找不到线路：{source_id}", status_code=404)

        changed: dict[str, Any] = {}

        # 超时（秒）：整数 1~3600；null/空串表示本次不改（0 是非法值，要报错）
        raw_timeout = params.get("timeout")
        if raw_timeout is not None and str(raw_timeout).strip() != "":
            try:
                timeout = int(float(raw_timeout))
            except (TypeError, ValueError):
                return error_response("超时必须是数字（秒）", status_code=400)
            if not 1 <= timeout <= 3600:
                return error_response("超时需在 1~3600 秒之间", status_code=400)
            if _int_of(source.get("timeout"), 0) != timeout:
                source["timeout"] = timeout
                changed["timeout"] = timeout

        # 接口地址：非空且必须 http/https
        if "api_base" in params and str(params.get("api_base") or "").strip() != "":
            api_base = str(params.get("api_base")).strip().rstrip("/")
            if not api_base.startswith(("http://", "https://")):
                return error_response(
                    "接口地址必须以 http:// 或 https:// 开头", status_code=400
                )
            if str(source.get("api_base") or "").strip().rstrip("/") != api_base:
                source["api_base"] = api_base
                changed["api_base"] = api_base

        # 代理：允许清空（传空串 = 移除代理）
        if "proxy" in params:
            proxy = str(params.get("proxy") or "").strip()
            if str(source.get("proxy") or "") != proxy:
                source["proxy"] = proxy
                changed["proxy"] = proxy

        # 线路启停
        if "enable" in params:
            raw_enable = params.get("enable")
            enable = (
                raw_enable
                if isinstance(raw_enable, bool)
                else str(raw_enable).strip().lower() in {"1", "true", "yes", "on"}
            )
            if bool(source.get("enable", True)) != enable:
                source["enable"] = enable
                changed["enable"] = enable

        if not changed:
            return json_response(
                {"ok": True, "changed": False, "source_id": source_id}
            )

        affected = [
            m
            for m in (astrbot_config.get("provider") or [])
            if str(m.get("provider_source_id") or "") == source_id
        ]
        saved = await self._persist_and_reload(affected)
        logger.info(
            "[%s] 线路 %s 参数更新：%s（影响 %d 个模型，落盘=%s）",
            PLUGIN_NAME,
            source_id,
            changed,
            len(affected),
            saved,
        )
        return json_response(
            {
                "ok": True,
                "changed": True,
                "saved": saved,
                "source_id": source_id,
                "changes": changed,
                "reloaded": len(affected),
            }
        )

    async def _set_model_enabled(self, enable: bool) -> Any:
        from astrbot.core import astrbot_config

        params = await self._params()
        model_id = str(params.get("model_id") or "").strip()
        if not model_id:
            return error_response("缺少 model_id", status_code=400)

        models = astrbot_config.get("provider") or []
        model = next(
            (m for m in models if str(m.get("id") or "") == model_id), None
        )
        if model is None:
            return error_response(f"找不到模型：{model_id}", status_code=404)

        was = bool(model.get("enable", True))
        if was == bool(enable):
            return json_response(
                {"ok": True, "changed": False, "model_id": model_id, "enable": enable}
            )

        model["enable"] = bool(enable)
        if not enable:
            # 删除：退出轮换 + 丢掉手动 Key 池（配置仍保留在后台，可随时恢复）
            self._overrides.pop(model_id, None)
            self._save_overrides()
            self._release_override(model_id)

        saved = await self._persist_and_reload([model])
        action = "启用" if enable else "删除"
        logger.info(
            "[%s] %s模型 %s（落盘=%s，实例已重建）",
            PLUGIN_NAME,
            action,
            model_id,
            saved,
        )
        return json_response(
            {
                "ok": True,
                "changed": True,
                "saved": saved,
                "model_id": model_id,
                "enable": bool(enable),
            }
        )

    async def api_delete_model(self) -> Any:
        """删除模型（停用）：退出轮换、面板不再显示，但配置保留在后台。"""
        return await self._set_model_enabled(False)

    async def api_restore_model(self) -> Any:
        """从回收站恢复模型：重新启用并重建实例。"""
        return await self._set_model_enabled(True)

    @filter.on_waiting_llm_request()
    async def apply_key_overrides(self, event: AstrMessageEvent) -> None:
        """选中提供商、发出请求之前，确保覆盖 Key 已经接管实例。

        放在这个钩子里（internal.py:217，早于 _select_provider），
        可以保证：WebUI 重建实例、插件热重载之后，下一条消息就能重新接管。
        """
        if not self._overrides:
            return
        try:
            self._apply_overrides()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 下发覆盖 Key 失败：%s", PLUGIN_NAME, exc)

    async def api_reset(self) -> Any:
        for prov in self._iter_rotating():
            ring = getattr(prov, "key_ring", None)
            if ring is None:
                continue
            for st in ring.stats.values():
                st.update(
                    {
                        "assigned": 0,
                        "ok": 0,
                        "fail": 0,
                        "consecutive_fail": 0,
                        "cooldown_until": 0.0,
                        "last_error": "",
                    }
                )
            ring._i = 0
        provider_mod.EVENTS.clear()
        logger.info("[%s] 已清空轮换统计与事件流", PLUGIN_NAME)
        return json_response({"ok": True})

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        self._sync_settings()
        registered = PROVIDER_TYPE in provider_mod.provider_cls_map
        self.logger.info(
            "[%s] 已加载，提供商类型注册=%s，轮换=%s，策略=%s，超时换Key=%s，面板数据目录=%s",
            PLUGIN_NAME,
            "正常" if registered else "异常（未找到注册表条目）",
            "开" if provider_mod.PLUGIN_SETTINGS.get("enable") else "关",
            provider_mod.PLUGIN_SETTINGS.get("rotate_mode"),
            "开" if provider_mod.PLUGIN_SETTINGS.get("rotate_on_timeout") else "关",
            self._data_dir,
        )
        if not registered:
            self.logger.warning(
                "[%s] 提供商类型未注册，轮询型服务商将无法加载。", PLUGIN_NAME
            )

    async def terminate(self) -> None:
        self.logger.info("[%s] 已卸载", PLUGIN_NAME)

    # ------------------------------------------------------------------
    # /keyrot 状态命令
    # ------------------------------------------------------------------
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("keyrot", alias={"密钥轮询", "keyrotate"})
    async def keyrot(self, event: AstrMessageEvent) -> None:
        """查看多 API Key 轮询状态"""
        if not bool(provider_mod.PLUGIN_SETTINGS.get("stats_command", True)):
            return

        status = self._build_status()
        s = status["settings"]
        head = [
            "## API Key 轮询状态",
            f"总开关：{'开启' if s.get('enable') else '关闭（退回原生随机取 Key）'}",
            f"策略：{'顺序轮询' if s.get('rotate_mode') == 'round_robin' else '随机'}"
            f"｜超时换Key：{'开' if s.get('rotate_on_timeout') else '关'}"
            f"｜429换Key：{'开' if s.get('rotate_on_429') else '关'}"
            f"｜换Key次数：{s.get('switch_retries')}",
            f"冷却：连续失败 {s.get('cooldown_threshold')} 次 -> 冷却 "
            f"{s.get('cooldown_seconds')} 秒",
            "",
        ]
        bodies: list[str] = []
        for line in status["lines"]:
            bodies.append(
                f"### {line['model']}（线路 {line['source_id']}）\n"
                f"{line['api_base']}｜Key {line['total']} 个｜下一个：第 {line['next_index']} 个"
            )
            for item in line["keys"]:
                row = (
                    f"{item['index']}. {item['key']}  "
                    f"分配 {item['assigned']}｜成功 {item['ok']}｜失败 {item['fail']}"
                )
                if item["cooldown"]:
                    row += f"｜⏳冷却 {item['cooldown']}s"
                if item["fail"] and item["last_error"]:
                    row += f"\n    └ {item['last_error']}"
                bodies.append(row)
            bodies.append("")
        if not bodies:
            bodies.append(
                "当前没有使用轮询类型的提供商。\n"
                "请把服务商的类型改成 openai_chat_completion_rotate"
                "（WebUI 服务提供商 -> 编辑 -> 类型），详见插件 README 的「接入方式 A」。"
            )
        event.set_result(MessageEventResult().message("\n".join(head + bodies)))
