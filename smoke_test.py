r"""开发用冒烟测试：验证语法、注册链路与轮换行为。

用法（在 AstrBot 环境下运行）：
    D:\bot\AstrBotLauncher-0.3.0\AstrBot\venv\Scripts\python.exe smoke_test.py

只读校验，不改动任何 AstrBot 配置。
"""
from __future__ import annotations

import asyncio
import json
import time
import os
import sys
import tempfile
from pathlib import Path

# AstrBot 的 get_astrbot_root() 默认取当前工作目录，import 时会在 CWD 下生成
# data\cmd_config.json、t2i_templates 等副产物。这里统一指到临时目录，
# 保证无论从哪里跑，工作区都不会被弄脏。
os.environ.setdefault("ASTRBOT_ROOT", tempfile.mkdtemp(prefix="astrbot-test-root-"))

HERE = Path(__file__).resolve().parent
ASTRBOT_ROOT = HERE.parents[1] / "AstrBot"  # 工作区与 AstrBot 同机时的默认位置
if not (ASTRBOT_ROOT / "astrbot").is_dir():
    ASTRBOT_ROOT = Path(r"D:\bot\AstrBotLauncher-0.3.0\AstrBot")
sys.path.insert(0, str(ASTRBOT_ROOT))
sys.path.insert(0, str(HERE.parent))


def main() -> int:
    import py_compile

    for name in ("main.py", "provider.py"):
        py_compile.compile(str(HERE / name), doraise=True)
        print("[OK] compile", name)

    schema = json.loads((HERE / "_conf_schema.json").read_text(encoding="utf-8"))
    print("[OK] _conf_schema.json:", ", ".join(schema.keys()))

    try:
        import yaml

        meta = yaml.safe_load((HERE / "metadata.yaml").read_text(encoding="utf-8"))
        assert meta.get("name") == HERE.name, "metadata.name 必须与目录名一致"
        print("[OK] metadata.yaml:", meta.get("name"), meta.get("version"))
    except ImportError:
        print("[SKIP] pyyaml 不可用")

    import astrbot_plugin_key_rotator.provider as provider_mod
    from astrbot.core.provider.register import provider_cls_map

    assert provider_mod.PROVIDER_TYPE in provider_cls_map, "类型未注册进 provider_cls_map"
    meta = provider_cls_map[provider_mod.PROVIDER_TYPE]
    print("[OK] 注册类型:", provider_mod.PROVIDER_TYPE)
    print("     类:", meta.cls_type.__name__, "| 父类:", meta.cls_type.__mro__[1].__name__)
    print("     显示名:", meta.provider_display_name)
    print("     配置模板字段:", ", ".join(sorted(meta.default_config_tmpl)))

    import importlib

    importlib.reload(provider_mod)  # 模拟插件热重载重复注册
    print("[OK] 幂等注册：重复 import 不报错")

    import astrbot_plugin_key_rotator.main as main_mod  # noqa: F401
    from astrbot.core.star.star import star_map

    print("[OK] Star 注册:", [k for k in star_map if "key_rotator" in k])

    ring = provider_mod.RotatingKeyRing(
        ["k1", "k2", "k3", "k4", "k5"],
        source_id="agens",
        source_config={},
    )
    provider_mod.PLUGIN_SETTINGS.update({"enable": True, "rotate_mode": "round_robin"})
    seq = [ring.copy()[0] for _ in range(7)]
    assert seq[:5] == ["k1", "k2", "k3", "k4", "k5"], f"顺序轮换不正确: {seq}"
    assert seq[5:] == ["k1", "k2"], f"循环不正确: {seq}"
    print("[OK] 顺序轮换:", seq)

    provider_mod.PLUGIN_SETTINGS["enable"] = False
    disabled = ring.copy()
    assert len(disabled) == 5, "关闭总开关应退回全量随机"
    provider_mod.PLUGIN_SETTINGS["enable"] = True
    print("[OK] 总开关关闭 -> 退回原生全量随机")

    ring.report_failure("k1", TimeoutError("read timeout"))
    ring.report_failure("k1", TimeoutError("read timeout"))
    cooling = [
        x["key"]
        for x in ring.describe()["keys"]
        if x["cooldown"] > 0
    ]
    assert cooling, "连续失败 2 次后应进入冷却"
    after = [ring.copy()[0] for _ in range(5)]
    assert not any(x.startswith("k1") for x in after), f"冷却中的 Key 不应被选中: {after}"
    print("[OK] 冷却生效，冷却期间轮换序列:", after)

    # ---- 面板接口数据面（网页面板靠它渲染） ----
    provider_mod.EVENTS.clear()
    src_cfg = {
        "id": "agens/agnes-2.5-flash",
        "provider_source_id": "agens",
        "type": provider_mod.PROVIDER_TYPE,
        "provider_type": "chat_completion",
        "enable": True,
        "provider": "openai",
        "key": ["k1", "k2", "k3", "k4", "k5"],
        "api_base": "https://api.agnes-ai.cn/v1",
        "timeout": 10,
        "custom_headers": {},
        "model": "agnes-2.5-flash",
    }
    prov = provider_mod.RotatingOpenAIProvider(src_cfg, {})
    assert isinstance(prov.key_ring, provider_mod.RotatingKeyRing), "api_keys 未被替换成环"

    import tempfile

    class _StubCtx:
        """模拟真实 Context：get_all_providers / get_config / register_web_api。"""

        def __init__(self):
            self.routes = []

        def get_all_providers(self):
            return [prov]

        def get_config(self, umo=None):
            return {
                "provider_sources": [
                    {
                        "id": "agens",
                        "type": provider_mod.PROVIDER_TYPE,
                        "provider_type": "chat_completion",
                        "enable": True,
                        "key": ["k1", "k2", "k3", "k4", "k5"],
                        "api_base": "https://api.agnes-ai.cn/v1",
                        "timeout": 10,
                    },
                    {
                        "id": "axis",
                        "type": "openai_chat_completion",
                        "provider_type": "chat_completion",
                        "enable": True,
                        "key": ["sk-only-one"],
                        "api_base": "https://ai.onyxaxis.org/v1",
                        "timeout": 15,
                    },
                ],
                "provider": [
                    {
                        "id": "agens/agnes-2.5-flash",
                        "model": "agnes-2.5-flash",
                        "provider_source_id": "agens",
                        "enable": True,
                    },
                    {
                        "id": "axis/gpt-5.4",
                        "model": "gpt-5.4",
                        "provider_source_id": "axis",
                        "enable": True,
                    },
                ],
            }

        def register_web_api(self, route, handler, methods, desc):
            self.routes.append((route, methods, desc))

    # 真实构造（不要用 __new__ 绕过 __init__，那样测不到 self.config 这类问题）
    ctx = _StubCtx()
    conf = {"enable": True, "rotate_mode": "round_robin", "switch_retries": 1}
    tmp_dir = Path(tempfile.mkdtemp())
    _orig_gdd = main_mod.StarTools.get_data_dir
    main_mod.StarTools.get_data_dir = classmethod(lambda cls, name=None: tmp_dir)
    try:
        plugin = main_mod.KeyRotatorPlugin(ctx, conf)
    finally:
        main_mod.StarTools.get_data_dir = _orig_gdd

    assert getattr(plugin, "config", None) is conf, "Star 子类必须自己保存 config"
    assert plugin._data_dir == tmp_dir, f"数据目录不对: {plugin._data_dir}"
    assert plugin._file_settings == {}, "首次启动应无面板设置文件"
    route_names = [r[0].rsplit("/", 1)[-1] for r in ctx.routes]
    expect = {
        "status", "settings", "reset", "keys", "set_key", "clear_key",
        "set_type", "set_source", "delete_model", "restore_model", "<path:rest>",
    }
    assert set(route_names) == expect, f"路由不符: {route_names}"
    assert all(r[0].startswith("/astrbot_plugin_key_rotator/") for r in ctx.routes), ctx.routes
    print("[OK] 真实实例化：config 已保存、数据目录、10 条面板路由 + 1 条兜底")
    plugin._sync_settings()
    assert provider_mod.PLUGIN_SETTINGS["switch_retries"] == 1, provider_mod.PLUGIN_SETTINGS

    prov.key_ring.copy()
    prov._on_ok()
    prov.key_ring.copy()
    prov._on_failed(TimeoutError("read timeout"))
    kinds = [e["kind"] for e in provider_mod.EVENTS]
    assert kinds == ["use", "fail"], f"事件流顺序不对: {kinds}"

    status = plugin._build_status()
    assert status["ok"] and status["registered"], "status 未就绪"
    assert status["totals"]["lines"] == 1 and status["totals"]["keys"] == 5, status["totals"]
    assert status["totals"]["assigned"] == 2, status["totals"]
    assert status["lines"][0]["source_id"] == "agens"
    assert status["lines"][0]["api_base"] == "https://api.agnes-ai.cn/v1"
    assert status["events"] and status["events"][0]["kind"] == "fail", "事件应倒序返回"
    print("[OK] 面板 status 聚合：1 线路 / 5 Key / 2 次分配 / 事件倒序")

    mc = status.get("model_config")
    assert mc and mc.get("sources") is not None and mc.get("models") is not None, "缺少 model_config"
    ids = [s["id"] for s in mc["sources"]]
    assert ids == ["agens", "axis"], f"应读到全部线路(含未轮换的): {ids}"
    agens = next(s for s in mc["sources"] if s["id"] == "agens")
    axis = next(s for s in mc["sources"] if s["id"] == "axis")
    assert agens["rotating"] and agens["key_count"] == 5, agens
    assert not axis["rotating"] and axis["key_count"] == 1, axis
    mids = [m["id"] for m in mc["models"]]
    assert mids == ["agens/agnes-2.5-flash", "axis/gpt-5.4"], mids
    assert mc["models"][0]["rotating"] and not mc["models"][1]["rotating"]
    assert mc["models"][0]["instantiated"], mc["models"][0]
    assert not mc["models"][1]["instantiated"], mc["models"][1]
    print("[OK] model_config：读到全部 2 条线路 + 2 个模型，轮换/实例标记正确")

    # ---- 覆盖 Key：接管任意 OpenAI 兼容实例（含未改 type 的原生实例） ----
    from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial

    native = ProviderOpenAIOfficial(dict(src_cfg, type="openai_chat_completion"), {})
    assert not isinstance(native, provider_mod.RotatingOpenAIProvider)
    assert provider_mod.adopt_key_override(native, "NEW-KEY-1234567890"), "应能接管原生实例"
    assert native.api_keys.copy() == ["NEW-KEY-1234567890"], "覆盖后每次调用必须用它"
    desc = native.api_keys.describe()
    assert desc["overridden"] and desc["override"][0].endswith("7890"), desc
    provider_mod.release_key_override(native)
    assert native.api_keys.override_keys == [], "撤销后应恢复原列表"
    assert native.api_keys.copy() in (["k1"], ["k2"], ["k3"], ["k4"], ["k5"])

    # 手动 Key 池：多个 Key -> 池内轮换（这是「配置拿不到 Key 时手动填充」的场景）
    ring2 = provider_mod.RotatingKeyRing(
        ["cfg-key"], source_id="t", source_config={}
    )
    assert ring2.set_override(["p1xxxxxx", "p2xxxxxx", "p3xxxxxx"]) == 3
    seq = [ring2.copy()[0] for _ in range(7)]
    assert seq[:3] == ["p1xxxxxx", "p2xxxxxx", "p3xxxxxx"], f"池内应顺序轮换: {seq}"
    assert set(seq) == {"p1xxxxxx", "p2xxxxxx", "p3xxxxxx"}, seq
    d2 = ring2.describe()
    assert d2["overridden"] and d2["total"] == 3 and d2["source_key_count"] == 1, d2
    assert d2["keys"][0]["manual"] is True, d2["keys"][0]
    ring2.clear_override()
    assert ring2.override_keys == [] and ring2.copy() == ["cfg-key"], "清空后应回到配置 Key"
    print("[OK] 手动 Key 池：3 个 Key 顺序轮换 / describe 展示生效集 / 清空回退")

    # ---- 限速：均分 -> 填格 -> 满额跳过 -> 全满移交备用 -> 窗口重置 ----
    provider_mod.PLUGIN_SETTINGS["enable"] = True
    rate_ring = provider_mod.RotatingKeyRing(
        ["k1", "k2", "k3", "k4", "k5", "k6", "k7"],
        source_id="rate-test",
        source_config={},
    )
    provider_mod.PLUGIN_SETTINGS["rate_limit_per_min"] = 60
    try:
        assert rate_ring.rate_limit == 60, rate_ring.rate_limit
        assert rate_ring.quota == 8, f"60/7 应为 8（舍去小数）: {rate_ring.quota}"

        # 转一圈：7 个 Key 各用 1 次，每格 +1
        first_round = [rate_ring.copy()[0] for _ in range(7)]
        assert first_round == ["k1", "k2", "k3", "k4", "k5", "k6", "k7"], first_round
        d = rate_ring.describe()
        assert d["limited"] and d["rate_limit"] == 60 and d["window"] == 60, d
        assert all(k["used"] == 1 and k["quota"] == 8 for k in d["keys"]), d["keys"]
        assert not any(k["exhausted"] for k in d["keys"]), d["keys"]

        # 持续轮换直到全部填满：一旦某 Key 满 8 次就绝不再被选中
        full_keys: set = set()
        raised = None
        for _ in range(500):
            try:
                picked = rate_ring.copy()[0]
            except provider_mod.RateLimitExhaustedError as exc:
                raised = exc
                break
            assert picked not in full_keys, f"满额 Key {picked} 不应再被选中"
            assert rate_ring.stats[picked]["used"] <= 8, "不允许超出额度"
            if rate_ring.stats[picked]["used"] == 8:
                full_keys.add(picked)
        assert raised is not None, "全满后应抛 RateLimitExhaustedError"
        assert len(full_keys) == 7, f"7 个 Key 都应被填满过: {full_keys}"
        d2 = rate_ring.describe()
        assert d2["all_exhausted"] and all(k["exhausted"] for k in d2["keys"]), d2
        assert any(ev["kind"] == "fallback" for ev in provider_mod.EVENTS), "应记移交事件"
        print(f"[OK] 限速：60/7=8、满额跳过、全满移交备用（{str(raised)[:42]}…）")

        # 窗口到期 -> 自动重置
        rate_ring._window_start -= provider_mod.RATE_WINDOW_S + 1
        rate_ring.copy()
        d3 = rate_ring.describe()
        assert not d3["all_exhausted"], d3
        assert max(k["used"] for k in d3["keys"]) == 1, d3["keys"]
        print("[OK] 60s 窗口到期自动重置，额度恢复")

        # 限速 < Key 数 -> 均分 0 -> 直接移交 + 诊断提示
        provider_mod.PLUGIN_SETTINGS["rate_limit_per_min"] = 3
        try:
            rate_ring.copy()
            raise AssertionError("quota=0 应该直接移交备用")
        except provider_mod.RateLimitExhaustedError:
            pass
        warns = plugin._diagnose(
            {"sources": [{"id": "agnes", "key_count": 7, "rotating": True,
                          "instance_rotating": True}]},
            1,
        )
        assert any("额度为 0" in w for w in warns), warns
        print("[OK] 限速 < Key 数：均分 0 立即移交 + 面板给出诊断")
    finally:
        provider_mod.PLUGIN_SETTINGS["rate_limit_per_min"] = 0

    # 不限速 -> 不计数，额度条为空
    no_limit = provider_mod.RotatingKeyRing(
        ["a", "b"], source_id="no-limit", source_config={}
    )
    for _ in range(50):
        no_limit.copy()
    d4 = no_limit.describe()
    assert d4["limited"] is False and d4["quota"] == 0, d4
    assert all(k["quota"] == 0 and not k["exhausted"] for k in d4["keys"]), d4["keys"]
    print("[OK] 不限速：无额度约束，quota=0（面板画空条）")

    # ---- 冷却：冷却中的 Key 跳过轮询；全部冷却 -> 移交备用 ----
    provider_mod.PLUGIN_SETTINGS["rate_limit_per_min"] = 0
    cool = provider_mod.RotatingKeyRing(
        ["c1", "c2", "c3"], source_id="cool-test", source_config={}
    )
    cool.report_failure("c1", TimeoutError("t1"))
    cool.report_failure("c1", TimeoutError("t2"))  # 连续 2 次 -> 冷却
    dc = cool.describe()
    assert dc["keys"][0]["cooldown"] > 0 and dc["keys"][0]["skipped"], dc["keys"][0]
    assert dc["available"] == 2 and not dc["all_unavailable"], dc
    seq = [cool.copy()[0] for _ in range(6)]
    assert "c1" not in seq, f"冷却中的 Key 不应被轮到: {seq}"
    assert cool.describe()["next_index"] != 1, "「下一个」不能指向冷却中的 Key"
    print("[OK] 冷却跳过：连续 6 次轮询都不碰冷却 Key，next_index 也绕开")

    # 报错显示：失败即显示、一直保留；只有该 Key 下一次成功才消除
    st_c1 = cool.stats["c1"]
    assert st_c1["last_error"], "刚失败应记录报错"
    assert cool.describe()["keys"][0]["last_error"] == st_c1["last_error"], "失败即显示"

    st_c1["cooldown_until"] = 0.0  # 模拟冷却结束
    after = cool.describe()["keys"][0]
    assert after["last_error"] == st_c1["last_error"], f"冷却结束不应自动隐藏: {after}"
    assert after["fail"] == st_c1["fail"], "失败计数必须保留"

    cool.report_failure("c1", TimeoutError("again"))  # 再次失败 -> 刷新
    shown = cool.describe()["keys"][0]
    assert shown["last_error"].endswith("again"), f"再次失败应刷新: {shown}"

    cool.report_success("c1")  # 成功一次 -> 消除
    cleared = cool.describe()["keys"][0]
    assert cleared["last_error"] == "", f"成功后应消除报错: {cleared}"
    assert cleared["fail"] == shown["fail"], "成功不清失败计数"
    print("[OK] 报错：失败即显示 → 冷却结束不隐藏 → 再次失败刷新 → 成功一次消除")

    # 全部冷却 -> 移交备用（不再 fail-open 硬用冷却中的 Key）
    for k in ("c1", "c2", "c3"):  # c1 刚被人为解除冷却，需要重新攒满阈值
        cool.report_failure(k, TimeoutError("x"))
        cool.report_failure(k, TimeoutError("y"))
    d_all = cool.describe()
    assert d_all["all_unavailable"] and d_all["available"] == 0, d_all
    try:
        cool.copy()
        raise AssertionError("全部冷却时应移交备用模型")
    except provider_mod.RateLimitExhaustedError as exc:
        assert "冷却" in str(exc), exc
    print("[OK] 全部 Key 冷却中 -> 移交备用模型（不再硬用冷却 Key）")

    # ---- 回归：换 Key 重试的第 2 次尝试在 copy() 前抛出，不得记到上一个 Key ----
    #（这正是"成功次数变成 -3"的根因：fail 被重复归因到已 assigned 的 Key）
    provider_mod.PLUGIN_SETTINGS["cooldown_threshold"] = 1  # 失败 1 次即冷却

    async def _boom(*args, **kwargs):
        raise TimeoutError("simulated timeout")

    try:
        solo = provider_mod.RotatingOpenAIProvider(
            {
                "id": "agens/solo-model",
                "provider_source_id": "agens",
                "type": provider_mod.PROVIDER_TYPE,
                "provider_type": "chat_completion",
                "enable": True,
                "provider": "openai",
                "key": ["only-one-key"],
                "api_base": "https://api.agnes-ai.cn/v1",
                "timeout": 5,
                "custom_headers": {},
                "model": "solo-model",
            },
            {},
        )
        solo._query = _boom  # 不发真实请求
        try:
            asyncio.run(solo.text_chat(prompt="hi"))
        except Exception:
            pass
        st_solo = solo.key_ring.stats["only-one-key"]
        assert st_solo["assigned"] >= st_solo["fail"], (
            "成功次数不能为负："
            f"assigned={st_solo['assigned']} fail={st_solo['fail']}"
        )
        print(
            "[OK] 归因修复：assigned="
            f"{st_solo['assigned']} fail={st_solo['fail']}（不再 fail>assigned）"
        )

        # ---- 模型级熔断：连续失败 -> 跳过整个模型，后续请求秒抛 ----
        provider_mod.PLUGIN_SETTINGS["cooldown_threshold"] = 2
        pm = provider_mod.RotatingOpenAIProvider(
            {
                "id": "agens/pm-model",
                "provider_source_id": "agens",
                "type": provider_mod.PROVIDER_TYPE,
                "provider_type": "chat_completion",
                "enable": True,
                "provider": "openai",
                "key": ["pk1", "pk2", "pk3"],
                "api_base": "https://api.agnes-ai.cn/v1",
                "timeout": 5,
                "custom_headers": {},
                "model": "pm-model",
            },
            {},
        )
        pm._query = _boom
        for _ in range(2):  # 两次完整请求失败
            try:
                asyncio.run(pm.text_chat(prompt="hi"))
            except Exception:
                pass
        assert pm.key_ring.model_skipped, "连续失败后模型应进入跳过状态"
        d_pm = pm.key_ring.describe()
        assert d_pm["model_skipped"] > 0 and d_pm["model_label"] == "agens/pm-model", d_pm

        start = time.monotonic()
        try:
            asyncio.run(pm.text_chat(prompt="hi"))
            raise AssertionError("跳过期间应直接抛出")
        except provider_mod.ModelSkippedError:
            pass
        elapsed = time.monotonic() - start
        assert elapsed < 0.05, f"跳过应秒抛，不应再走网络/重试（实际 {elapsed:.3f}s）"
        print(
            "[OK] 模型级熔断：连续失败 -> 跳过整个模型 -> 后续请求 "
            f"{elapsed * 1000:.0f}ms 秒抛给下一个模型"
        )
    finally:
        provider_mod.PLUGIN_SETTINGS["cooldown_threshold"] = 2

    # ---- 端到端：5 个 Key 全冷却 -> 真实 AgentRunner 直接调用备用 API ----
    from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    from astrbot.core.provider.entities import LLMResponse, ProviderRequest

    five = provider_mod.RotatingOpenAIProvider(
        {
            "id": "agens/agnes-2.5-flash",
            "provider_source_id": "agens",
            "type": provider_mod.PROVIDER_TYPE,
            "provider_type": "chat_completion",
            "enable": True,
            "provider": "openai",
            "key": ["k1", "k2", "k3", "k4", "k5"],
            "api_base": "https://api.agnes-ai.cn/v1",
            "timeout": 5,
            "custom_headers": {},
            "model": "agnes-2.5-flash",
        },
        {},
    )
    for k in list(five.key_ring):
        five.key_ring.report_failure(k, TimeoutError("boom"))
        five.key_ring.report_failure(k, TimeoutError("boom"))
    assert five.key_ring.describe()["available"] == 0, "5 个 Key 应全部处于冷却"

    # 冷却移交错误必须是「不可切换」，否则 wrapper 会先做无谓的换 Key 重试
    assert five.key_ring.should_switch(
        provider_mod.RateLimitExhaustedError("x")
    ) is False, "冷却移交错误不应触发换 Key 重试"

    class BackupStub:
        """备用 API：只记录被调用次数并返回固定结果，不发真实请求。"""

        def __init__(self):
            self.calls = 0
            self.provider_config = {
                "id": "axis/gpt-5.4",
                "modalities": ["text"],
                "enable": True,
            }

        async def text_chat(self, **kwargs):
            self.calls += 1
            resp = LLMResponse(role="assistant")
            resp.completion_text = "来自备用 API"
            return resp

    backup = BackupStub()

    runner = object.__new__(ToolLoopAgentRunner)
    runner.req = ProviderRequest()
    runner.req.session_id = "test:1"
    runner.req.prompt = "hi"
    runner.req.func_tool = None
    runner.req.model = None
    runner.req.extra_user_content_parts = []
    runner.run_context = type(
        "FakeRunContext", (), {"messages": [{"role": "user", "content": "hi"}]}
    )()
    runner.provider = five
    runner.fallback_providers = [backup]
    runner.streaming = False
    runner.request_max_retries = 1
    runner._abort_signal = asyncio.Event()

    collected: list = []

    async def _drive():
        async for resp in runner._iter_llm_responses_with_fallback():
            collected.append(resp)

    asyncio.run(_drive())

    assert backup.calls == 1, f"备用 API 应被调用 1 次，实际 {backup.calls}"
    assert collected, "应有返回结果"
    assert collected[0].completion_text == "来自备用 API", collected[0].completion_text
    print("[OK] 端到端：5 Key 全冷却 -> 真实 AgentRunner 直接调用备用 API 并返回其结果")

    # 多 Key 文本解析
    got = main_mod.KeyRotatorPlugin._split_keys(
        "aaa11111, bbb22222\nccc33333，ddd44444; eee55555  fff66666"
    )
    assert got == ["aaa11111", "bbb22222", "ccc33333", "ddd44444", "eee55555", "fff66666"], got
    print("[OK] _split_keys：换行/中英文逗号分号/空格分隔")

    class _NotOpenAI:
        api_keys = ["x"]

    assert provider_mod.adopt_key_override(_NotOpenAI(), "k") is False, "非 OpenAI 兼容不应接管"

    plugin._overrides["agens/agnes-2.5-flash"] = {
        "key": "NEW-KEY-1234567890",
        "at": 1,
    }
    assert plugin._save_overrides(), "overrides 应能落盘"
    reloaded_ov = plugin._load_overrides()["agens/agnes-2.5-flash"]
    assert reloaded_ov["keys"] == ["NEW-KEY-1234567890"], reloaded_ov
    applied = plugin._apply_overrides()
    assert applied >= 1, f"应至少接管 1 个实例，实际 {applied}"
    ovs = plugin._overrides_status()
    o = ovs["agens/agnes-2.5-flash"]
    assert o["active"] and o["instantiated"] and o["count"] == 1, o
    assert all("NEW-KEY-1234567890" not in k for k in o["keys"]), f"不能回显明文: {o}"
    assert hasattr(plugin, "apply_key_overrides"), "缺少 on_waiting_llm_request 钩子"
    print("[OK] 手动池下发：原生实例接管 / 撤销 / 脱敏状态 / 钩子存在")

    # ---- 诊断：解释「配置 vs 实例」不一致 ----
    warns = plugin._diagnose(
        {
            "sources": [
                {"id": "agnes", "key_count": 5, "rotating": False, "instance_rotating": False},
                {"id": "empty", "key_count": 0, "rotating": True, "instance_rotating": False},
            ]
        },
        0,
    )
    assert len(warns) == 3, warns
    assert any("5 个 Key" in w for w in warns), warns
    assert any("待重建" in w or "原生型" in w for w in warns), warns
    assert any("没有任何 Key" in w for w in warns), warns
    assert plugin._diagnose({"sources": []}, 0), "没有任何线路时应给出引导提示"
    print("[OK] _diagnose：多 Key 未轮换 / 实例未重建 / 无 Key 三类提示齐全")

    # ---- 配置写入：一键改类型 / 删除 / 恢复（假配置，绝不写真实 cmd_config） ----
    import json as _json

    import astrbot.core as _core

    class FakeCfg(dict):
        def __init__(self, data):
            super().__init__(data)
            self.saves = 0

        async def save_config_async(self, *args, **kwargs):
            self.saves += 1
            return True

    class FakePM:
        def __init__(self):
            self.reloaded = []

        async def reload(self, cfg):
            self.reloaded.append(cfg.get("id"))

    class FakeReq:
        def __init__(self, payload):
            self.query = {}
            self._payload = payload

        async def json(self):
            return self._payload

    fake_cfg = FakeCfg(
        {
            "provider_sources": [
                {
                    "id": "agens",
                    "type": provider_mod.PROVIDER_TYPE,
                    "provider_type": "chat_completion",
                    "enable": True,
                    "key": ["k1", "k2", "k3", "k4", "k5"],
                    "api_base": "https://api.agnes-ai.cn/v1",
                    "timeout": 10,
                },
                {
                    "id": "axis",
                    "type": "openai_chat_completion",
                    "provider_type": "chat_completion",
                    "enable": True,
                    "key": ["sk-only-one"],
                    "api_base": "https://ai.onyxaxis.org/v1",
                    "timeout": 15,
                },
            ],
            "provider": [
                {
                    "id": "agens/agnes-2.5-flash",
                    "model": "agnes-2.5-flash",
                    "provider_source_id": "agens",
                    "enable": True,
                },
                {
                    "id": "axis/gpt-5.4",
                    "model": "gpt-5.4",
                    "provider_source_id": "axis",
                    "enable": True,
                },
            ],
        }
    )
    plugin.context = _StubCtx()
    plugin.context.get_config = lambda umo=None: fake_cfg
    plugin.context.provider_manager = FakePM()

    _orig_cfg = _core.astrbot_config
    _orig_req = main_mod.web_request

    def call(name, payload):
        main_mod.web_request = FakeReq(payload)
        resp = asyncio.run(getattr(plugin, name)())
        status = getattr(resp, "status_code", 200)
        body = _json.loads(bytes(resp.body).decode())
        return status, body

    _core.astrbot_config = fake_cfg
    try:
        # 1) 一键改类型
        sc, body = call(
            "api_set_type",
            {"source_id": "axis", "type": provider_mod.PROVIDER_TYPE},
        )
        assert sc == 200 and body["ok"] and body["changed"], body
        assert fake_cfg["provider_sources"][1]["type"] == provider_mod.PROVIDER_TYPE
        assert fake_cfg.saves >= 1, "应调用 save_config_async 落盘"
        assert "axis/gpt-5.4" in plugin.context.provider_manager.reloaded, "应重建实例"
        assert body["reloaded"] == 1, body
        print("[OK] 一键改类型：写 config + 落盘 + 重建受影响实例")

        sc, body = call("api_set_type", {"source_id": "axis", "type": "whatever"})
        assert sc == 400, (sc, body)
        sc, body = call(
            "api_set_type", {"source_id": "nope", "type": provider_mod.PROVIDER_TYPE}
        )
        assert sc == 404, (sc, body)
        print("[OK] set_type 参数校验：非法类型 400 / 不存在线路 404")

        # 1.5) 修改线路参数（超时/代理写服务商配置 + 重建实例）
        sc, body = call(
            "api_set_source",
            {"source_id": "axis", "timeout": 30, "proxy": "http://127.0.0.1:7890"},
        )
        assert sc == 200 and body["changed"] and body["saved"], body
        assert body["changes"] == {"timeout": 30, "proxy": "http://127.0.0.1:7890"}, body
        assert fake_cfg["provider_sources"][1]["timeout"] == 30, "超时必须写进配置"
        assert fake_cfg["provider_sources"][1]["proxy"] == "http://127.0.0.1:7890"
        assert "axis/gpt-5.4" in plugin.context.provider_manager.reloaded, "应重建受影响实例"
        assert body["reloaded"] == 1, body
        print("[OK] set_source：超时/代理写服务商配置 + 落盘 + 重建实例")

        # 幂等：相同值 -> changed=False；清空代理：传空串
        sc, body = call("api_set_source", {"source_id": "axis", "timeout": 30})
        assert sc == 200 and body["changed"] is False, body
        sc, body = call("api_set_source", {"source_id": "axis", "proxy": ""})
        assert sc == 200 and body["changes"] == {"proxy": ""}, body
        assert fake_cfg["provider_sources"][1]["proxy"] == "", "空串应清掉代理"
        print("[OK] set_source：幂等不落盘 / 空串清空代理")

        # 参数校验：超时越界 400 / api_base 非 http 400 / 不存在线路 404 / 缺 source_id 400
        sc, body = call("api_set_source", {"source_id": "axis", "timeout": 0})
        assert sc == 400, (sc, body)
        sc, body = call("api_set_source", {"source_id": "axis", "api_base": "ftp://x"})
        assert sc == 400, (sc, body)
        sc, body = call("api_set_source", {"source_id": "nope", "timeout": 20})
        assert sc == 404, (sc, body)
        sc, body = call("api_set_source", {"timeout": 20})
        assert sc == 400, (sc, body)
        print("[OK] set_source 参数校验：越界/非法地址 400、缺 id 400、不存在 404")

        # 2) 删除模型 -> 退出轮回 + 保留在后台 + 清 Key 池
        plugin._overrides["axis/gpt-5.4"] = {
            "keys": ["manual-key-123456"],
            "at": 1,
        }
        sc, body = call("api_delete_model", {"model_id": "axis/gpt-5.4"})
        assert sc == 200 and body["changed"], body
        assert fake_cfg["provider"][1]["enable"] is False, "配置必须保留但 enable=false"
        assert "axis/gpt-5.4" not in plugin._overrides, "删除应清掉手动 Key 池"
        assert "axis/gpt-5.4" in plugin.context.provider_manager.reloaded
        mc2 = plugin._read_model_config()
        assert mc2["models"][1]["deleted"] is True, mc2["models"][1]
        print("[OK] 删除模型：enable=false（后台保留）+ 清 Key 池 + 标记 deleted")

        # 3) 恢复模型
        sc, body = call("api_restore_model", {"model_id": "axis/gpt-5.4"})
        assert sc == 200 and body["changed"], body
        assert fake_cfg["provider"][1]["enable"] is True
        mc3 = plugin._read_model_config()
        assert mc3["models"][1]["deleted"] is False, mc3["models"][1]
        print("[OK] 恢复模型：enable=true + deleted 标记清除")

        # 最后兜底：追加到回退链末位，保证全报错时一定有备用可转交
        fake_cfg["agent_runner"] = {
            "config": {
                "model": {
                    "provider_id": "agens/agnes-2.5-flash",
                    "fallback_provider_ids": [],
                }
            }
        }
        sc, body = call(
            "api_settings", {"backup_provider_id": "axis/gpt-5.4"}
        )
        assert sc == 200 and body["ok"], body
        br = body["backup"]
        assert br["added"] is True and br["chain"] == ["axis/gpt-5.4"], br
        assert (
            fake_cfg["agent_runner"]["config"]["model"]["fallback_provider_ids"]
            == ["axis/gpt-5.4"]
        ), "回退链必须被写入配置"
        print("[OK] 最后兜底：模型已追加到回退链末位并落盘")

        # 幂等 + 非法值
        sc, body = call("api_settings", {"backup_provider_id": "axis/gpt-5.4"})
        assert body["backup"]["added"] is False and body["backup"]["note"], body["backup"]
        sc, body = call("api_settings", {"backup_provider_id": "no/such-model"})
        assert body["backup"].get("error"), body["backup"]
        print("[OK] 最后兜底：重复设置幂等 / 不存在的模型报错")

        # 底层请求重试（AstrBot 原生 agent_runner 配置）与插件设置同一次提交保存
        sc, body = call(
            "api_settings", {"switch_retries": 2, "request_max_retries": 1}
        )
        assert sc == 200 and body["ok"], body
        assert (
            fake_cfg["agent_runner"]["config"]["model"]["request_max_retries"] == 1
        ), "底层重试必须写进 agent_runner 配置"
        assert body["settings"]["switch_retries"] == 2, "同次提交的插件设置也要生效"
        assert body["agent"]["request_max_retries"] == 1, body["agent"]
        assert body["agent_config"]["request_max_retries"] == 1, body["agent_config"]
        print("[OK] 底层重试：写 agent_runner 配置 + 与插件设置同次保存")

        # AstrBot 归一化规则：<1 按 1 处理；非整数报错不写配置
        sc, body = call("api_settings", {"request_max_retries": 0})
        assert body["agent"]["request_max_retries"] == 1, body["agent"]
        sc, body = call("api_settings", {"request_max_retries": "abc"})
        assert body["agent"].get("error"), body["agent"]
        assert (
            fake_cfg["agent_runner"]["config"]["model"]["request_max_retries"] == 1
        ), "非法值不得写配置"
        # status 里能读到配对所需的原生参数
        sc, body = call("api_status", {})
        assert body["agent"]["request_max_retries"] == 1, body["agent"]
        assert body["agent"]["main_provider_id"] == "agens/agnes-2.5-flash", body["agent"]
        print("[OK] 底层重试：<1 钳到 1 / 非法值拒绝 / status 暴露配对参数")

        # 模型配置里能看到兜底设置
        assert provider_mod.PLUGIN_SETTINGS.get("backup_provider_id") == "no/such-model"
        provider_mod.PLUGIN_SETTINGS["backup_provider_id"] = ""
    finally:
        main_mod.web_request = _orig_req
        _core.astrbot_config = _orig_cfg

    saved = plugin._apply_settings(
        {"enable": False, "switch_retries": 99, "rotate_mode": "bad", "cooldown_threshold": 0}
    )
    assert saved["enable"] is False
    assert saved["switch_retries"] == 5, f"越界未钳制: {saved}"
    assert saved["rotate_mode"] == "round_robin", f"非法策略未回退: {saved}"
    assert saved["cooldown_threshold"] == 1, f"下界未钳制: {saved}"
    assert plugin._settings_file.exists(), "settings.json 未落盘"
    reloaded = plugin._load_file_settings()
    assert reloaded["enable"] is False and reloaded["switch_retries"] == 5, reloaded
    provider_mod.PLUGIN_SETTINGS.update(
        {"enable": True, "switch_retries": 1, "rotate_mode": "round_robin", "cooldown_threshold": 2}
    )
    print("[OK] 面板 settings：校验钳制 + 持久化 + 重新加载")

    print("[ALL PASS]")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
