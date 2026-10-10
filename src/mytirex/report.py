"""A few figures using the original plotting and display-thinning helpers."""
from pathlib import Path
import html
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics import numeric_column

METHOD_ORDER = ("TiRex-2", "TiRex-2 (official run)", "TiRex-2 (official)", "TiRex-2 (current historical)")
COLORS = {"TiRex-2": "#0072b2", "TiRex-2 (official run)": "#0072b2", "TiRex-2 (official)": "#777777", "TiRex-2 (current historical)": "#d55e00"}
MAX_DISPLAY_POINTS = 2400

def _methods(frame: pd.DataFrame) -> list[str]:
    available = set(frame["method"].astype(str))
    return [method for method in METHOD_ORDER if method in available] + sorted(available - set(METHOD_ORDER))


def _save(fig: plt.Figure, destination: Path, filename: str) -> Path:
    path = destination / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _display(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("query_time_s")
    if len(ordered) <= MAX_DISPLAY_POINTS:
        return ordered
    indices = np.unique(np.linspace(0, len(ordered) - 1, MAX_DISPLAY_POINTS, dtype=int))
    return ordered.iloc[indices]


def _typical_sequence(frame: pd.DataFrame) -> str:
    # Prefer a trajectory with visible speed changes; choose by reference alone.
    reference = frame[["sequence", "query_index", "reference"]].drop_duplicates(["sequence", "query_index"])
    spans = reference.groupby("sequence")["reference"].agg(
        lambda values: float(values.quantile(0.95) - values.quantile(0.05))
    )
    return str(spans.idxmax())


def _error_distribution(frame: pd.DataFrame, destination: Path) -> Path:
    delays = sorted(frame["delay_ms"].unique())
    methods = _methods(frame)
    fig, axes = plt.subplots(1, len(delays), figsize=(6 * len(delays), 4.4), squeeze=False)
    for axis, delay in zip(axes.flat, delays):
        for method in methods:
            group = frame[(frame["delay_ms"] == delay) & (frame["method"] == method)]
            errors = np.abs(numeric_column(group, "prediction") - numeric_column(group, "reference"))
            errors = np.sort(errors[np.isfinite(errors)])
            if not errors.size:
                continue
            probability = np.arange(1, len(errors) + 1) / len(errors)
            # Thinning is only for drawing the ECDF; all rows enter its ranks.
            indices = np.unique(np.linspace(0, len(errors) - 1, MAX_DISPLAY_POINTS, dtype=int))
            axis.plot(errors[indices], probability[indices], label=method, color=COLORS.get(method), linewidth=1.6)
        axis.set_title(f"Delay {delay:g} ms")
        axis.set_xlabel("Absolute error (m/s)")
        axis.set_ylabel("Empirical cumulative probability")
        axis.set_ylim(0, 1.01)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Absolute-error distributions across all sequences")
    fig.tight_layout()
    return _save(fig, destination, "absolute_error_ecdf.png")


def _plot_lines(axis, subset: pd.DataFrame, methods: list[str], title: str | None = None) -> None:
    shared = _display(subset.drop_duplicates("query_index"))
    axis.plot(numeric_column(shared, "query_time_s"), numeric_column(shared, "reference"),
              color="black", label="Reference", linewidth=1.8)
    axis.plot(numeric_column(shared, "query_time_s"), numeric_column(shared, "last_observation"),
              color="#888888", linestyle="--", label="Delayed observation", linewidth=1.2)
    for method in methods:
        group = _display(subset[subset["method"] == method])
        axis.plot(numeric_column(group, "query_time_s"), numeric_column(group, "prediction"),
                  label=method, color=COLORS.get(method), linewidth=1.0, alpha=0.9)
    if title:
        axis.set_title(title)
    axis.set_xlabel("Query time (s)")
    axis.set_ylabel("Speed (m/s)")
    axis.grid(alpha=0.25)


def make_plots(frame: pd.DataFrame, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    sequence = _typical_sequence(frame)
    paths = []
    for delay in sorted(frame.delay_ms.unique()):
        subset = frame.loc[(frame.sequence == sequence) & (frame.delay_ms == delay)]
        fig, axis = plt.subplots(figsize=(11, 4))
        _plot_lines(axis, subset, _methods(subset), f"{sequence} | fixed delay {delay} ms")
        axis.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        paths.append(_save(fig, output, f"trajectory_{int(delay)}ms.png"))
    paths.append(_error_distribution(frame, output))
    return paths


def _page(title: str, body: str) -> str:
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title>"
            "<style>body{max-width:1160px;margin:30px auto;padding:0 20px;font:16px/1.6 sans-serif;color:#202c35}"
            "table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}td,th{border:1px solid #d5dde3;padding:7px;text-align:right}"
            "th{background:#f0f5f7}img{width:100%;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f8;padding:15px}"
            "a{color:#006a9e}.note{background:#eff6fb;padding:15px}</style>"
            f"<body><h1>{html.escape(title)}</h1>{body}</body></html>")


def write_report(output: Path, metrics: dict, metadata: dict, comparison: pd.DataFrame | None,
                 plots: list[Path]) -> None:
    columns = ["method", "delay_ms", "query_count", "MAE", "RMSE", "Bias", "abs_error_p95", "abs_error_p99",
               "failure_count", "fallback_count"]
    table = metrics["summary"][columns].to_html(index=False, escape=True, float_format=lambda x: f"{x:.7g}")
    model = metadata["model"]
    settings = model["settings"]
    body = (f"<p>本次新预测：{metadata['query_count']} 条；设备：{html.escape(model['gpu_name'])}。"
            f"源码：{html.escape(model['source'])}；上下文 {settings['context_length']} 点；"
            f"pad_context={settings['pad_context']}；tta_diff={settings['tta_diff']}；"
            f"趋势窗口={settings['trend_window_points']}，阈值={settings['trend_threshold']}，R²={settings['trend_min_r2']}。</p>"
            "<p class='note'>原五条插值速度轨迹，仅用于开发回归；整数微秒到达边界包含等号。"
            "此评估以查询时刻为目标，尚未补偿排队和推理耗时。40 Hz 是数据采样频率。"
            "批量 elapsed_s 为摊销时间；batch=1 延迟请单独运行 benchmark。</p>"
            f"<p>加载 {model['load_time_s']:.3f} s；首批预测 {model['first_forecast']['elapsed_s']:.3f} s"
            "（可能包含首次 FlashRNN/Triton 编译）。历史参考的耗时不作为本次性能结论。</p>"
            f"<h2>新运行指标（m/s）</h2>{table}")
    if comparison is not None:
        body += "<h2>相同查询的固定对照</h2>" + comparison.to_html(index=False, escape=True, float_format=lambda x: f"{x:.7g}")
        body += "<p>official 为固定的官方源码与 2048 点配置；current historical 为配置匹配的优化副本历史预测。两者身份在 metadata.json 中分别记录。</p>"
    for path in plots:
        body += f"<img src='{html.escape(str(path.relative_to(output)))}' alt='{html.escape(path.stem)}'>"
    body += ("<p><a href='predictions.csv'>预测明细</a> · <a href='metrics_by_sequence.csv'>分序列指标</a> · "
             "<a href='metrics_failure_reasons.csv'>失败/回退</a> · <a href='config.json'>配置</a> · "
             "<a href='metadata.json'>内容身份与运行信息</a></p>")
    (output / "report.html").write_text(_page("TiRex-2 speed 开发回归", body), encoding="utf-8")


def write_json_report(output: Path, title: str, summary: dict) -> None:
    content = html.escape(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    links = "<p><a href='metadata.json'>完整运行信息</a> · <a href='config.json'>配置</a></p>"
    for name in ("benchmark.csv", "smoke_forecast.csv"):
        if (output / name).is_file():
            links += f"<p><a href='{name}'>{name}</a></p>"
    (output / "report.html").write_text(_page(title, f"<pre>{content}</pre>{links}"), encoding="utf-8")

