
import argparse
import json
import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


@dataclass
class Params:
    grid_step: float = 250.0
    source_spl_db_at_1m: float = 95.0
    wall_loss_db: float = 8.0
    threshold_db: float = 60.0
    opening_snap_tolerance: float = 300.0
    max_distance_units: float = 80000.0
    reference_distance_units: Optional[float] = None
    min_distance_units: Optional[float] = None

def log_step(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pt(v: Any) -> Point:
    return (float(v[0]), float(v[1]))


def seg(v: Any) -> Segment:
    return (pt(v[0]), pt(v[1]))


def as_polygon(boundary: Any) -> List[Point]:
    poly = [pt(p) for p in boundary]
    if poly and poly[0] != poly[-1]:
        poly.append(poly[0])
    return poly


def bbox_of_points(points: List[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def detect_units(data: Dict[str, Any], raw_params: Dict[str, Any]) -> str:
    boundary = as_polygon(data["boundary"])
    xmin, ymin, xmax, ymax = bbox_of_points(boundary[:-1])
    span = max(xmax - xmin, ymax - ymin)
    grid_step = float(raw_params.get("grid_step", 0.0) or 0.0)
    max_distance = float(raw_params.get("max_distance_units", raw_params.get("max_distance", 0.0)) or 0.0)
    # Heuristic: large spans and engineering-style params mean drawing units are probably mm.
    if span > 1000 or grid_step > 10 or max_distance > 1000:
        return "mm"
    return "m"


def normalize_params(data: Dict[str, Any]) -> Tuple[Params, str]:
    raw = dict(data.get("params", {}))
    if "opening_tolerance" in raw and "opening_snap_tolerance" not in raw:
        raw["opening_snap_tolerance"] = raw.pop("opening_tolerance")
    if "max_distance" in raw and "max_distance_units" not in raw:
        raw["max_distance_units"] = raw.pop("max_distance")

    units = detect_units(data, raw)

    if raw.get("reference_distance_units") is None:
        raw["reference_distance_units"] = 1000.0 if units == "mm" else 1.0
    if raw.get("min_distance_units") is None:
        raw["min_distance_units"] = 300.0 if units == "mm" else 0.3

    return Params(**raw), units


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: Point, b: Point, c: Point, eps: float = 1e-9) -> bool:
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
        and abs(orient(a, b, c)) <= eps
    )


def segments_intersect(s1: Segment, s2: Segment, eps: float = 1e-9) -> bool:
    p1, q1 = s1
    p2, q2 = s2
    o1 = orient(p1, q1, p2)
    o2 = orient(p1, q1, q2)
    o3 = orient(p2, q2, p1)
    o4 = orient(p2, q2, q1)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    if abs(o1) <= eps and on_segment(p1, p2, q1, eps):
        return True
    if abs(o2) <= eps and on_segment(p1, q2, q1, eps):
        return True
    if abs(o3) <= eps and on_segment(p2, p1, q2, eps):
        return True
    if abs(o4) <= eps and on_segment(p2, q1, q2, eps):
        return True
    return False


def point_segment_distance(p: Point, s: Segment) -> float:
    a, b = s
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return distance(p, a)
    apx = p[0] - a[0]
    apy = p[1] - a[1]
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    proj = (a[0] + abx * t, a[1] + aby * t)
    return distance(p, proj)


def intersection_point(s1: Segment, s2: Segment) -> Optional[Point]:
    (x1, y1), (x2, y2) = s1
    (x3, y3), (x4, y4) = s2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    p = (px, py)
    if point_segment_distance(p, s1) < 1e-6 and point_segment_distance(p, s2) < 1e-6:
        return p
    return None


def intersection_allowed(inter_pt: Point, openings: List[Segment], tol: float) -> bool:
    return any(point_segment_distance(inter_pt, op) <= tol for op in openings)


def sum_db(levels: List[float]) -> Optional[float]:
    vals = [10 ** (l / 10.0) for l in levels if l is not None and math.isfinite(l)]
    if not vals:
        return None
    return 10.0 * math.log10(sum(vals))


def spl_from_source(distance_units: float, params: Params) -> float:
    d = max(distance_units, params.min_distance_units or 1.0)
    ref = params.reference_distance_units or 1.0
    return params.source_spl_db_at_1m - 20.0 * math.log10(d / ref)


def build_grid(poly: List[Point], step: float) -> List[Point]:
    xmin, ymin, xmax, ymax = bbox_of_points(poly[:-1])
    poly_path = Path(poly)
    pts: List[Point] = []
    y = ymin + step / 2.0
    while y < ymax:
        row: List[Point] = []
        x = xmin + step / 2.0
        while x < xmax:
            row.append((x, y))
            x += step
        mask = poly_path.contains_points(row)
        pts.extend([p for p, inside in zip(row, mask) if inside])
        y += step
    return pts


def preprocess_segments(segments: List[Segment]) -> List[Tuple[Segment, Tuple[float, float, float, float]]]:
    out = []
    for s in segments:
        (a, b) = s
        out.append((s, (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))))
    return out


def count_blocker_crossings(ray: Segment,
                            blockers_idx: List[Tuple[Segment, Tuple[float, float, float, float]]],
                            openings: List[Segment],
                            tol: float) -> int:
    (p, q) = ray
    rxmin, rymin = min(p[0], q[0]), min(p[1], q[1])
    rxmax, rymax = max(p[0], q[0]), max(p[1], q[1])

    count = 0
    for b, (bxmin, bymin, bxmax, bymax) in blockers_idx:
        if bxmax < rxmin or bxmin > rxmax or bymax < rymin or bymin > rymax:
            continue
        if not segments_intersect(ray, b):
            continue
        ip = intersection_point(ray, b)
        if ip is not None and intersection_allowed(ip, openings, tol):
            continue
        count += 1
    return count


def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
    log_step("解析参数与几何数据")
    params, units = normalize_params(data)
    boundary = as_polygon(data["boundary"])
    blockers = [seg(s) for s in data.get("blockers", [])]
    openings = [seg(s) for s in data.get("openings", [])]
    sounders = [pt(p) for p in data.get("sounder_points", [])]
    if not sounders:
        raise ValueError("sounder_points is empty")

    log_step(f"单位识别完成: {units}")
    log_step("开始生成分析网格")
    grid_points = build_grid(boundary, params.grid_step)
    log_step(f"网格生成完成，共 {len(grid_points)} 个点")

    log_step("开始预处理 blocker 线段")
    blockers_idx = preprocess_segments(blockers)
    log_step(f"blocker 预处理完成，共 {len(blockers_idx)} 条")

    samples = []
    covered = 0
    min_db = None
    max_db = None
    total_grid = len(grid_points)

    log_step("开始逐点计算声压，这一步可能较慢")
    next_progress_pct = 10
    for i, gp in enumerate(grid_points, start=1):
        levels = []
        for sp in sounders:
            d = distance(gp, sp)
            if d > params.max_distance_units:
                continue
            crossings = count_blocker_crossings((sp, gp), blockers_idx, openings, params.opening_snap_tolerance)
            level = spl_from_source(d, params) - crossings * params.wall_loss_db
            levels.append(level)

        total = sum_db(levels)
        if total is not None:
            if min_db is None or total < min_db:
                min_db = total
            if max_db is None or total > max_db:
                max_db = total
            if total >= params.threshold_db:
                covered += 1

        samples.append({
            "point": [gp[0], gp[1]],
            "spl_db": total,
            "covered": bool(total is not None and total >= params.threshold_db),
        })

        if total_grid:
            pct_int = int(i * 100 / total_grid)
        else:
            pct_int = 100

        if i == total_grid or pct_int >= next_progress_pct:
            print(f"计算声压中: {pct_int}%", flush=True)
            while next_progress_pct <= pct_int:
                next_progress_pct += 10

    print()
    log_step("声压计算完成，开始汇总结果")

    total_pts = len(samples)
    return {
        "project": data.get("project", "ALMSOUND"),
        "units": units,
        "params": {
            "grid_step": params.grid_step,
            "source_spl_db_at_1m": params.source_spl_db_at_1m,
            "wall_loss_db": params.wall_loss_db,
            "threshold_db": params.threshold_db,
            "opening_snap_tolerance": params.opening_snap_tolerance,
            "max_distance_units": params.max_distance_units,
            "reference_distance_units": params.reference_distance_units,
            "min_distance_units": params.min_distance_units,
        },
        "summary": {
            "grid_point_count": total_pts,
            "covered_point_count": covered,
            "coverage_ratio": (covered / total_pts) if total_pts else 0.0,
            "threshold_db": params.threshold_db,
            "sounder_count": len(sounders),
            "blocker_count": len(blockers),
            "opening_count": len(openings),
            "min_spl_db": min_db,
            "max_spl_db": max_db,
        },
        "samples": samples,
    }


def plot_geometry_debug(data: Dict[str, Any], out_png: str) -> None:
    boundary = as_polygon(data["boundary"])
    blockers = [seg(s) for s in data.get("blockers", [])]
    openings = [seg(s) for s in data.get("openings", [])]
    sounders = [pt(p) for p in data.get("sounder_points", [])]

    fig, ax = plt.subplots(figsize=(12, 8))

    bx = [p[0] for p in boundary]
    by = [p[1] for p in boundary]
    ax.plot(bx, by, linewidth=1.8, label="boundary")

    for i, s in enumerate(blockers):
        ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], linewidth=1.1,
                label="blockers" if i == 0 else None)

    for i, s in enumerate(openings):
        ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], linewidth=2.4,
                label="openings" if i == 0 else None)

    if sounders:
        ax.scatter([p[0] for p in sounders], [p[1] for p in sounders], marker="*", s=70,
                   label=f"sounders ({len(sounders)})")

    ax.set_title("ALMSOUND geometry debug")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_heatmap(data: Dict[str, Any], result: Dict[str, Any], out_path: str) -> None:
    log_step("正在组织欠覆盖区图形")

    boundary = as_polygon(data["boundary"])
    blockers = [seg(s) for s in data.get("blockers", [])]
    openings = [seg(s) for s in data.get("openings", [])]
    sounders = [pt(p) for p in data.get("sounder_points", [])]
    threshold = float(result["summary"]["threshold_db"])
    units = result.get("units", "mm")

    samples = result["samples"]
    if not samples:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("ALMSOUND under-threshold map")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    xs = np.array([s["point"][0] for s in samples], dtype=float)
    ys = np.array([s["point"][1] for s in samples], dtype=float)
    vals = np.array(
        [np.nan if s["spl_db"] is None else float(s["spl_db"]) for s in samples],
        dtype=float,
    )

    ux = np.unique(xs)
    uy = np.unique(ys)

    nx = len(ux)
    ny = len(uy)

    z = np.full((ny, nx), np.nan, dtype=float)
    x_index = {x: i for i, x in enumerate(ux)}
    y_index = {y: i for i, y in enumerate(uy)}

    for s in samples:
        x, y = s["point"]
        v = s["spl_db"]
        ix = x_index[x]
        iy = y_index[y]
        z[iy, ix] = np.nan if v is None else float(v)

    # 只保留 < threshold 的区域，其余全部透明
    z_under = np.where(z < threshold, z, np.nan)

    fig, ax = plt.subplots(figsize=(12, 8))

    # 欠覆盖区着色
    im = ax.imshow(
        z_under,
        extent=[ux.min(), ux.max(), uy.min(), uy.max()],
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap="Reds_r",
    )

    # 画出 60 dB 分界线
    valid_mask = np.where(np.isnan(z), np.nan, z)
    try:
        ax.contour(
            ux,
            uy,
            valid_mask,
            levels=[threshold],
            colors="red",
            linewidths=1.5,
        )
    except Exception:
        pass

    # boundary
    bx = [p[0] for p in boundary]
    by = [p[1] for p in boundary]
    ax.plot(bx, by, color="black", linewidth=1.2, label="boundary")

    # blockers
    for i, (p1, p2) in enumerate(blockers):
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="dimgray",
            linewidth=0.8,
            label="blockers" if i == 0 else None,
        )

    # openings
    for i, (p1, p2) in enumerate(openings):
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="limegreen",
            linewidth=1.6,
            label="openings" if i == 0 else None,
        )

    # sounders
    if sounders:
        sx = [p[0] for p in sounders]
        sy = [p[1] for p in sounders]
        ax.scatter(
            sx,
            sy,
            marker="*",
            s=90,
            color="dodgerblue",
            edgecolors="black",
            linewidths=0.4,
            label=f"sounders ({len(sounders)})",
            zorder=5,
        )

    s = result["summary"]
    under_cnt = s["grid_point_count"] - s["covered_point_count"]

    info = (
        f"units: {units}\n"
        f"grid points: {s['grid_point_count']}\n"
        f"< {threshold:.0f} dB points: {under_cnt}\n"
        f"coverage: {s['coverage_ratio']:.1%}\n"
        f"sounders: {s['sounder_count']}\n"
        f"blockers/openings: {s['blocker_count']} / {s['opening_count']}"
    )

    ax.text(
        -0.28,
        1.00,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_title(f"ALMSOUND under-threshold map (< {threshold:.0f} dB)")
    ax.set_xlabel(f"X ({units})")
    ax.set_ylabel(f"Y ({units})")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.02, 0.88),
        borderaxespad=0.0,
        framealpha=0.85,
    )
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SPL (dB, only under-threshold area)")

    log_step("正在保存欠覆盖区图片")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="ALMSOUND geometry debug + heatmap")
    parser.add_argument("-i", "--input", default="input.json", help="input json")
    parser.add_argument("-j", "--json", default="result.json", help="result json")
    parser.add_argument("-g", "--geometry", default="geometry_debug.png", help="geometry debug png")
    parser.add_argument("-p", "--plot", default="sound_pressure_heatmap.png", help="heatmap png")
    args = parser.parse_args()

    log_step(f"读取输入文件: {args.input}")
    data = load_json(args.input)

    log_step("开始绘制 geometry_debug 图")
    plot_geometry_debug(data, args.geometry)
    log_step(f"geometry_debug 已保存: {args.geometry}")

    log_step("开始计算声压分布")
    result = analyze(data)

    log_step(f"写出结果 JSON: {args.json}")
    save_json(args.json, result)

    log_step("开始绘制热力图")
    plot_heatmap(data, result, args.plot)
    log_step(f"热力图已保存: {args.plot}")

    s = result["summary"]
    log_step("全部完成")
    print(f"units = {result['units']}")
    print(f"grid_point_count = {s['grid_point_count']}")
    print(f"covered_point_count = {s['covered_point_count']}")
    print(f"coverage_ratio = {s['coverage_ratio']:.2%}")
    print(f"min_spl_db = {s['min_spl_db']}")
    print(f"max_spl_db = {s['max_spl_db']}")
    print(f"sounder_count = {s['sounder_count']}")
    print(f"blocker_count = {s['blocker_count']}")
    print(f"opening_count = {s['opening_count']}")


if __name__ == "__main__":
    main()
