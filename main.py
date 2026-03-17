
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


@dataclass
class Params:
    grid_step: float = 0.5
    source_spl_db_at_1m: float = 90.0
    reference_distance_m: float = 1.0
    wall_loss_db: float = 8.0
    max_distance_m: float = 80.0
    min_distance_m: float = 0.3
    threshold_db: float = 60.0
    opening_snap_tolerance: float = 0.15


def load_input(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pt(v) -> Point:
    return (float(v[0]), float(v[1]))


def seg(v) -> Segment:
    return (pt(v[0]), pt(v[1]))


def as_polygon(boundary: Any) -> List[Point]:
    if isinstance(boundary, list) and boundary and isinstance(boundary[0][0], (int, float)):
        poly = [pt(p) for p in boundary]
    else:
        raise ValueError('boundary must be a single polygon [[x,y], ...]')
    if poly[0] != poly[-1]:
        poly.append(poly[0])
    return poly


def polygon_bbox(poly: List[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly[:-1]]
    ys = [p[1] for p in poly[:-1]]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_polygon(point: Point, poly_path: MplPath) -> bool:
    return bool(poly_path.contains_point(point))


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def mul(a: Point, k: float) -> Point:
    return (a[0] * k, a[1] * k)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def point_segment_distance(p: Point, s: Segment) -> float:
    a, b = s
    ab = sub(b, a)
    ab2 = dot(ab, ab)
    if ab2 == 0:
        return distance(p, a)
    t = clamp(dot(sub(p, a), ab) / ab2, 0.0, 1.0)
    proj = add(a, mul(ab, t))
    return distance(p, proj)


def orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: Point, b: Point, c: Point, eps: float = 1e-9) -> bool:
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps and
        min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps and
        abs(orient(a, b, c)) <= eps
    )


def segments_intersect(s1: Segment, s2: Segment, eps: float = 1e-9) -> bool:
    p1, q1 = s1
    p2, q2 = s2
    o1 = orient(p1, q1, p2)
    o2 = orient(p1, q1, q2)
    o3 = orient(p2, q2, p1)
    o4 = orient(p2, q2, q1)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
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


def intersection_point(s1: Segment, s2: Segment) -> Optional[Point]:
    (x1, y1), (x2, y2) = s1
    (x3, y3), (x4, y4) = s2
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
    p = (px, py)
    if point_segment_distance(p, s1) < 1e-6 and point_segment_distance(p, s2) < 1e-6:
        return p
    return None


def intersection_allowed(inter_pt: Point, openings: List[Segment], tol: float) -> bool:
    for op in openings:
        if point_segment_distance(inter_pt, op) <= tol:
            return True
    return False


def count_blocker_crossings(ray: Segment, blockers: List[Segment], openings: List[Segment], tol: float) -> int:
    count = 0
    for b in blockers:
        if segments_intersect(ray, b):
            ip = intersection_point(ray, b)
            if ip is not None and intersection_allowed(ip, openings, tol):
                continue
            count += 1
    return count


def spl_from_source(distance_m: float, params: Params) -> float:
    d = max(distance_m, params.min_distance_m)
    return params.source_spl_db_at_1m - 20.0 * math.log10(d / params.reference_distance_m)


def sum_db(levels: List[float]) -> Optional[float]:
    vals = [10 ** (l / 10.0) for l in levels if l is not None and math.isfinite(l)]
    if not vals:
        return None
    return 10.0 * math.log10(sum(vals))


def build_grid(poly: List[Point], step: float) -> List[Point]:
    xmin, ymin, xmax, ymax = polygon_bbox(poly)
    poly_path = MplPath(poly)
    pts: List[Point] = []
    y = ymin + step / 2.0
    while y < ymax:
        x = xmin + step / 2.0
        while x < xmax:
            p = (x, y)
            if point_in_polygon(p, poly_path):
                pts.append(p)
            x += step
        y += step
    return pts


def normalize_params(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    rp = dict(raw_params)
    if 'opening_tolerance' in rp and 'opening_snap_tolerance' not in rp:
        rp['opening_snap_tolerance'] = rp.pop('opening_tolerance')
    if 'max_distance' in rp and 'max_distance_m' not in rp:
        rp['max_distance_m'] = rp.pop('max_distance')
    return rp


def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
    params = Params(**normalize_params(data.get('params', {})))
    boundary = as_polygon(data['boundary'])
    blockers = [seg(s) for s in data.get('blockers', [])]
    openings = [seg(s) for s in data.get('openings', [])]
    sounders = [pt(p) for p in data.get('sounder_points', [])]
    if not sounders:
        raise ValueError('sounder_points is empty')

    grid_points = build_grid(boundary, params.grid_step)
    samples = []
    covered = 0

    for gp in grid_points:
        levels = []
        src_details = []
        for sp in sounders:
            d = distance(gp, sp)
            if d > params.max_distance_m:
                continue
            ray = (sp, gp)
            crossings = count_blocker_crossings(ray, blockers, openings, params.opening_snap_tolerance)
            level = spl_from_source(d, params) - crossings * params.wall_loss_db
            levels.append(level)
            src_details.append({
                'source': [sp[0], sp[1]],
                'distance_m': d,
                'blocker_crossings': crossings,
                'spl_db': level,
            })
        total = sum_db(levels)
        is_covered = bool(total is not None and total >= params.threshold_db)
        if is_covered:
            covered += 1
        samples.append({
            'point': [gp[0], gp[1]],
            'spl_db': total,
            'covered': is_covered,
            'sources': src_details,
        })

    totals = [s['spl_db'] for s in samples if s['spl_db'] is not None]
    under = [s['point'] for s in samples if s['spl_db'] is not None and s['spl_db'] < params.threshold_db]

    return {
        'params': params.__dict__,
        'summary': {
            'grid_point_count': len(samples),
            'covered_point_count': covered,
            'coverage_ratio': (covered / len(samples)) if samples else 0.0,
            'threshold_db': params.threshold_db,
            'sounder_count': len(sounders),
            'min_spl_db': min(totals) if totals else None,
            'max_spl_db': max(totals) if totals else None,
            'under_threshold_point_count': len(under),
        },
        'under_threshold_points': under,
        'samples': samples,
    }


def plot_heatmap(data: Dict[str, Any], result: Dict[str, Any], out_png: str):
    boundary = as_polygon(data['boundary'])
    blockers = [seg(s) for s in data.get('blockers', [])]
    openings = [seg(s) for s in data.get('openings', [])]
    sounders = [pt(p) for p in data.get('sounder_points', [])]
    threshold = result['params']['threshold_db']
    cov = result['summary']['coverage_ratio'] * 100.0
    min_spl = result['summary']['min_spl_db']
    max_spl = result['summary']['max_spl_db']

    xs = np.array([s['point'][0] for s in result['samples'] if s['spl_db'] is not None], dtype=float)
    ys = np.array([s['point'][1] for s in result['samples'] if s['spl_db'] is not None], dtype=float)
    cs = np.array([s['spl_db'] for s in result['samples'] if s['spl_db'] is not None], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8))

    if len(xs) > 0:
        filled = ax.tricontourf(xs, ys, cs, levels=18)
        cbar = plt.colorbar(filled, ax=ax)
        cbar.set_label('Sound Pressure Level (dB)')

        all_levels = np.linspace(float(np.nanmin(cs)), float(np.nanmax(cs)), 7)
        if len(np.unique(all_levels)) >= 2:
            ax.tricontour(xs, ys, cs, levels=all_levels, linewidths=0.6)

        if float(np.nanmin(cs)) <= threshold <= float(np.nanmax(cs)):
            threshold_contour = ax.tricontour(xs, ys, cs, levels=[threshold], linewidths=2.0)
            try:
                threshold_contour.collections[0].set_label(f'{threshold:.0f} dB contour')
            except Exception:
                pass

    bx = [p[0] for p in boundary]
    by = [p[1] for p in boundary]
    ax.plot(bx, by, linewidth=2, label='Boundary')

    for i, s in enumerate(blockers):
        ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], linewidth=2.5, label='Blocker' if i == 0 else None)

    for i, s in enumerate(openings):
        ax.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], linewidth=4, label='Opening' if i == 0 else None)

    if sounders:
        ax.scatter([p[0] for p in sounders], [p[1] for p in sounders], marker='*', s=220, label='Sounder')

    under_x = [s['point'][0] for s in result['samples'] if s['spl_db'] is not None and s['spl_db'] < threshold]
    under_y = [s['point'][1] for s in result['samples'] if s['spl_db'] is not None and s['spl_db'] < threshold]
    if under_x:
        ax.scatter(under_x, under_y, facecolors='none', edgecolors='k', s=24, marker='s', label=f'< {threshold:.0f} dB')

    stat_lines = [
        f'Coverage: {cov:.1f}%',
        f'Threshold: {threshold:.0f} dB',
        f'Min/Max: {min_spl:.2f} / {max_spl:.2f} dB' if min_spl is not None and max_spl is not None else 'Min/Max: n/a',
        f'Sounders: {len(sounders)}',
    ]
    ax.text(
        0.02, 0.98, '\n'.join(stat_lines),
        transform=ax.transAxes,
        va='top', ha='left',
        bbox=dict(boxstyle='round', alpha=0.9)
    )

    ax.set_title('ALMSOUND Sound Pressure Heatmap')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (m or drawing units)')
    ax.set_ylabel('Y (m or drawing units)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=240)
    plt.close(fig)


def main():
    input_path = 'input.json'
    output_json = 'result_v2.json'
    output_png = 'sound_pressure_heatmap_v2.png'

    data = load_input(input_path)
    result = analyze(data)
    save_json(output_json, result)
    plot_heatmap(data, result, output_png)
    print(f'Wrote {output_json} and {output_png}')


if __name__ == '__main__':
    main()
