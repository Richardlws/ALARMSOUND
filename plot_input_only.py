import json
import matplotlib.pyplot as plt


def load_input(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_input_only(data: dict, output_path: str = "input_plot.png") -> None:
    boundary = [tuple(p) for p in data.get("boundary", [])]
    alarm_points = [tuple(p) for p in data.get("alarm_points", [])]
    blockers = [(tuple(seg[0]), tuple(seg[1])) for seg in data.get("blockers", [])]
    openings = [(tuple(seg[0]), tuple(seg[1])) for seg in data.get("openings", [])]

    fig, ax = plt.subplots(figsize=(12, 9))

    # Boundary
    if boundary:
        bx = [p[0] for p in boundary]
        by = [p[1] for p in boundary]
        ax.plot(
            bx, by,
            color="gold",
            linewidth=2.2,
            label=f"Boundary ({len(boundary)} pts)"
        )

    # Blockers
    if blockers:
        for i, seg in enumerate(blockers):
            (x1, y1), (x2, y2) = seg
            if i == 0:
                ax.plot(
                    [x1, x2], [y1, y2],
                    color="purple",
                    linewidth=1.2,
                    label=f"Blockers ({len(blockers)} segs)"
                )
            else:
                ax.plot(
                    [x1, x2], [y1, y2],
                    color="purple",
                    linewidth=1.2
                )

    # Openings
    if openings:
        for i, seg in enumerate(openings):
            (x1, y1), (x2, y2) = seg
            if i == 0:
                ax.plot(
                    [x1, x2], [y1, y2],
                    color="deepskyblue",
                    linewidth=2.0,
                    label=f"Openings ({len(openings)} segs)"
                )
            else:
                ax.plot(
                    [x1, x2], [y1, y2],
                    color="deepskyblue",
                    linewidth=2.0
                )

    # Alarm points
    if alarm_points:
        ax.scatter(
            [p[0] for p in alarm_points],
            [p[1] for p in alarm_points],
            marker="s",
            s=45,
            color="limegreen",
            label=f"Alarm points ({len(alarm_points)})",
            zorder=5
        )

    ax.set_title("ALMCHECK Input Geometry Preview")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    input_path = "input.json"
    output_path = "input_plot.png"

    data = load_input(input_path)
    plot_input_only(data, output_path)
    print(f"Input preview saved to: {output_path}")


if __name__ == "__main__":
    main()