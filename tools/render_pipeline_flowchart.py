"""Render the one-page pipeline workflow flowchart."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUT_DIR = Path("figures")
PNG_OUT = OUT_DIR / "pipeline_workflow_flowchart.png"
PDF_OUT = OUT_DIR / "pipeline_workflow_flowchart.pdf"


COLORS = {
    "start": "#1F4E5F",
    "data": "#E6F2EF",
    "train": "#EAF0FB",
    "validate": "#FFF2CC",
    "cfd": "#F7E6E6",
    "output": "#EAF4E1",
    "stroke": "#263238",
    "arrow": "#455A64",
    "loop": "#8A4F1D",
}


def box(ax, key, x, y, w, h, text, fc, shape="round", fs=9.5):
    if shape == "diamond":
        pts = [(x + w / 2, y + h), (x + w, y + h / 2), (x + w / 2, y), (x, y + h / 2)]
        patch = Polygon(pts, closed=True, facecolor=fc, edgecolor=COLORS["stroke"], linewidth=1.25)
        ax.add_patch(patch)
    else:
        rounding = 0.16 if shape == "round" else 0.04
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.03,rounding_size={rounding}",
            facecolor=fc,
            edgecolor=COLORS["stroke"],
            linewidth=1.25,
        )
        ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111111")
    nodes[key] = (x, y, w, h)


def anchor(name, side):
    x, y, w, h = nodes[name]
    if side == "left":
        return x, y + h / 2
    if side == "right":
        return x + w, y + h / 2
    if side == "top":
        return x + w / 2, y + h
    if side == "bottom":
        return x + w / 2, y
    raise ValueError(side)


def arrow(ax, src, dst, src_side="right", dst_side="left", label="", color=None, rad=0.0, lw=1.4):
    color = color or COLORS["arrow"]
    a = anchor(src, src_side)
    b = anchor(dst, dst_side)
    patch = FancyArrowPatch(
        a,
        b,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        shrinkA=6,
        shrinkB=6,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if label:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx, my + 0.15, label, fontsize=7.5, color=color, ha="center", va="center")


nodes = {}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        8,
        8.55,
        "Chemistry Calibration Pipeline Workflow",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
        color="#102027",
    )
    ax.text(
        8,
        8.18,
        "one public route for data intake, training, validation, CFD coupling, and deployment",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#455A64",
    )

    box(ax, "start", 0.45, 6.95, 1.65, 0.62, "START\nprocess need", COLORS["start"], fs=8.7)
    ax.text(1.28, 6.78, "growth, Ge/C,\ntransfer, WIWNU", ha="center", va="top", fontsize=7.4, color="#455A64")

    box(ax, "scalar", 2.55, 7.0, 2.05, 0.78, "Scalar wafers\nCSV: Si/SiGe/SiGeC", COLORS["data"])
    box(ax, "spatial", 2.55, 5.95, 2.05, 0.78, "Wafer scans\nruns + points CSV", COLORS["data"])
    box(ax, "cfd_in", 2.55, 4.9, 2.05, 0.78, "CFD-ACE+ profiles\ncontract CSV", COLORS["cfd"])
    box(ax, "tomasini", 2.55, 3.85, 2.05, 0.78, "Tomasini baseline\nreproduction data", COLORS["data"])

    box(ax, "register", 5.25, 6.55, 2.12, 0.9, "register_experiment\nschema + manifests", COLORS["data"])
    box(ax, "features", 8.0, 6.55, 2.05, 0.9, "Feature builder\nlegacy 4 + process slots", COLORS["data"])

    box(ax, "route", 10.65, 6.55, 2.0, 0.9, "Workflow router\nintent + class", COLORS["validate"], fs=8.8)

    box(ax, "chem", 9.75, 5.0, 2.12, 0.78, "Chemistry training\npooled / warm start", COLORS["train"])
    box(ax, "class", 12.55, 5.0, 2.15, 0.78, "Class slots\nSiGeC carbon if C%", COLORS["train"])
    box(ax, "transfer", 9.75, 3.9, 2.12, 0.78, "Reactor transfer\nfrozen chemistry", COLORS["train"])
    box(ax, "spatial_fit", 12.55, 3.9, 2.15, 0.78, "Spatial transfer\nradial delta(r)", COLORS["train"])

    box(ax, "export", 5.25, 2.35, 2.12, 0.78, "Export mechanism\nsurface UDF", COLORS["cfd"])
    box(ax, "cfd_run", 8.0, 2.35, 2.05, 0.78, "CFD-ACE+ run\n3D reactor mesh", COLORS["cfd"])
    box(ax, "al", 8.0, 1.2, 2.05, 0.78, "Active learning\nnext CFD batch", COLORS["cfd"])

    box(ax, "validate", 11.15, 2.0, 1.72, 1.02, "Validation\npasses?", COLORS["validate"], shape="diamond", fs=9.2)
    box(ax, "refit", 5.25, 0.9, 2.12, 0.78, "Add data / refine\nthen refit", COLORS["validate"])

    box(ax, "end", 12.85, 1.75, 2.7, 1.0, "END\nvalidated posteriors,\nreports, recipes,\ndecisions", COLORS["output"], fs=9.0)

    arrow(ax, "start", "scalar")
    arrow(ax, "start", "spatial", src_side="right", dst_side="left", rad=-0.08)
    arrow(ax, "start", "cfd_in", src_side="right", dst_side="left", rad=-0.15)
    arrow(ax, "start", "tomasini", src_side="right", dst_side="left", rad=-0.22)

    arrow(ax, "scalar", "register")
    arrow(ax, "spatial", "register", src_side="right", dst_side="left", rad=-0.08)
    arrow(ax, "cfd_in", "register", src_side="right", dst_side="left", rad=-0.16)
    arrow(ax, "tomasini", "register", src_side="right", dst_side="left", rad=-0.22)
    arrow(ax, "register", "features")
    arrow(ax, "features", "route")

    arrow(ax, "route", "chem", src_side="bottom", dst_side="top", label="chemistry", rad=0.04)
    arrow(ax, "route", "transfer", src_side="bottom", dst_side="top", label="reactor", rad=-0.12)
    arrow(ax, "route", "spatial_fit", src_side="right", dst_side="top", label="spatial", rad=-0.18)
    arrow(ax, "chem", "class", label="class gate")
    arrow(ax, "transfer", "spatial_fit", label="if scan")

    arrow(ax, "chem", "validate", src_side="bottom", dst_side="top", rad=-0.28)
    arrow(ax, "class", "validate", src_side="bottom", dst_side="top", rad=0.08)
    arrow(ax, "transfer", "validate", src_side="bottom", dst_side="top", rad=0.02)
    arrow(ax, "spatial_fit", "validate", src_side="bottom", dst_side="top", rad=-0.08)

    arrow(ax, "validate", "end", src_side="right", dst_side="left")
    arrow(ax, "validate", "refit", src_side="bottom", dst_side="right", color=COLORS["loop"], rad=0.22)
    ax.text(13.1, 2.62, "YES", fontsize=7.6, color=COLORS["arrow"], ha="center", va="center")
    ax.text(9.4, 1.35, "NO", fontsize=7.6, color=COLORS["loop"], ha="center", va="center")
    arrow(ax, "refit", "register", src_side="top", dst_side="bottom", color=COLORS["loop"], rad=-0.12)

    arrow(ax, "chem", "export", src_side="left", dst_side="top", label="validated chemistry", rad=0.05)
    arrow(ax, "export", "cfd_run")
    arrow(ax, "cfd_run", "cfd_in", src_side="top", dst_side="bottom", label="profiles", color=COLORS["loop"], rad=-0.3)
    arrow(ax, "cfd_run", "al", src_side="bottom", dst_side="top")
    arrow(ax, "al", "cfd_run", src_side="top", dst_side="bottom", color=COLORS["loop"], rad=0.45)

    ax.text(6.3, 7.72, "Single intake facade", ha="center", va="center", fontsize=8, color="#00695C")
    ax.text(12.6, 5.92, "Class-gated models", ha="center", va="center", fontsize=8, color="#1A4F8B")
    ax.text(7.65, 3.45, "Expensive reactor physics loop", ha="center", va="center", fontsize=8, color="#8A4F1D")

    plt.tight_layout(pad=0.25)
    fig.savefig(PNG_OUT, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(PNG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
