"""Restore the submitted figure motifs using the archived scientific estimates."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image


def make(assets, csv, out, annual_models, paired_contrasts):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / (name + "." + ext), dpi=260, facecolor="white")
        plt.close(fig)

    def crop(name, rect):
        im = Image.open(assets / (name + ".png")).convert("RGB")
        return im.crop(
            tuple(
                round(v * (im.width if j % 2 == 0 else im.height))
                for j, v in enumerate(rect)
            )
        )

    def canvas(size):
        f = plt.figure(figsize=size)
        a = f.add_axes([0, 0, 1, 1])
        a.set(xlim=(0, 1), ylim=(0, 1))
        a.axis("off")
        return f, a

    def motif(fig, rect, img):
        ax = fig.add_axes(rect)
        ax.imshow(img)
        ax.axis("off")

    f, a = canvas((8.2, 4.2))
    for j, (title, lines, color, rect) in enumerate(
        [
            (
                "Data sources",
                "158 country or regional means\n143 with GDP records",
                "#daeffa",
                (0.017, 0.272, 0.23, 0.422),
            ),
            (
                "Between countries",
                "16 foods × 3 outcomes\n1992–2014 common window",
                "#eaf1d2",
                (0.312, 0.268, 0.43, 0.425),
            ),
            (
                "Covariate adjustment",
                "Log GDP and total energy\nAll 48 pairs enter each screen",
                "#f0dff8",
                (0.565, 0.268, 0.69, 0.425),
            ),
            (
                "Within countries",
                "Country and year effects\nDifferences and trend models",
                "#f3ddda",
                None,
            ),
        ]
    ):
        left = 0.012 + 0.247 * j
        a.add_patch(
            FancyBboxPatch(
                (left, 0.77),
                0.23,
                0.17,
                boxstyle="round,pad=.004,rounding_size=.023",
                facecolor=color,
                edgecolor="#7c91b0",
            )
        )
        a.text(
            left + 0.115,
            0.855,
            title,
            ha="center",
            va="center",
            fontsize=10.6,
            weight="bold",
        )
        if rect:
            motif(f, [left + 0.04, 0.39, 0.15, 0.30], crop("flow", rect))
        else:
            a.plot([left + 0.052, left + 0.052], [0.44, 0.66], color="#444", lw=1)
            a.plot([left + 0.052, left + 0.185], [0.44, 0.44], color="#444", lw=1)
            for j2, ys in enumerate(
                [
                    [0.47, 0.51, 0.51, 0.54],
                    [0.53, 0.55, 0.58, 0.60],
                    [0.51, 0.49, 0.53, 0.52],
                ]
            ):
                a.plot(
                    np.linspace(left + 0.06, left + 0.18, 4),
                    ys,
                    color=["#4e84ca", "#bd6266", "#6c9273"][j2],
                    lw=1.3,
                )
        a.text(
            left + 0.115,
            0.30,
            lines,
            ha="center",
            va="center",
            fontsize=9,
            linespacing=1.55,
        )
    a.text(
        0.5,
        0.095,
        "Report coefficients, 95% intervals, country counts and BH-adjusted P values for each model.",
        ha="center",
        fontsize=10.4,
    )
    save(f, "Figure_1")
    cross = pd.read_csv(csv).query(
        "specification == 'updated_common_window_logGDP_energy'"
    )
    panel = pd.read_csv(annual_models).query("specification == 'TWFE_logGDP_energy'")
    paired = pd.read_csv(paired_contrasts).query("omitted_region == 'Full sample'")
    keys = ["exposure", "outcome"]
    screens = [frame.set_index(keys).sort_index() for frame in (cross, panel, paired)]
    for screen in screens:
        if (
            len(screen) != 48
            or not screen.index.is_unique
            or not screen.index.equals(screens[0].index)
        ):
            raise ValueError(
                "The graphical abstract requires the same 48 food–outcome pairs in every screen."
            )
    across_count, within_count, paired_count = [
        int(screen.q_BH.lt(0.05).sum()) for screen in screens
    ]
    f, a = canvas((8.2, 3.2))
    for x, title in [
        (0.012, "National indicators"),
        (0.34, "Across countries"),
        (0.666, "Within countries"),
    ]:
        a.add_patch(
            Rectangle(
                (x, 0.17), 0.318, 0.70, facecolor="none", edgecolor="#869cc6", lw=1.05
            )
        )
        a.text(x + 0.159, 0.925, title, ha="center", fontsize=13)
    motif(f, [0.035, 0.21, 0.27, 0.59], crop("abstract", (0.008, 0.061, 0.337, 0.378)))
    a.text(
        0.5,
        0.59,
        f"{across_count} of 48 pairs",
        ha="center",
        fontsize=19,
        color="#375f99",
    )
    a.text(
        0.5,
        0.36,
        "met BH q < 0.05\nwith log GDP and total energy",
        ha="center",
        fontsize=10.6,
        linespacing=1.5,
    )
    a.text(
        0.825,
        0.59,
        f"{within_count} of 48 pairs",
        ha="center",
        fontsize=19,
        color="#98514d",
    )
    a.text(
        0.825,
        0.36,
        "met BH q < 0.05 with\ncountry and year fixed effects",
        ha="center",
        fontsize=10.6,
        linespacing=1.5,
    )
    a.text(
        0.5,
        0.065,
        f"1992–2014   ·   {paired_count} of 48 within-minus-between slope differences met BH q < 0.05.",
        ha="center",
        fontsize=11,
    )
    save(f, "Graphical_abstract")
    # Retain the author's database illustrations; the two arrows describe prospective steps.
    f, a = canvas((8.2, 2.0))
    for x, rect, label in [
        (0.018, (0.046, 0.030, 0.249, 0.216), "Country-level indicators"),
        (0.365, (0.407, 0.029, 0.614, 0.213), "Within-database estimates"),
        (0.708, (0.765, 0.034, 0.977, 0.218), "Cross-database comparison"),
    ]:
        motif(f, [x, 0.22, 0.263, 0.72], crop("network", rect))
        a.text(x + 0.131, 0.092, label, fontsize=10.6, ha="center")
    for x in (0.285, 0.63):
        a.annotate(
            "",
            xy=(x + 0.065, 0.55),
            xytext=(x, 0.55),
            arrowprops={"arrowstyle": "-|>", "color": "#7baaa1", "lw": 2},
        )
    a.add_patch(
        Rectangle((0.005, 0.02), 0.99, 0.96, fill=False, edgecolor="#96aacf", lw=1.5)
    )
    save(f, "Figure_5")
    g = pd.read_csv(csv)
    g = g[g.specification == "updated_common_window_logGDP_energy"]
    labs = [
        "Red meat",
        "Alcohol",
        "Animal fats",
        "Milk",
        "Eggs",
        "Sugar and\nsweeteners",
        "Vegetable oils",
        "Nuts",
        "Poultry",
        "Vegetables",
        "Pulses",
        "Complete\ncereals",
        "Fish and\nseafood",
        "Starchy roots",
        "Fruit",
        "Other meat",
    ]
    names = [
        "Red meat",
        "Alcoholic Beverages",
        "Animal fats group",
        "Milk",
        "Eggs",
        "Sugar and Sweeteners",
        "Vegetable Oils",
        "Nuts",
        "Poultry meat",
        "Vegetables",
        "Pulses",
        "Cereals",
        "Fish and seafood",
        "Starchy Roots",
        "Fruit",
        "Other meat",
    ]
    for number, outcome in enumerate(
        ["Non-HDL", "Diabetes1992-2014", "Hypertension"], 2
    ):
        z = g[g.outcome == outcome].set_index("exposure").loc[names]
        assert len(z) == 16
        f = plt.figure(figsize=(7.5, 5.85))
        ax = f.add_axes([0.20, 0.145, 0.60, 0.72], projection="polar")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        theta = np.arange(16) * np.pi / 8
        rr = z.r.to_numpy() + 1
        ax.set_ylim(0, 2)
        ax.set_yticks(
            [0.5, 1, 1.5, 2], ["−0.5", "0", "0.5", "1.0"], fontsize=8, color="#333"
        )
        ax.set_rlabel_position(11.25)
        ax.set_xticks(theta, labs, fontsize=10)
        ax.tick_params(axis="x", pad=12)
        for label, t in zip(ax.get_xticklabels(), theta):
            label.set_ha(
                "left"
                if 0.1 < t < np.pi - 0.1
                else "right" if t > np.pi + 0.1 else "center"
            )
        ax.grid(linewidth=0.6, color="#ccd2d9")
        ax.spines["polar"].set_color("#bdc5d1")
        ax.plot(
            np.linspace(0, np.pi * 2, 240), np.ones(240), color="#555", lw=0.9, ls="--"
        )
        ax.plot(np.r_[theta, theta[0]], np.r_[rr, rr[0]], color="#387dcc", lw=1.65)
        ax.scatter(
            theta,
            rr,
            s=27,
            facecolors=["#387dcc" if v < 0.05 else "white" for v in z.q_BH],
            edgecolors="#387dcc",
            zorder=5,
        )
        f.text(
            0.5,
            0.016,
            "Dashed ring = zero correlation     •  Filled point = BH q < 0.05",
            ha="center",
            fontsize=10,
        )
        save(f, f"Figure_{number}a")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    root = ap.parse_args().root
    make(
        root / "artwork_sources",
        root / "annual_results/reconstructed_cross_sectional.csv",
        root / "figures",
        root / "annual_results/annual_models.csv",
        root / "comparison/paired_slope_contrasts.csv",
    )
    # Reuse the archived numerical plotting code for the annual and supplementary panels.
    from tempfile import TemporaryDirectory
    from shutil import copy2
    from figures import run as build_estimate_panels

    with TemporaryDirectory() as temp:
        temp = Path(temp)
        build_estimate_panels(
            root / "results", temp, root / "annual_results", root / "diagnostics"
        )
        for source, target in [
            ("Figure_4a_within_country", "Figure_2b"),
            ("Figure_4b_within_country", "Figure_3b"),
            ("Figure_4c_within_country", "Figure_4b"),
            ("Figure_2_all_associations", "Supplementary_Figure_A1"),
            ("Figure_3_model_sensitivity", "Supplementary_Figure_A2"),
        ]:
            for ext in ["png", "pdf"]:
                copy2(
                    temp / (source + "." + ext), root / "figures" / (target + "." + ext)
                )
